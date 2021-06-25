# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import recursive_tile
from ...core.custom_log import redirect_custom_log
from ...serialization.serializables import KeyField, FunctionField, \
    TupleField, DictField, BoolField
from ...utils import enter_current_session, has_unknown_shape, quiet_stdio
from ..operands import DataFrameOperand, DataFrameOperandMixin, OutputType
from ..utils import build_df, build_empty_df, build_series, parse_index, \
    validate_output_types


class DataFrameMapChunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.MAP_CHUNK

    _input = KeyField('input')
    _func = FunctionField('func')
    _args = TupleField('args')
    _kwargs = DictField('kwargs')
    _with_chunk_index = BoolField('with_chunk_index')

    def __init__(self, input=None, func=None, args=None, kwargs=None, output_types=None,
                 with_chunk_index=None, **kw):
        super().__init__(_input=input, _func=func, _args=args, _kwargs=kwargs,
                         _output_types=output_types, _with_chunk_index=with_chunk_index, **kw)

    @property
    def input(self):
        return self._input

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def with_chunk_index(self):
        return self._with_chunk_index

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, df_or_series, index=None, dtypes=None):
        test_obj = build_df(df_or_series, size=2) \
            if df_or_series.ndim == 2 else \
            build_series(df_or_series, size=2, name=df_or_series.name)
        output_type = self._output_types[0] if self.output_types else None

        # try run to infer meta
        try:
            kwargs = self.kwargs or dict()
            if self.with_chunk_index:
                kwargs['chunk_index'] = (0,) * df_or_series.ndim
            with np.errstate(all='ignore'), quiet_stdio():
                obj = self._func(test_obj, *self._args, **kwargs)
        except:  # noqa: E722  # nosec
            if df_or_series.ndim == 1 or output_type == OutputType.series:
                obj = pd.Series([], dtype=np.dtype(object))
            elif output_type == OutputType.dataframe and dtypes is not None:
                obj = build_empty_df(dtypes)
            else:
                raise TypeError('Cannot determine `output_type`, '
                                'you have to specify it as `dataframe` or `series`, '
                                'for dataframe, `dtypes` is required as well '
                                'if output_type=\'dataframe\'')

        if getattr(obj, 'ndim', 0) == 1 or output_type == OutputType.series:
            shape = self._kwargs.pop('shape', None)
            if shape is None:
                # series
                if obj.shape == test_obj.shape:
                    shape = df_or_series.shape
                else:
                    shape = (np.nan,)
            if index is None:
                index = obj.index
            index_value = parse_index(index, df_or_series,
                                      self._func, self._args, self._kwargs)
            return self.new_series([df_or_series], dtype=obj.dtype,
                                   shape=shape, index_value=index_value,
                                   name=obj.name)
        else:
            dtypes = dtypes if dtypes is not None else obj.dtypes
            # dataframe
            if obj.shape == test_obj.shape:
                shape = (df_or_series.shape[0], len(dtypes))
            else:
                shape = (np.nan, len(dtypes))
            columns_value = parse_index(dtypes.index, store_data=True)
            if index is None:
                index = obj.index
            index_value = parse_index(index, df_or_series,
                                      self._func, self._args, self._kwargs)
            return self.new_dataframe([df_or_series], shape=shape,
                                      dtypes=dtypes, index_value=index_value,
                                      columns_value=columns_value)

    @classmethod
    def tile(cls, op: "DataFrameMapChunk"):
        inp = op.input
        out = op.outputs[0]

        if inp.ndim == 2 and inp.chunk_shape[1] > 1:
            if has_unknown_shape(inp):
                yield
            # if input is a DataFrame, make sure 1 chunk on axis columns
            inp = yield from recursive_tile(inp.rechunk({1: inp.shape[1]}))

        out_chunks = []
        nsplits = [[]] if out.ndim == 1 else [[], [out.shape[1]]]
        for chunk in inp.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op.tileable_op_key = op.key
            if op.output_types[0] == OutputType.dataframe:
                if np.isnan(out.shape[0]):
                    shape = (np.nan, out.shape[1])
                else:
                    shape = (chunk.shape[0], out.shape[1])
                index_value = parse_index(out.index_value.to_pandas(), chunk,
                                          op.func, op.args, op.kwargs)
                out_chunk = chunk_op.new_chunk([chunk], shape=shape,
                                               dtypes=out.dtypes,
                                               index_value=index_value,
                                               columns_value=out.columns_value,
                                               index=(chunk.index[0], 0))
                out_chunks.append(out_chunk)
                nsplits[0].append(out_chunk.shape[0])
            else:
                if np.isnan(out.shape[0]):
                    shape = (np.nan,)
                else:
                    shape = (chunk.shape[0],)
                index_value = parse_index(out.index_value.to_pandas(), chunk,
                                          op.func, op.args, op.kwargs)
                out_chunk = chunk_op.new_chunk([chunk], shape=shape,
                                               index_value=index_value,
                                               name=out.name,
                                               dtype=out.dtype,
                                               index=(chunk.index[0],))
                out_chunks.append(out_chunk)
                nsplits[0].append(out_chunk.shape[0])

        params = out.params
        params['nsplits'] = tuple(tuple(ns) for ns in nsplits)
        params['chunks'] = out_chunks
        new_op = op.copy()
        return new_op.new_tileables(op.inputs, kws=[params])

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op: "DataFrameMapChunk"):
        inp = ctx[op.input.key]
        out_chunk = op.outputs[0]

        kwargs = op.kwargs or dict()
        if op.with_chunk_index:
            kwargs['chunk_index'] = out_chunk.index
        ctx[out_chunk.key] = op.func(inp, *op.args, **kwargs)


def map_chunk(df_or_series, func, args=(), **kwargs):
    """
    Apply function to each chunk.

    Parameters
    ----------
    func : function
        Function to apply to each chunk.
    args : tuple
        Positional arguments to pass to func in addition to the array/series.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to func.

    Returns
    -------
    Series or DataFrame
        Result of applying ``func`` to each chunk of the DataFrame or Series.

    See Also
    --------
    DataFrame.apply : Perform any type of operations.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df.execute()
       A  B
    0  4  9
    1  4  9
    2  4  9

    Output type including Series or DataFrame will be auto inferred.

    >>> df.map_chunk(lambda c: c['A'] + c['B']).execute()
    0    13
    1    13
    2    13
    dtype: int64

    You can specify ``output_type`` by yourself if auto infer failed.

    >>> import pandas as pd
    >>> import numpy as np
    >>> df['c'] = ['s1', 's2', 's3']
    >>> df.map_chunk(lambda c: pd.concat([c['A'], c['c'].str.slice(1).astype(int)], axis=1)).execute()
    Traceback (most recent call last):
    TypeError: Cannot determine `output_type`, you have to specify it as `dataframe` or `series`...
    >>> df.map_chunk(lambda c: pd.concat([c['A'], c['c'].str.slice(1).astype(int)], axis=1),
    >>>              output_type='dataframe', dtypes=pd.Series([np.dtype(object), np.dtype(int)])).execute()
       A  c
    0  4  1
    1  4  2
    2  4  3
    """
    output_type = kwargs.pop('output_type', None)
    output_types = kwargs.pop('output_types', None)
    object_type = kwargs.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else None
    if output_type:
        output_types = [output_type]
    index = kwargs.pop('index', None)
    dtypes = kwargs.pop('dtypes', None)
    with_chunk_index = kwargs.pop('with_chunk_index', False)

    op = DataFrameMapChunk(input=df_or_series, func=func, args=args, kwargs=kwargs,
                           output_types=output_types, with_chunk_index=with_chunk_index)
    return op(df_or_series, index=index, dtypes=dtypes)
