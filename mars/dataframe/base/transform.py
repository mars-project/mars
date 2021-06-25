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
from ...config import options
from ...core import OutputType, recursive_tile
from ...core.custom_log import redirect_custom_log
from ...lib.version import parse as parse_version
from ...serialization.serializables import AnyField, BoolField, \
    TupleField, DictField
from ...utils import enter_current_session, quiet_stdio
from ..core import DATAFRAME_CHUNK_TYPE, DATAFRAME_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_df, build_series, validate_axis, \
    parse_index, filter_dtypes_by_index, make_dtypes


class TransformOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM

    _func = AnyField('func')
    _axis = AnyField('axis')
    _convert_dtype = BoolField('convert_dtype')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    _call_agg = BoolField('call_agg')

    def __init__(self, func=None, axis=None, convert_dtype=None, args=None, kwds=None,
                 call_agg=None, output_types=None, memory_scale=None, **kw):
        super().__init__(_func=func, _axis=axis, _convert_dtype=convert_dtype, _args=args,
                         _kwds=kwds, _call_agg=call_agg, _output_types=output_types,
                         _memory_scale=memory_scale, **kw)

    @property
    def func(self):
        return self._func

    @property
    def convert_dtype(self):
        return self._convert_dtype

    @property
    def axis(self):
        return self._axis

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @property
    def call_agg(self):
        return self._call_agg

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        out_chunk = op.outputs[0]

        if op.call_agg:
            result = in_data.agg(op.func, axis=op.axis, *op.args, **op.kwds)
        else:
            result = in_data.transform(op.func, axis=op.axis, *op.args, **op.kwds)

        if isinstance(out_chunk, DATAFRAME_CHUNK_TYPE):
            result.columns = out_chunk.dtypes.index
        ctx[op.outputs[0].key] = result

    @classmethod
    def tile(cls, op: "TransformOperand"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        axis = op.axis

        if isinstance(in_df, DATAFRAME_TYPE):
            if in_df.chunk_shape[axis] > 1:
                chunk_size = (in_df.shape[axis],
                              max(1, options.chunk_store_limit // in_df.shape[axis]))
                if axis == 1:
                    chunk_size = chunk_size[::-1]
                in_df = yield from recursive_tile(in_df.rechunk(chunk_size))
        elif isinstance(op.func, str) or \
                 (isinstance(op.func, list) and any(isinstance(e, str) for e in op.func)):
            # builtin cols handles whole columns, thus merge is needed
            if in_df.chunk_shape[0] > 1:
                in_df = yield from recursive_tile(in_df.rechunk((in_df.shape[axis],)))

        chunks = []
        axis_index_map = dict()
        col_sizes = []
        for c in in_df.chunks:
            new_op = op.copy().reset_key()
            new_op.tileable_op_key = op.key
            params = c.params.copy()

            if out_df.ndim == 2:
                if isinstance(c, DATAFRAME_CHUNK_TYPE):
                    columns = c.columns_value.to_pandas()
                    new_dtypes = filter_dtypes_by_index(out_df.dtypes, columns)

                    if len(new_dtypes) == 0:
                        continue
                    if c.index[0] == 0:
                        col_sizes.append(len(new_dtypes))

                    new_index = list(c.index)
                    try:
                        new_index[1 - op.axis] = axis_index_map[c.index[1 - op.axis]]
                    except KeyError:
                        new_index[1 - op.axis] = axis_index_map[c.index[1 - op.axis]] = len(axis_index_map)

                    if isinstance(op.func, dict):
                        new_op._func = dict((k, v) for k, v in op.func.items() if k in new_dtypes)

                    new_shape = list(c.shape)
                    new_shape[1] = len(new_dtypes)

                    if op.call_agg:
                        new_shape[op.axis] = np.nan
                        params['index_value'] = parse_index(
                            None, c.key, c.index_value.key)
                    new_columns_value = parse_index(new_dtypes.index)
                else:
                    new_dtypes = out_df.dtypes
                    new_index = c.index + (0,)
                    new_shape = [c.shape[0], len(new_dtypes)]
                    if op.call_agg:
                        new_shape[0] = np.nan
                    if c.index[0] == 0:
                        col_sizes.append(len(new_dtypes))
                    new_columns_value = out_df.columns_value
                params.update(dict(dtypes=new_dtypes, shape=tuple(new_shape), index=tuple(new_index),
                                   columns_value=new_columns_value))
            else:
                params['dtype'] = out_df.dtype
                if isinstance(in_df, DATAFRAME_TYPE):
                    params.pop('columns_value', None)
                    params['index_value'] = out_df.index_value
                    params['shape'] = (c.shape[1 - op.axis],)
                    params['index'] = (c.index[1 - op.axis],)
            chunks.append(new_op.new_chunk([c], **params))

        if out_df.ndim == 2:
            new_nsplits = [in_df.nsplits[0], tuple(col_sizes)]
            if op.call_agg:
                new_nsplits[op.axis] = (np.nan,)
        elif op.call_agg:
            if isinstance(in_df, DATAFRAME_TYPE):
                new_nsplits = (in_df.nsplits[1],)
            else:
                new_nsplits = ((np.nan,),)
        else:
            new_nsplits = in_df.nsplits

        new_op = op.copy()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=tuple(new_nsplits)))
        return new_op.new_tileables(op.inputs, **kw)

    def _infer_df_func_returns(self, df, dtypes):
        if self.output_types[0] == OutputType.dataframe:
            test_df = build_df(df, fill_value=1, size=2)
            try:
                with np.errstate(all='ignore'), quiet_stdio():
                    if self.call_agg:
                        infer_df = test_df.agg(self._func, axis=self._axis, *self.args, **self.kwds)
                    else:
                        infer_df = test_df.transform(self._func, axis=self._axis, *self.args, **self.kwds)
            except:  # noqa: E722
                infer_df = None
        else:
            test_df = build_series(df, size=2, name=df.name)
            try:
                with np.errstate(all='ignore'), quiet_stdio():
                    if self.call_agg:
                        infer_df = test_df.agg(self._func, args=self.args, **self.kwds)
                    else:
                        if parse_version(pd.__version__) >= parse_version('1.2.0'):
                            infer_df = test_df.transform(self._func, *self.args, **self.kwds)
                        else:  # pragma: no cover
                            infer_df = test_df.transform(self._func, convert_dtype=self.convert_dtype,
                                                         args=self.args, **self.kwds)
            except:  # noqa: E722
                infer_df = None

        if infer_df is None and dtypes is None:
            raise TypeError('Failed to infer dtype, please specify dtypes as arguments.')

        if infer_df is None:
            is_df = self.output_types[0] == OutputType.dataframe
        else:
            is_df = isinstance(infer_df, pd.DataFrame)

        if is_df:
            new_dtypes = make_dtypes(dtypes) if dtypes is not None else infer_df.dtypes
            self.output_types = [OutputType.dataframe]
        else:
            new_dtypes = dtypes if dtypes is not None else (infer_df.name, infer_df.dtype)
            self.output_types = [OutputType.series]

        return new_dtypes

    def __call__(self, df, dtypes=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        self._axis = validate_axis(axis, df)

        dtypes = self._infer_df_func_returns(df, dtypes)

        if self.output_types[0] == OutputType.dataframe:
            new_shape = list(df.shape)
            new_index_value = df.index_value
            if len(new_shape) == 1:
                new_shape.append(len(dtypes))
            else:
                new_shape[1] = len(dtypes)

            if self.call_agg:
                new_shape[self.axis] = np.nan
                new_index_value = parse_index(None, (df.key, df.index_value.key))
            return self.new_dataframe([df], shape=tuple(new_shape), dtypes=dtypes, index_value=new_index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes

            if isinstance(df, DATAFRAME_TYPE):
                new_shape = (df.shape[1 - axis],)
                new_index_value = [df.columns_value, df.index_value][axis]
            else:
                new_shape = (np.nan,) if self.call_agg else df.shape
                new_index_value = df.index_value

            return self.new_series([df], shape=new_shape, name=name, dtype=dtype, index_value=new_index_value)


def df_transform(df, func, axis=0, *args, dtypes=None, **kwargs):
    """
    Call ``func`` on self producing a DataFrame with transformed values.

    Produced DataFrame will have same axis length as self.

    Parameters
    ----------
    func : function, str, list or dict
        Function to use for transforming the data. If a function, must either
        work when passed a DataFrame or when passed to DataFrame.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
        - dict of axis labels -> functions, function names or list of such.
    axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or 'index': apply function to each column.
            If 1 or 'columns': apply function to each row.

    dtypes : Series, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    *args
        Positional arguments to pass to `func`.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    DataFrame
        A DataFrame that must have the same length as self.

    Raises
    ------
    ValueError : If the returned DataFrame has a different length than self.

    See Also
    --------
    DataFrame.agg : Only perform aggregating type operations.
    DataFrame.apply : Invoke function on a DataFrame.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock DataFrame and the apply call may
    fail. When this happens, you need to specify a list or a pandas
    Series as ``dtypes`` of output DataFrame.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': range(3), 'B': range(1, 4)})
    >>> df.execute()
       A  B
    0  0  1
    1  1  2
    2  2  3
    >>> df.transform(lambda x: x + 1).execute()
       A  B
    0  1  2
    1  2  3
    2  3  4

    Even though the resulting DataFrame must have the same length as the
    input DataFrame, it is possible to provide several input functions:

    >>> s = md.Series(range(3))
    >>> s.execute()
    0    0
    1    1
    2    2
    dtype: int64
    >>> s.transform([mt.sqrt, mt.exp]).execute()
           sqrt        exp
    0  0.000000   1.000000
    1  1.000000   2.718282
    2  1.414214   7.389056
    """
    op = TransformOperand(func=func, axis=axis, args=args, kwds=kwargs, output_types=[OutputType.dataframe],
                          call_agg=kwargs.pop('_call_agg', False))
    return op(df, dtypes=dtypes)


def series_transform(series, func, convert_dtype=True, axis=0, *args, dtype=None, **kwargs):
    """
    Call ``func`` on self producing a Series with transformed values.

    Produced Series will have same axis length as self.

    Parameters
    ----------
    func : function, str, list or dict
    Function to use for transforming the data. If a function, must either
    work when passed a Series or when passed to Series.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.exp. 'sqrt']``
    - dict of axis labels -> functions, function names or list of such.
    axis : {0 or 'index'}
        Parameter needed for compatibility with DataFrame.

    dtype : numpy.dtype, default None
        Specify dtypes of returned DataFrames. See `Notes` for more details.

    *args
        Positional arguments to pass to `func`.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    Series
    A Series that must have the same length as self.

    Raises
    ------
    ValueError : If the returned Series has a different length than self.

    See Also
    --------
    Series.agg : Only perform aggregating type operations.
    Series.apply : Invoke function on a Series.

    Notes
    -----
    When deciding output dtypes and shape of the return value, Mars will
    try applying ``func`` onto a mock Series, and the transform call may
    fail. When this happens, you need to specify ``dtype`` of output
    Series.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': range(3), 'B': range(1, 4)})
    >>> df.execute()
    A  B
    0  0  1
    1  1  2
    2  2  3
    >>> df.transform(lambda x: x + 1).execute()
    A  B
    0  1  2
    1  2  3
    2  3  4

    Even though the resulting Series must have the same length as the
    input Series, it is possible to provide several input functions:

    >>> s = md.Series(range(3))
    >>> s.execute()
    0    0
    1    1
    2    2
    dtype: int64
    >>> s.transform([mt.sqrt, mt.exp]).execute()
       sqrt        exp
    0  0.000000   1.000000
    1  1.000000   2.718282
    2  1.414214   7.389056
   """
    op = TransformOperand(func=func, axis=axis, convert_dtype=convert_dtype, args=args, kwds=kwargs,
                          output_types=[OutputType.series], call_agg=kwargs.pop('_call_agg', False))
    dtypes = (series.name, dtype) if dtype is not None else None
    return op(series, dtypes=dtypes)
