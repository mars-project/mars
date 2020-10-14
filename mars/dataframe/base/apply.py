# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import inspect

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...custom_log import redirect_custom_log
from ...serialize import StringField, AnyField, BoolField, \
    TupleField, DictField, FunctionField
from ...utils import enter_current_session
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_df, build_series, parse_index, validate_axis, \
    validate_output_types


class ApplyOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLY

    _func = FunctionField('func')
    _axis = AnyField('axis')
    _convert_dtype = BoolField('convert_dtype')
    _raw = BoolField('raw')
    _result_type = StringField('result_type')
    _elementwise = BoolField('elementwise')
    _args = TupleField('args')
    _kwds = DictField('kwds')
    # for chunk
    _tileable_op_key = StringField('tileable_op_key')

    def __init__(self, func=None, axis=None, convert_dtype=None, raw=None, result_type=None,
                 args=None, kwds=None, output_type=None, elementwise=None,
                 tileable_op_key=None, **kw):
        if output_type:
            kw['_output_types'] = [output_type]
        super().__init__(_func=func, _axis=axis, _convert_dtype=convert_dtype, _raw=raw,
                         _result_type=result_type, _args=args, _kwds=kwds,
                         _elementwise=elementwise, _tileable_op_key=tileable_op_key, **kw)

    @property
    def func(self):
        return self._func

    @property
    def axis(self):
        return self._axis

    @property
    def convert_dtype(self):
        return self._convert_dtype

    @property
    def raw(self):
        return self._raw

    @property
    def result_type(self):
        return self._result_type

    @property
    def elementwise(self):
        return self._elementwise

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @property
    def tileable_op_key(self):
        return self._tileable_op_key

    @classmethod
    @redirect_custom_log
    @enter_current_session
    def execute(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        if isinstance(input_data, pd.DataFrame):
            result = input_data.apply(op.func, axis=op.axis, raw=op.raw, result_type=op.result_type,
                                      args=op.args, **op.kwds)
        else:
            result = input_data.apply(op.func, convert_dtype=op.convert_dtype, args=op.args,
                                      **op.kwds)
        ctx[op.outputs[0].key] = result

    @classmethod
    def _tile_df(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        axis = op.axis
        elementwise = op.elementwise

        if not elementwise and in_df.chunk_shape[axis] > 1:
            chunk_size = (
                in_df.shape[axis],
                max(1, options.chunk_store_limit // in_df.shape[axis]),
            )
            if axis == 1:
                chunk_size = chunk_size[::-1]
            in_df = in_df.rechunk(chunk_size)._inplace_tile()

        chunks = []
        if out_df.ndim == 2:
            for c in in_df.chunks:
                if elementwise:
                    new_shape = c.shape
                    new_index_value, new_columns_value = c.index_value, c.columns_value
                else:
                    new_shape = [np.nan, np.nan]
                    new_shape[1 - axis] = c.shape[1 - axis]
                    if axis == 0:
                        new_index_value = out_df.index_value
                        new_columns_value = c.columns_value
                    else:
                        new_index_value = c.index_value
                        new_columns_value = out_df.columns_value

                if op.axis == 0:
                    new_dtypes = out_df.dtypes[c.dtypes.keys()]
                else:
                    new_dtypes = out_df.dtypes

                new_op = op.copy().reset_key()
                new_op._tileable_op_key = op.key
                chunks.append(new_op.new_chunk([c], shape=tuple(new_shape), index=c.index, dtypes=new_dtypes,
                                               index_value=new_index_value, columns_value=new_columns_value))

            new_nsplits = list(in_df.nsplits)
            if not elementwise:
                new_nsplits[axis] = (np.nan,) * len(new_nsplits[axis])
        else:
            for c in in_df.chunks:
                shape_len = c.shape[1 - axis]
                new_index_value = c.index_value if axis == 1 else c.columns_value
                new_index = (c.index[1 - axis],)
                new_op = op.copy().reset_key()
                new_op._tileable_op_key = op.key
                chunks.append(new_op.new_chunk([c], shape=(shape_len,), index=new_index, dtype=out_df.dtype,
                                               index_value=new_index_value))
            new_nsplits = (in_df.nsplits[1 - axis],)

        new_op = op.copy()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=tuple(new_nsplits)))
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def _tile_series(cls, op):
        in_series = op.inputs[0]
        out_series = op.outputs[0]

        chunks = []
        for c in in_series.chunks:
            new_op = op.copy().reset_key()
            new_op._tileable_op_key = op.key
            kw = c.params.copy()
            if out_series.ndim == 1:
                kw['dtype'] = out_series.dtype
            else:
                kw['index'] = (c.index[0], 0)
                kw['shape'] = (c.shape[0], out_series.shape[1])
                kw['dtypes'] = out_series.dtypes
                kw['columns_value'] = out_series.columns_value
            chunks.append(new_op.new_chunk([c], **kw))

        new_op = op.copy()
        kw = out_series.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_series.nsplits))
        if out_series.ndim == 2:
            kw['nsplits'] = (in_series.nsplits[0], (out_series.shape[1],))
            kw['columns_value'] = out_series.columns_value
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op):
        if op.inputs[0].ndim == 2:
            return cls._tile_df(op)
        else:
            return cls._tile_series(op)

    def _infer_df_func_returns(self, df, dtypes, index):
        if isinstance(self._func, np.ufunc):
            output_type, new_dtypes, index_value, new_elementwise = \
                OutputType.dataframe, None, 'inherit', True
        else:
            output_type, new_dtypes, index_value, new_elementwise = None, None, None, False

        try:
            empty_df = build_df(df, size=2)
            with np.errstate(all='ignore'):
                infer_df = empty_df.apply(self._func, axis=self._axis, raw=self._raw,
                                          result_type=self._result_type, args=self.args, **self.kwds)
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = 'inherit'
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            if isinstance(infer_df, pd.DataFrame):
                output_type = output_type or OutputType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                output_type = output_type or OutputType.series
                new_dtypes = new_dtypes or infer_df.dtype
            new_elementwise = False if new_elementwise is None else new_elementwise
        except:  # noqa: E722  # nosec
            pass

        self.output_types = [output_type] if not self.output_types else self.output_types
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        self._elementwise = new_elementwise if self._elementwise is None else self._elementwise
        return dtypes, index_value

    def _call_dataframe(self, df, dtypes=None, index=None):
        dtypes, index_value = self._infer_df_func_returns(df, dtypes, index)
        if index_value is None:
            index_value = parse_index(None, (df.key, df.index_value.key))
        for arg, desc in zip((self.output_types, dtypes), ('output_types', 'dtypes')):
            if arg is None:
                raise TypeError(f'Cannot determine {desc} by calculating with enumerate data, '
                                'please specify it as arguments')

        if index_value == 'inherit':
            index_value = df.index_value

        if self._elementwise:
            shape = df.shape
        elif self.output_types[0] == OutputType.dataframe:
            shape = [np.nan, np.nan]
            shape[1 - self.axis] = df.shape[1 - self.axis]
            shape = tuple(shape)
        else:
            shape = (df.shape[1 - self.axis],)

        if self.output_types[0] == OutputType.dataframe:
            if self.axis == 0:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=index_value,
                                          columns_value=parse_index(dtypes.index))
            else:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=df.index_value,
                                          columns_value=parse_index(dtypes.index))
        else:
            return self.new_series([df], shape=shape, dtype=dtypes, index_value=index_value)

    def _call_series(self, series, dtype=None, index=None):
        if self._convert_dtype:
            try:
                test_series = build_series(series, size=2, name=series.name)
                with np.errstate(all='ignore'):
                    infer_series = test_series.apply(self._func, args=self.args, **self.kwds)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                infer_series = None

            output_type = self._output_types[0]

            if index is not None:
                index_value = parse_index(index)
            elif infer_series is not None:
                index_value = parse_index(infer_series.index)
            else:
                index_value = parse_index(None, series)

            if output_type == OutputType.dataframe:
                if dtype is not None:
                    dtypes = dtype
                elif infer_series is not None and infer_series.ndim == 2:
                    dtypes = infer_series.dtypes
                else:
                    raise TypeError('Cannot determine dtypes, '
                                    'please specify `dtypes` as argument')

                columns_value = parse_index(dtypes.index, store_data=True)

                return self.new_dataframe([series], shape=(series.shape[0], len(dtypes)),
                                          index_value=index_value, columns_value=columns_value,
                                          dtypes=dtypes)
            else:
                if dtype is None and infer_series is not None and infer_series.ndim == 1:
                    dtype = infer_series.dtype
                else:
                    dtype = np.dtype(object)
                if infer_series is not None and infer_series.ndim == 1:
                    name = infer_series.name
                else:
                    name = None
                return self.new_series([series], dtype=dtype, shape=series.shape,
                                       index_value=index_value, name=name)
        else:
            dtype, name = np.dtype('object'), None
            return self.new_series([series], dtype=dtype, shape=series.shape,
                                   index_value=series.index_value, name=name)

    def __call__(self, df_or_series, dtypes=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        self._axis = validate_axis(axis, df_or_series)

        if df_or_series.op.output_types[0] == OutputType.dataframe:
            return self._call_dataframe(df_or_series, dtypes=dtypes, index=index)
        else:
            return self._call_series(df_or_series, dtype=dtypes, index=index)


def df_apply(df, func, axis=0, raw=False, result_type=None, args=(), dtypes=None,
             output_type=None, index=None, elementwise=None, **kwds):
    if isinstance(func, (list, dict)):
        return df.aggregate(func)

    output_types = kwds.pop('output_types', None)
    object_type = kwds.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else None

    # calling member function
    if isinstance(func, str):
        func = getattr(df, func)
        sig = inspect.getfullargspec(func)
        if "axis" in sig.args:
            kwds["axis"] = axis
        return func(*args, **kwds)

    op = ApplyOperand(func=func, axis=axis, raw=raw, result_type=result_type, args=args, kwds=kwds,
                      output_type=output_type, elementwise=elementwise)
    return op(df, dtypes=dtypes, index=index)


def series_apply(series, func, convert_dtype=True, output_type=None,
                 args=(), index=None, **kwds):
    if isinstance(func, (list, dict)):
        return series.aggregate(func)

    # calling member function
    if isinstance(func, str):
        func_body = getattr(series, func, None)
        if func_body is not None:
            return func_body(*args, **kwds)
        func_str = func
        func = getattr(np, func_str, None)
        if func is None:
            raise AttributeError(f"'{func_str!r}' is not a valid function "
                                 f"for '{type(series).__name__}' object")

    output_types = kwds.pop('output_types', None)
    object_type = kwds.pop('object_type', None)
    output_types = validate_output_types(
        output_type=output_type, output_types=output_types, object_type=object_type)
    output_type = output_types[0] if output_types else OutputType.series
    dtypes = kwds.pop('dtypes', kwds.pop('dtype', None))

    op = ApplyOperand(func=func, convert_dtype=convert_dtype, args=args, kwds=kwds,
                      output_type=output_type)
    return op(series, dtypes=dtypes, index=index)
