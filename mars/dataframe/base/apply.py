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
from ...serialize import StringField, AnyField, BoolField, \
    TupleField, DictField, FunctionField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index, validate_axis


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

    def __init__(self, func=None, axis=None, convert_dtype=None, raw=None, result_type=None,
                 args=None, kwds=None, object_type=None, elementwise=None, **kw):
        super().__init__(_func=func, _axis=axis, _convert_dtype=convert_dtype, _raw=raw,
                         _result_type=result_type, _args=args, _kwds=kwds, _object_type=object_type,
                         _elementwise=elementwise, **kw)

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

    @classmethod
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
        if op.object_type == ObjectType.dataframe:
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

                new_dtypes = out_df.dtypes[c.dtypes.keys()]

                new_op = op.copy().reset_key()
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
                chunks.append(new_op.new_chunk([c], shape=(shape_len,), index=new_index, dtype=out_df.dtype,
                                               index_value=new_index_value))
            new_nsplits = (in_df.nsplits[1 - axis],)

        new_op = op.copy().reset_key()
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
            kw = c.params.copy()
            kw['dtype'] = out_series.dtype
            if op.object_type == ObjectType.dataframe:
                kw['columns_value'] = out_series.columns_value
            chunks.append(new_op.new_chunk([c], **kw))

        new_op = op.copy().reset_key()
        kw = out_series.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_series.nsplits))
        if op.object_type == ObjectType.dataframe:
            kw['columns_value'] = out_series.columns_value
        return new_op.new_tileables(op.inputs, **kw)

    @classmethod
    def tile(cls, op):
        if op.inputs[0].op.object_type == ObjectType.dataframe:
            return cls._tile_df(op)
        else:
            return cls._tile_series(op)

    def _infer_df_func_returns(self, in_dtypes, dtypes, index):
        if isinstance(self._func, np.ufunc):
            object_type, new_dtypes, index_value, new_elementwise = \
                ObjectType.dataframe, None, 'inherit', True
        else:
            object_type, new_dtypes, index_value, new_elementwise = None, None, None, False

        try:
            empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(2))
            with np.errstate(all='ignore'):
                infer_df = empty_df.apply(self._func, axis=self._axis, raw=self._raw,
                                          result_type=self._result_type, args=self.args, **self.kwds)
            if index_value is None:
                if infer_df.index is empty_df.index:
                    index_value = 'inherit'
                else:
                    index_value = parse_index(pd.RangeIndex(-1))

            if isinstance(infer_df, pd.DataFrame):
                object_type = object_type or ObjectType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                object_type = object_type or ObjectType.series
                new_dtypes = new_dtypes or infer_df.dtype
            new_elementwise = False if new_elementwise is None else new_elementwise
        except:  # noqa: E722  # nosec
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        self._elementwise = new_elementwise if self._elementwise is None else self._elementwise
        return dtypes, index_value

    def _infer_series_func_returns(self, in_dtype):
        try:
            empty_series = build_empty_series(in_dtype, index=pd.RangeIndex(2))
            with np.errstate(all='ignore'):
                infer_series = empty_series.apply(self._func, args=self.args, **self.kwds)
            new_dtype = infer_series.dtype
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            new_dtype = np.dtype('object')
        return new_dtype

    def _call_dataframe(self, df, dtypes=None, index=None):
        dtypes, index_value = self._infer_df_func_returns(df.dtypes, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if index_value == 'inherit':
            index_value = df.index_value

        if self._elementwise:
            shape = df.shape
        elif self._object_type == ObjectType.dataframe:
            shape = [np.nan, np.nan]
            shape[1 - self.axis] = df.shape[1 - self.axis]
            shape = tuple(shape)
        else:
            shape = (df.shape[1 - self.axis],)

        if self._object_type == ObjectType.dataframe:
            if self.axis == 0:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=index_value,
                                          columns_value=parse_index(dtypes.index))
            else:
                return self.new_dataframe([df], shape=shape, dtypes=dtypes, index_value=df.index_value,
                                          columns_value=parse_index(dtypes.index))
        else:
            return self.new_series([df], shape=shape, dtype=dtypes, index_value=index_value)

    def _call_series(self, series):
        if self._convert_dtype:
            dtype = self._infer_series_func_returns(series.dtype)
        else:
            dtype = np.dtype('object')
        return self.new_series([series], dtype=dtype, shape=series.shape,
                               index_value=series.index_value)

    def __call__(self, df, dtypes=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        self._axis = validate_axis(axis, df)

        if df.op.object_type == ObjectType.dataframe:
            return self._call_dataframe(df, dtypes=dtypes, index=index)
        else:
            return self._call_series(df)


def df_apply(df, func, axis=0, raw=False, result_type=None, args=(), dtypes=None,
             object_type=None, index=None, elementwise=None, **kwds):
    if isinstance(func, (list, dict)):
        return df.aggregate(func)

    if isinstance(object_type, str):
        object_type = getattr(ObjectType, object_type.lower())

    # calling member function
    if isinstance(func, str):
        func = getattr(df, func)
        sig = inspect.getfullargspec(func)
        if "axis" in sig.args:
            kwds["axis"] = axis
        return func(*args, **kwds)

    op = ApplyOperand(func=func, axis=axis, raw=raw, result_type=result_type, args=args, kwds=kwds,
                      object_type=object_type, elementwise=elementwise)
    return op(df, dtypes=dtypes, index=index)


def series_apply(series, func, convert_dtype=True, args=(), **kwds):
    if isinstance(func, (list, dict)):
        return series.aggregate(func)

    # calling member function
    if isinstance(func, str):
        func_body = getattr(series, func, None)
        if func_body is not None:
            return func_body(*args, **kwds)
        func = getattr(np, func, None)
        if func is None:
            raise AttributeError("'%r' is not a valid function for '%s' object" %
                                 (func, type(series).__name__))

    op = ApplyOperand(func=func, convert_dtype=convert_dtype, args=args, kwds=kwds,
                      object_type=ObjectType.series)
    return op(series)
