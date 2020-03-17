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

import numpy as np
import pandas as pd

from ... import opcodes
from ...config import options
from ...serialize import AnyField, BoolField, TupleField, DictField, FunctionField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index, validate_axis


class DataFrameTransform(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.DATAFRAME_TRANSFORM

    _func = FunctionField('func')
    _axis = AnyField('axis')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, axis=None, args=None, kwds=None, object_type=None, **kw):
        super().__init__(_func=func, _axis=axis, _args=args, _kwds=kwds, _object_type=object_type, **kw)

    @property
    def func(self):
        return self._func

    @property
    def axis(self):
        return self._axis

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @classmethod
    def execute(cls, ctx, op):
        in_df = op.inputs[0]
        df = op.outputs[0]

        input_data = ctx[in_df.key]
        ctx[df.key] = input_data.transform(op.func, axis=op.axis, *op.args, **op.kwds)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        axis = op.axis

        chunk_size = (
            in_df.shape[axis],
            max(1, options.chunk_store_limit // in_df.shape[axis]),
        )
        if axis == 1:
            chunk_size = chunk_size[::-1]
        in_df = in_df.rechunk(chunk_size).tiles()

        chunks = []
        for c in in_df.chunks:
            new_shape = c.shape
            new_index_value, new_columns_value = c.index_value, c.columns_value

            new_dtypes = out_df.dtypes.loc[c.dtypes.keys()]

            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([c], shape=tuple(new_shape), index=c.index, dtypes=new_dtypes,
                                           index_value=new_index_value, columns_value=new_columns_value))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_df.nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    def _infer_df_func_returns(self, in_dtypes, dtypes, index):
        object_type, new_dtypes, index_value = None, None, None
        try:
            empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(2))
            with np.errstate(all='ignore'):
                infer_df = empty_df.transform(self._func, axis=self._axis, *self.args, **self.kwds)
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
        except:  # noqa: E722  # nosec
            import traceback
            traceback.print_exc()
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, df, dtypes=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        self._axis = axis = validate_axis(axis, df)

        dtypes, index_value = self._infer_df_func_returns(df.dtypes, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if index_value == 'inherit':
            index_value = df.index_value

        if axis == 0:
            return self.new_dataframe([df], shape=df.shape, dtypes=dtypes, index_value=index_value,
                                      columns_value=df.columns_value)
        else:
            return self.new_dataframe([df], shape=df.shape, dtypes=dtypes, index_value=df.index_value,
                                      columns_value=index_value)


class SeriesTransform(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.SERIES_TRANSFORM

    _func = FunctionField('func')
    _convert_dtype = BoolField('convert_dtype')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    @property
    def func(self):
        return self._func

    @property
    def convert_dtype(self):
        return self._convert_dtype

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    def __init__(self, func=None, convert_dtype=None, args=None, kwds=None, object_type=None, **kw):
        super().__init__(_func=func, _convert_dtype=convert_dtype, _args=args, _kwds=kwds,
                         _object_type=object_type, **kw)

    @classmethod
    def execute(cls, ctx, op):
        input_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = input_data.transform(op.func, *op.args, **op.kwds)

    @classmethod
    def tile(cls, op):
        in_series = op.inputs[0]
        out_series = op.outputs[0]

        chunks = []
        for c in in_series.chunks:
            new_op = op.copy().reset_key()
            kw = c.params.copy()
            kw['dtype'] = out_series.dtype
            chunks.append(new_op.new_chunk([c], **kw))

        new_op = op.copy().reset_key()
        kw = out_series.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_series.nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    def _infer_series_func_returns(self, in_dtype):
        try:
            empty_series = build_empty_series(in_dtype, index=pd.RangeIndex(2))
            with np.errstate(all='ignore'):
                infer_series = empty_series.apply(self._func, args=self.args, **self.kwds)
            new_dtype = infer_series.dtype
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            new_dtype = np.dtype('object')
        return new_dtype

    def __call__(self, series):
        if self._convert_dtype:
            dtype = self._infer_series_func_returns(series.dtype)
        else:
            dtype = np.dtype('object')
        return self.new_series([series], dtype=dtype, shape=series.shape,
                               index_value=series.index_value)


def df_transform(df, func, axis=0, *args, dtypes=None, index=None, **kwargs):
    # todo fulfill support on df.agg later
    op = DataFrameTransform(func=func, axis=axis, args=args, kwds=kwargs)
    return op(df, dtypes=dtypes, index=index)


def series_transform(series, func, axis=0, *args, **kwargs):
    # todo fulfill support on df.agg later
    op = SeriesTransform(func=func, convert_dtype=True, axis=axis, args=args, kwds=kwargs,
                         object_type=ObjectType.series)
    return op(series)
