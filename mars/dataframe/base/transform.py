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
from ..core import DATAFRAME_CHUNK_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, validate_axis, parse_index


class TransformOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM

    _func = FunctionField('func')
    _axis = AnyField('axis')
    _convert_dtype = BoolField('convert_dtype')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    _call_agg = BoolField('call_agg')

    def __init__(self, func=None, axis=None, convert_dtype=None, args=None, kwds=None,
                 call_agg=None, object_type=None, **kw):
        super().__init__(_func=func, _axis=axis, _convert_dtype=convert_dtype, _args=args,
                         _kwds=kwds, _call_agg=call_agg, _object_type=object_type, **kw)

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
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        axis = op.axis

        if in_df.op.object_type == ObjectType.dataframe:
            chunk_size = (in_df.shape[axis],
                          max(1, options.chunk_store_limit // in_df.shape[axis]))
            if axis == 1:
                chunk_size = chunk_size[::-1]
            in_df = in_df.rechunk(chunk_size).tiles()
        elif isinstance(op.func, str) or \
                 (isinstance(op.func, list) and any(isinstance(e, str) for e in op.func)):
            # builtin cols handles whole columns, thus merge is needed
            in_df = in_df.rechunk((in_df.shape[axis],)).tiles()

        chunks = []
        axis_index_map = dict()
        for c in in_df.chunks:
            new_op = op.copy().reset_key()
            params = c.params.copy()

            if op.object_type == ObjectType.dataframe:
                if isinstance(c, DATAFRAME_CHUNK_TYPE):
                    columns = c.columns_value.to_pandas()
                    try:
                        new_dtypes = out_df.dtypes.loc[columns]
                    except KeyError:
                        new_dtypes = out_df.dtypes.reindex(columns).dropna()

                    if len(new_dtypes) == 0:
                        continue

                    new_index = list(c.index)
                    try:
                        new_index[op.axis] = axis_index_map[c.index[op.axis]]
                    except KeyError:
                        new_index[op.axis] = axis_index_map[c.index[op.axis]] = len(axis_index_map)

                    if isinstance(op.func, dict):
                        new_op._func = dict((k, v) for k, v in op.func.items() if k in new_dtypes)
                else:
                    new_dtypes, new_index = out_df.dtypes, c.index
                params.update(dict(dtypes=new_dtypes, index=tuple(new_index)))
            else:
                params['dtype'] = out_df.dtype
                if in_df.op.object_type == ObjectType.dataframe:
                    params.pop('columns_value', None)
                    params['index_value'] = out_df.index_value
            chunks.append(new_op.new_chunk([c], **params))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=in_df.nsplits))
        return new_op.new_tileables(op.inputs, **kw)

    def _infer_df_func_returns(self, in_dtypes, dtypes):
        if self.object_type == ObjectType.dataframe:
            empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(2))
            with np.errstate(all='ignore'):
                if self.call_agg:
                    infer_df = empty_df.agg(self._func, axis=self._axis, *self.args, **self.kwds)
                else:
                    infer_df = empty_df.transform(self._func, axis=self._axis, *self.args, **self.kwds)
        else:
            empty_df = build_empty_series(in_dtypes[1], index=pd.RangeIndex(2), name=in_dtypes[0])
            with np.errstate(all='ignore'):
                if self.call_agg:
                    infer_df = empty_df.agg(self._func, args=self.args, **self.kwds)
                else:
                    infer_df = empty_df.transform(self._func, convert_dtype=self.convert_dtype,
                                                  args=self.args, **self.kwds)

        if isinstance(infer_df, pd.DataFrame):
            new_dtypes = dtypes or infer_df.dtypes
            self._object_type = ObjectType.dataframe
        else:
            new_dtypes = dtypes or (infer_df.name, infer_df.dtype)
            self._object_type = ObjectType.series

        return new_dtypes

    def __call__(self, df, dtypes=None, index=None):
        axis = getattr(self, 'axis', None) or 0
        self._axis = validate_axis(axis, df)

        if self.object_type == ObjectType.dataframe:
            dtypes = self._infer_df_func_returns(df.dtypes, dtypes)
        else:
            dtypes = self._infer_df_func_returns((df.name, df.dtype), dtypes)

        for arg, desc in zip((self._object_type, dtypes), ('object_type', 'dtypes')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if self.object_type == ObjectType.dataframe:
            return self.new_dataframe([df], shape=df.shape, dtypes=dtypes, index_value=df.index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes
            return self.new_series([df], shape=df.shape, name=name, dtype=dtype, index_value=df.index_value)


def df_transform(df, func, axis=0, *args, dtypes=None, **kwargs):
    op = TransformOperand(func=func, axis=axis, args=args, kwds=kwargs, object_type=df.op.object_type,
                          call_agg=kwargs.pop('_call_agg', False))
    return op(df, dtypes=dtypes)


def series_transform(series, func, convert_dtype=True, axis=0, *args, dtype=None, **kwargs):
    op = TransformOperand(func=func, axis=axis, convert_dtype=convert_dtype, args=args, kwds=kwargs,
                          object_type=series.op.object_type, call_agg=kwargs.pop('_call_agg', False))
    dtypes = (series.name, dtype) if dtype is not None else None
    return op(series, dtypes=dtypes)
