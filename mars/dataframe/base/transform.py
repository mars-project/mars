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
from ...serialize import AnyField, BoolField, TupleField, DictField
from ..core import DATAFRAME_CHUNK_TYPE, DATAFRAME_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, validate_axis, parse_index


class TransformOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM

    _func = AnyField('func')
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
                in_df = in_df.rechunk(chunk_size)._inplace_tile()
        elif isinstance(op.func, str) or \
                 (isinstance(op.func, list) and any(isinstance(e, str) for e in op.func)):
            # builtin cols handles whole columns, thus merge is needed
            if in_df.chunk_shape[0] > 1:
                in_df = in_df.rechunk((in_df.shape[axis],))._inplace_tile()

        chunks = []
        axis_index_map = dict()
        col_sizes = []
        for c in in_df.chunks:
            new_op = op.copy().reset_key()
            params = c.params.copy()

            if op.object_type == ObjectType.dataframe:
                if isinstance(c, DATAFRAME_CHUNK_TYPE):
                    columns = c.columns_value.to_pandas()
                    try:
                        new_dtypes = out_df.dtypes.loc[columns].dropna()
                    except KeyError:
                        new_dtypes = out_df.dtypes.reindex(columns).dropna()

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

        if op.object_type == ObjectType.dataframe:
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

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw.update(dict(chunks=chunks, nsplits=tuple(new_nsplits)))
        return new_op.new_tileables(op.inputs, **kw)

    def _infer_df_func_returns(self, in_dtypes, dtypes):
        if self.object_type == ObjectType.dataframe:
            empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(2))
            try:
                with np.errstate(all='ignore'):
                    if self.call_agg:
                        infer_df = empty_df.agg(self._func, axis=self._axis, *self.args, **self.kwds)
                    else:
                        infer_df = empty_df.transform(self._func, axis=self._axis, *self.args, **self.kwds)
            except:  # noqa: E722
                infer_df = None
        else:
            empty_df = build_empty_series(in_dtypes[1], index=pd.RangeIndex(2), name=in_dtypes[0])
            try:
                with np.errstate(all='ignore'):
                    if self.call_agg:
                        infer_df = empty_df.agg(self._func, args=self.args, **self.kwds)
                    else:
                        infer_df = empty_df.transform(self._func, convert_dtype=self.convert_dtype,
                                                      args=self.args, **self.kwds)
            except:  # noqa: E722
                infer_df = None

        if infer_df is None and dtypes is None:
            raise TypeError('Failed to infer dtype, please specify dtypes as arguments.')

        if infer_df is None:
            is_df = self.object_type == ObjectType.dataframe
        else:
            is_df = isinstance(infer_df, pd.DataFrame)

        if is_df:
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
    op = TransformOperand(func=func, axis=axis, args=args, kwds=kwargs, object_type=ObjectType.dataframe,
                          call_agg=kwargs.pop('_call_agg', False))
    return op(df, dtypes=dtypes)


def series_transform(series, func, convert_dtype=True, axis=0, *args, dtype=None, **kwargs):
    op = TransformOperand(func=func, axis=axis, convert_dtype=convert_dtype, args=args, kwds=kwargs,
                          object_type=ObjectType.series, call_agg=kwargs.pop('_call_agg', False))
    dtypes = (series.name, dtype) if dtype is not None else None
    return op(series, dtypes=dtypes)
