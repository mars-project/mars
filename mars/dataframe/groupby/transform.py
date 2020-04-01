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
from ...serialize import AnyField, BoolField, TupleField, DictField, FunctionField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index


class GroupByTransform(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM
    _op_module_ = 'dataframe.groupby'

    # todo these three args below shall be redesigned when we extend
    #  the functionality of groupby func
    _by = AnyField('by')
    _as_index = BoolField('as_index')

    _func = FunctionField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    _call_agg = BoolField('call_agg')

    def __init__(self, func=None, by=None, as_index=None, args=None, kwds=None,
                 call_agg=None, object_type=None, **kw):
        super().__init__(_func=func, _by=by, _as_index=as_index, _args=args, _kwds=kwds,
                         _call_agg=call_agg, _object_type=object_type, **kw)

    @property
    def func(self):
        return self._func

    @property
    def by(self):
        return self._by

    @property
    def as_index(self):
        return self._as_index

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @property
    def call_agg(self):
        return self._call_agg

    def _infer_df_func_returns(self, in_df, dtypes, index):
        index_value, object_type, new_dtypes = None, None, None
        object_type = in_df.op.object_type if not self.call_agg else None

        try:
            if in_df.op.object_type == ObjectType.dataframe:
                empty_df = build_empty_df(in_df.dtypes, index=pd.RangeIndex(2))
            else:
                empty_df = build_empty_series(in_df.dtype, index=pd.RangeIndex(2), name=in_df.name)

            with np.errstate(all='ignore'):
                if self.call_agg:
                    infer_df = empty_df.groupby(by=self.by, as_index=self.as_index) \
                        .agg(self.func, *self.args, **self.kwds)
                else:
                    infer_df = empty_df.groupby(by=self.by, as_index=self.as_index) \
                            .transform(self.func, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(None, in_df.key, self.func)

            if isinstance(infer_df, pd.DataFrame):
                object_type = object_type or ObjectType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                object_type = object_type or ObjectType.series
                new_dtypes = new_dtypes or (infer_df.name, infer_df.dtype)
        except:  # noqa: E722  # nosec
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, index=None):
        in_df = groupby.inputs[0]

        dtypes, index_value = self._infer_df_func_returns(in_df, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if self.object_type == ObjectType.dataframe:
            new_shape = (np.nan if self.call_agg else in_df.shape[0], len(dtypes))
            return self.new_dataframe([groupby], shape=new_shape, dtypes=dtypes, index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes
            new_shape = (np.nan,) if self.call_agg else in_df.shape
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            inp_chunks = [c]

            new_op = op.copy().reset_key()
            if op.object_type == ObjectType.dataframe:
                chunks.append(new_op.new_chunk(
                    inp_chunks, index=c.index, shape=(np.nan, len(out_df.dtypes)), dtypes=out_df.dtypes,
                    columns_value=out_df.columns_value, index_value=out_df.index_value))
            else:
                chunks.append(new_op.new_chunk(
                    inp_chunks, name=out_df.name, index=(c.index[0],), shape=(np.nan,), dtype=out_df.dtype,
                    index_value=out_df.index_value))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        if op.object_type == ObjectType.dataframe:
            kw['nsplits'] = ((np.nan,) * len(chunks), (len(out_df.dtypes),))
        else:
            kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        out_chunk = op.outputs[0]

        if not in_data:
            if op.object_type == ObjectType.dataframe:
                ctx[op.outputs[0].key] = build_empty_df(out_chunk.dtypes)
            else:
                ctx[op.outputs[0].key] = build_empty_series(out_chunk.dtype)
            return

        concatenated = pd.concat([df for _, df in in_data])
        grouped = concatenated.groupby(op.by, as_index=op.as_index)

        if op.call_agg:
            result = grouped.agg(op.func, *op.args, **op.kwds)
        else:
            result = grouped.transform(op.func, *op.args, **op.kwds)
        ctx[op.outputs[0].key] = result


def groupby_transform(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.as_index:
        raise NotImplementedError('transform when set_index == False is not supported')

    call_agg = kwargs.pop('_call_agg', False)
    if not call_agg and isinstance(func, (dict, list)):
        raise TypeError('Does not support transform with %r' % type(func))

    op = GroupByTransform(func=func, by=groupby.op.by, as_index=groupby.op.as_index,
                          args=args, kwds=kwargs, object_type=object_type, call_agg=call_agg)
    return op(groupby, dtypes=dtypes, index=index)
