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
from ...serialize import BoolField, TupleField, DictField, AnyField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index


class GroupByTransform(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.TRANSFORM
    _op_module_ = 'dataframe.groupby'

    _func = AnyField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    _call_agg = BoolField('call_agg')

    def __init__(self, func=None, args=None, kwds=None, call_agg=None, object_type=None, **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _call_agg=call_agg,
                         _object_type=object_type, **kw)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @property
    def call_agg(self):
        return self._call_agg

    def _infer_df_func_returns(self, in_groupby, dtypes, index):
        index_value, object_type, new_dtypes = None, None, None

        object_type = ObjectType.dataframe \
            if in_groupby.op.object_type == ObjectType.dataframe_groupby else ObjectType.series

        try:
            empty_groupby = in_groupby.op.build_mock_groupby()
            with np.errstate(all='ignore'):
                if self.call_agg:
                    infer_df = empty_groupby.agg(self.func, *self.args, **self.kwds)
                else:
                    infer_df = empty_groupby.transform(self.func, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(None, in_groupby.key, self.func)

            if isinstance(infer_df, pd.DataFrame):
                object_type = ObjectType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            else:
                object_type = ObjectType.series
                new_dtypes = new_dtypes or (infer_df.name, infer_df.dtype)
        except:  # noqa: E722  # nosec
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, index=None):
        in_df = groupby.inputs[0]

        dtypes, index_value = self._infer_df_func_returns(groupby, dtypes, index)
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
            new_shape = (np.nan,) if self.call_agg else groupby.shape
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
                new_index = c.index if c.ndim == 2 else c.index + (0,)
                chunks.append(new_op.new_chunk(
                    inp_chunks, index=new_index, shape=(np.nan, len(out_df.dtypes)), dtypes=out_df.dtypes,
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

        if op.call_agg:
            result = in_data.agg(op.func, *op.args, **op.kwds)
        else:
            result = in_data.transform(op.func, *op.args, **op.kwds)

        if result.ndim == 2:
            result = result.astype(op.outputs[0].dtypes, copy=False)
        else:
            result = result.astype(op.outputs[0].dtype, copy=False)
        ctx[op.outputs[0].key] = result


def groupby_transform(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.groupby_params.get('as_index', True):
        raise NotImplementedError('transform when set_index == False is not supported')

    call_agg = kwargs.pop('_call_agg', False)
    if not call_agg and isinstance(func, (dict, list)):
        raise TypeError('Does not support transform with %r' % type(func))

    op = GroupByTransform(func=func, args=args, kwds=kwargs, object_type=object_type,
                          call_agg=call_agg)
    return op(groupby, dtypes=dtypes, index=index)
