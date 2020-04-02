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
from ...serialize import TupleField, DictField, FunctionField
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..utils import build_empty_df, build_empty_series, parse_index


class GroupByApply(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.APPLY
    _op_module_ = 'dataframe.groupby'

    _func = FunctionField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, args=None, kwds=None, object_type=None, **kw):
        super().__init__(_func=func, _args=args, _kwds=kwds, _object_type=object_type, **kw)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return getattr(self, '_args', None) or ()

    @property
    def kwds(self):
        return getattr(self, '_kwds', None) or dict()

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        if not in_data:
            if op.object_type == ObjectType.dataframe:
                ctx[op.outputs[0].key] = build_empty_df(op.outputs[0].dtypes)
            else:
                ctx[op.outputs[0].key] = build_empty_series(op.outputs[0].dtype)
            return

        applied = in_data.apply(op.func, *op.args, **op.kwds)

        # when there is only one group, pandas tend to return a DataFrame, while
        # we need to convert it into a compatible series
        if op.object_type == ObjectType.series and isinstance(applied, pd.DataFrame):
            assert len(applied.index) == 1
            applied_idx = pd.MultiIndex.from_arrays(
                [[applied.index[0]] * len(applied.columns), applied.columns.tolist()])
            applied_idx.names = [applied.index.name, None]
            applied = pd.Series(np.array(applied.iloc[0]), applied_idx, name=applied.columns.name)
        ctx[op.outputs[0].key] = applied

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
            kw['nsplits'] = ((np.nan,) * len(chunks), (out_df.shape[1],))
        else:
            kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    def _infer_df_func_returns(self, in_groupby, in_df, dtypes, index):
        index_value, object_type, new_dtypes = None, None, None

        try:
            if in_df.op.object_type == ObjectType.dataframe:
                empty_df = build_empty_df(in_df.dtypes, index=pd.RangeIndex(2))
            else:
                empty_df = build_empty_series(in_df.dtype, index=pd.RangeIndex(2), name=in_df.name)

            selection = getattr(in_groupby.op, 'selection', None)
            if selection:
                empty_df = empty_df[selection]

            with np.errstate(all='ignore'):
                infer_df = self.func(empty_df, *self.args, **self.kwds)

            # todo return proper index when sort=True is implemented
            index_value = parse_index(None, in_df.key, self.func)

            if isinstance(infer_df, pd.DataFrame):
                object_type = object_type or ObjectType.dataframe
                new_dtypes = new_dtypes or infer_df.dtypes
            elif isinstance(infer_df, pd.Series):
                object_type = object_type or ObjectType.series
                new_dtypes = new_dtypes or (infer_df.name, infer_df.dtype)
            else:
                object_type = ObjectType.series
                new_dtypes = (None, pd.Series(infer_df).dtype)
        except:  # noqa: E722  # nosec
            pass

        self._object_type = object_type if self._object_type is None else self._object_type
        dtypes = new_dtypes if dtypes is None else dtypes
        index_value = index_value if index is None else parse_index(index)
        return dtypes, index_value

    def __call__(self, groupby, dtypes=None, index=None):
        in_df = groupby
        while in_df.op.object_type not in (ObjectType.dataframe, ObjectType.series):
            in_df = in_df.inputs[0]

        dtypes, index_value = self._infer_df_func_returns(groupby, in_df, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if self.object_type == ObjectType.dataframe:
            new_shape = (np.nan, len(dtypes))
            return self.new_dataframe([groupby], shape=new_shape, dtypes=dtypes, index_value=index_value,
                                      columns_value=parse_index(dtypes.index, store_data=True))
        else:
            name, dtype = dtypes
            new_shape = (np.nan,)
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)


def groupby_apply(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.groupby_params.get('as_index', True):
        raise NotImplementedError('apply when set_index == False is not supported')
    op = GroupByApply(func=func, args=args, kwds=kwargs, object_type=object_type)
    return op(groupby, dtypes=dtypes, index=index)
