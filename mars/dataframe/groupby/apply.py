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


class GroupByApplyTransform(DataFrameOperand, DataFrameOperandMixin):
    # todo these three args below shall be redesigned when we extend
    #  the functionality of groupby func
    _by = AnyField('by')
    _as_index = BoolField('as_index')

    _is_transform = BoolField('is_transform')

    _func = FunctionField('func')
    _args = TupleField('args')
    _kwds = DictField('kwds')

    def __init__(self, func=None, by=None, as_index=None, is_transform=None,
                 args=None, kwds=None, object_type=None, **kw):
        super().__init__(_func=func, _by=by, _as_index=as_index,
                         _is_transform=is_transform, _args=args, _kwds=kwds,
                         _object_type=object_type, **kw)

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
    def is_transform(self):
        return self._is_transform

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

        concatenated = pd.concat([df for _, df in in_data])
        grouped = concatenated.groupby(op.by, as_index=op.as_index)

        if not op.is_transform:
            applied = grouped.apply(op.func, *op.args, **op.kwds)

            # when there is only one group, pandas tend to return a DataFrame, while
            # we need to convert it into a compatible series
            if op.object_type == ObjectType.series and isinstance(applied, pd.DataFrame):
                assert len(applied.index) == 1
                applied_idx = pd.MultiIndex.from_arrays(
                    [[applied.index[0]] * len(applied.columns), applied.columns.to_list()])
                applied_idx.names = [applied.index.name, None]
                applied = pd.Series(applied.iloc[0].to_numpy(), applied_idx, name=applied.columns.name)
        else:
            applied = grouped.transform(op.func, *op.args, **op.kwds)
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
                    inp_chunks, name=out_df.name, index=c.index, shape=(np.nan,), dtype=out_df.dtype,
                    index_value=out_df.index_value))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        if not op.is_transform:
            if op.object_type == ObjectType.dataframe:
                kw['nsplits'] = ((np.nan,) * len(chunks), (out_df.shape[1],))
            else:
                kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    def _infer_df_func_returns(self, in_object_type, in_dtypes, dtypes, index):
        index_value, object_type, new_dtypes = None, None, None
        if self.is_transform:
            object_type = in_object_type

        try:
            if in_object_type == ObjectType.dataframe:
                empty_df = build_empty_df(in_dtypes, index=pd.RangeIndex(2))
            else:
                empty_df = build_empty_series(in_dtypes[1], index=pd.RangeIndex(2), name=in_dtypes[0])

            with np.errstate(all='ignore'):
                if self.is_transform:
                    infer_df = empty_df.apply(self.func, *self.args, **self.kwds)
                else:
                    infer_df = self.func(empty_df, *self.args, **self.kwds)

            if isinstance(infer_df, np.number):
                index_value = parse_index(pd.RangeIndex(0, 1))
            else:
                # todo return proper index when sort=True is implemented
                index_value = parse_index(pd.RangeIndex(-1))

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
        in_df = groupby.inputs[0]
        in_dtypes = getattr(in_df, 'dtypes', None)
        if in_dtypes is None:
            in_dtypes = (in_df.name, in_df.dtype)

        dtypes, index_value = self._infer_df_func_returns(
            in_df.op.object_type, in_dtypes, dtypes, index)
        for arg, desc in zip((self._object_type, dtypes, index_value),
                             ('object_type', 'dtypes', 'index')):
            if arg is None:
                raise TypeError('Cannot determine %s by calculating with enumerate data, '
                                'please specify it as arguments' % desc)

        if self.object_type == ObjectType.dataframe:
            new_shape = in_df.shape if self.is_transform else (np.nan, len(dtypes))
            return self.new_dataframe([groupby], shape=new_shape, dtypes=dtypes,
                                      index_value=index_value, columns_value=in_df.columns_value)
        else:
            name, dtype = dtypes
            new_shape = in_df.shape if self.is_transform else (np.nan,)
            return self.new_series([groupby], name=name, shape=new_shape, dtype=dtype,
                                   index_value=index_value)


class GroupByApply(GroupByApplyTransform):
    _op_type_ = opcodes.GROUPBY_APPLY


class GroupByTransform(GroupByApplyTransform):
    _op_type_ = opcodes.GROUPBY_TRANSFORM


def groupby_apply(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.as_index:
        raise NotImplementedError('apply when set_index == False is not supported')
    op = GroupByApply(func=func, by=groupby.op.by,
                      as_index=groupby.op.as_index, is_transform=False,
                      args=args, kwds=kwargs, object_type=object_type)
    return op(groupby, dtypes=dtypes, index=index)


def groupby_transform(groupby, func, *args, dtypes=None, index=None, object_type=None, **kwargs):
    # todo this can be done with sort_index implemented
    if not groupby.op.as_index:
        raise NotImplementedError('transform when set_index == False is not supported')
    op = GroupByTransform(func=func, by=groupby.op.by,
                          as_index=groupby.op.as_index, is_transform=True, args=args,
                          kwds=kwargs, object_type=object_type)
    return op(groupby, dtypes=dtypes, index=index)
