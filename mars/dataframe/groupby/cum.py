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
from ...utils import lazy_import
from ...serialize import BoolField, AnyField
from ..utils import parse_index, build_empty_df, build_empty_series, validate_axis
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType

cudf = lazy_import('cudf', globals=globals())


class GroupByCumReductionOperand(DataFrameOperandMixin, DataFrameOperand):
    _op_module_ = 'dataframe.groupby'

    _axis = AnyField('axis')
    _ascending = BoolField('ascending')

    _by = AnyField('by')
    _as_index = BoolField('as_index')

    def __init__(self, axis=None, ascending=None, by=None, as_index=None, gpu=None, sparse=None,
                 object_type=None, stage=None, **kw):
        super().__init__(_axis=axis, _ascending=ascending, _by=by, _as_index=as_index,
                         _gpu=gpu, _sparse=sparse, _object_type=object_type, _stage=stage, **kw)

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def ascending(self) -> bool:
        return self._ascending

    @property
    def by(self):
        return self._by

    @property
    def as_index(self) -> bool:
        return self._as_index

    def _calc_out_dtypes(self, in_object_type, in_dtypes):
        if in_object_type == ObjectType.dataframe:
            empty_df = build_empty_df(in_dtypes, pd.RangeIndex(0, 5))
        else:
            empty_df = build_empty_series(in_dtypes, pd.RangeIndex(0, 5))

        func_name = getattr(self, '_func_name')

        if func_name == 'cumcount':
            result_df = empty_df.groupby(self.by, as_index=self.as_index) \
                .cumcount(ascending=self.ascending)
        else:
            grouped = empty_df.groupby(self.by, as_index=self.as_index)
            result_df = getattr(grouped, func_name)(axis=self.axis)

        return result_df.dtypes

    def __call__(self, groupby):
        in_df = groupby.op.inputs[0]

        if in_df.op.object_type == ObjectType.dataframe:
            out_dtypes = self._calc_out_dtypes(in_df.op.object_type, in_df.dtypes)
        else:
            out_dtypes = self._calc_out_dtypes(in_df.op.object_type, in_df.dtype)

        kw = in_df.params.copy()
        kw['index_value'] = parse_index(pd.RangeIndex(-1), groupby.key)
        if self.object_type == ObjectType.dataframe:
            kw.update(dict(columns_value=parse_index(out_dtypes.index, store_data=True),
                           dtypes=out_dtypes, shape=(in_df.shape[0], len(out_dtypes))))
        else:
            kw.update(dtype=out_dtypes, shape=(in_df.shape[0],))
        return self.new_tileable([groupby], **kw)

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            new_op = op.copy().reset_key()

            new_index = parse_index(pd.RangeIndex(-1), c.key)
            if op.object_type == ObjectType.dataframe:
                chunks.append(new_op.new_chunk(
                    [c], index=c.index, shape=(np.nan, len(out_df.dtypes)), dtypes=out_df.dtypes,
                    columns_value=out_df.columns_value, index_value=new_index))
            else:
                chunks.append(new_op.new_chunk(
                    [c], index=c.index, shape=(np.nan,), dtype=out_df.dtype,
                    index_value=new_index))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        return new_op.new_tileables([in_groupby], **kw)

    @classmethod
    def execute(cls, ctx, op: "GroupByCumReductionOperand"):
        in_data = ctx[op.inputs[0].key]
        out_df = op.outputs[0]

        if not in_data:
            ctx[out_df.key] = build_empty_df(out_df.dtypes) \
                if op.object_type == ObjectType.dataframe else build_empty_series(out_df.dtype)
            return

        concatenated = pd.concat([df for _, df in in_data])
        grouped = concatenated.groupby(op.by, as_index=op.as_index)

        func_name = getattr(op, '_func_name')
        if func_name == 'cumcount':
            ctx[out_df.key] = grouped.cumcount(ascending=op.ascending)
        else:
            ctx[out_df.key] = getattr(grouped, func_name)(axis=op.axis)


class GroupByCummin(GroupByCumReductionOperand):
    _op_type_ = opcodes.CUMMIN
    _func_name = 'cummin'


class GroupByCummax(GroupByCumReductionOperand):
    _op_type_ = opcodes.CUMMAX
    _func_name = 'cummax'


class GroupByCumsum(GroupByCumReductionOperand):
    _op_type_ = opcodes.CUMSUM
    _func_name = 'cumsum'


class GroupByCumprod(GroupByCumReductionOperand):
    _op_type_ = opcodes.CUMPROD
    _func_name = 'cumprod'


class GroupByCumcount(GroupByCumReductionOperand):
    _op_type_ = opcodes.CUMCOUNT
    _func_name = 'cumcount'


def cumcount(groupby, ascending: bool = True):
    op = GroupByCumcount(ascending=ascending, by=groupby.op.by, as_index=groupby.op.as_index,
                         object_type=ObjectType.series)
    return op(groupby)


def cummin(groupby, axis=0):
    in_df = groupby.op.inputs[0]
    op = GroupByCummin(axis=axis, by=groupby.op.by, as_index=groupby.op.as_index,
                       object_type=in_df.op.object_type)
    return op(groupby)


def cummax(groupby, axis=0):
    in_df = groupby.op.inputs[0]
    op = GroupByCummax(axis=axis, by=groupby.op.by, as_index=groupby.op.as_index,
                       object_type=in_df.op.object_type)
    return op(groupby)


def cumprod(groupby, axis=0):
    in_df = groupby.op.inputs[0]
    op = GroupByCumprod(axis=axis, by=groupby.op.by, as_index=groupby.op.as_index,
                        object_type=in_df.op.object_type)
    return op(groupby)


def cumsum(groupby, axis=0):
    in_df = groupby.op.inputs[0]
    axis = validate_axis(axis, in_df)
    op = GroupByCumsum(axis=axis, by=groupby.op.by, as_index=groupby.op.as_index,
                       object_type=in_df.op.object_type)
    return op(groupby)
