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

    def __init__(self, axis=None, ascending=None, gpu=None, sparse=None, object_type=None,
                 stage=None, **kw):
        super().__init__(_axis=axis, _ascending=ascending, _gpu=gpu, _sparse=sparse,
                         _object_type=object_type, _stage=stage, **kw)

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def ascending(self) -> bool:
        return self._ascending

    def _calc_out_dtypes(self, in_groupby):
        empty_groupby = in_groupby.op.build_mock_groupby()
        func_name = getattr(self, '_func_name')

        if func_name == 'cumcount':
            result_df = empty_groupby.cumcount(ascending=self.ascending)
        else:
            result_df = getattr(empty_groupby, func_name)(axis=self.axis)

        if isinstance(result_df, pd.DataFrame):
            self._object_type = ObjectType.dataframe
            return result_df.dtypes
        else:
            self._object_type = ObjectType.series
            return result_df.name, result_df.dtype

    def __call__(self, groupby):
        in_df = groupby
        while in_df.op.object_type not in (ObjectType.dataframe, ObjectType.series):
            in_df = in_df.inputs[0]

        self._axis = validate_axis(self.axis or 0, in_df)

        out_dtypes = self._calc_out_dtypes(groupby)

        kw = in_df.params.copy()
        kw['index_value'] = parse_index(pd.RangeIndex(-1), groupby.key)
        if self.object_type == ObjectType.dataframe:
            kw.update(dict(columns_value=parse_index(out_dtypes.index, store_data=True),
                           dtypes=out_dtypes, shape=(groupby.shape[0], len(out_dtypes))))
        else:
            name, dtype = out_dtypes
            kw.update(dtype=dtype, name=name, shape=(groupby.shape[0],))
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
                    [c], index=(c.index[0],), shape=(np.nan,), dtype=out_df.dtype,
                    index_value=new_index))

        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw['chunks'] = chunks
        if op.object_type == ObjectType.dataframe:
            kw['nsplits'] = ((np.nan,) * len(chunks), (len(out_df.dtypes),))
        else:
            kw['nsplits'] = ((np.nan,) * len(chunks),)
        return new_op.new_tileables([in_groupby], **kw)

    @classmethod
    def execute(cls, ctx, op: "GroupByCumReductionOperand"):
        in_data = ctx[op.inputs[0].key]
        out_df = op.outputs[0]

        if not in_data or in_data.empty:
            ctx[out_df.key] = build_empty_df(out_df.dtypes) \
                if op.object_type == ObjectType.dataframe else build_empty_series(out_df.dtype)
            return

        func_name = getattr(op, '_func_name')
        if func_name == 'cumcount':
            ctx[out_df.key] = in_data.cumcount(ascending=op.ascending)
        else:
            result = getattr(in_data, func_name)(axis=op.axis)
            if result.ndim == 2:
                ctx[out_df.key] = result.astype(out_df.dtypes, copy=False)
            else:
                ctx[out_df.key] = result.astype(out_df.dtype, copy=False)


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
    op = GroupByCumcount(ascending=ascending)
    return op(groupby)


def cummin(groupby, axis=0):
    op = GroupByCummin(axis=axis)
    return op(groupby)


def cummax(groupby, axis=0):
    op = GroupByCummax(axis=axis)
    return op(groupby)


def cumprod(groupby, axis=0):
    op = GroupByCumprod(axis=axis)
    return op(groupby)


def cumsum(groupby, axis=0):
    op = GroupByCumsum(axis=axis)
    return op(groupby)
