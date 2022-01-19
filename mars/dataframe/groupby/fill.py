# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
from ...core import OutputType
from ...serialization.serializables import AnyField, DictField
from ..utils import parse_index, build_empty_df, build_empty_series
from ..operands import DataFrameOperandMixin, DataFrameOperand


class GroupByFillOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_module_ = "dataframe.groupby"

    value = AnyField("value", default=None)
    method = AnyField("method", default=None)
    axis = AnyField("axis", default=0)
    limit = AnyField("limit", default=None)
    downcast = DictField("downcast", default=None)

    def __init__(
        self,
        value=None,
        method=None,
        axis=0,
        limit=None,
        downcast=None,
        output_types=None,
        **kw
    ):
        super().__init__(
            value=value,
            method=method,
            axis=axis,
            limit=limit,
            downcast=downcast,
            output_types=output_types,
            **kw
        )

    def _calc_out_dtypes(self, in_groupby):
        mock_groupby = in_groupby.op.build_mock_groupby()
        func_name = getattr(self, "_func_name")

        if func_name == "fillna":
            result_df = mock_groupby.fillna(
                value=self.value,
                method=self.method,
                axis=self.axis,
                limit=self.limit,
                downcast=self.downcast,
            )
        else:
            result_df = getattr(mock_groupby, func_name)(limit=self.limit)

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return result_df.dtypes

    def __call__(self, groupby):
        in_df = groupby
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]
        out_dtypes = self._calc_out_dtypes(groupby)

        kw = in_df.params.copy()
        kw["index_value"] = parse_index(pd.RangeIndex(-1), groupby.key)
        if self.output_types[0] == OutputType.dataframe:
            kw.update(
                dict(
                    columns_value=parse_index(out_dtypes.index, store_data=True),
                    dtypes=out_dtypes,
                    shape=(groupby.shape[0], len(out_dtypes)),
                )
            )
        return self.new_tileable([groupby], **kw)

    @classmethod
    def tile(cls, op):
        in_groupby = op.inputs[0]
        out_df = op.outputs[0]

        chunks = []
        for c in in_groupby.chunks:
            new_op = op.copy().reset_key()

            new_index = parse_index(pd.RangeIndex(-1), c.key)
            if op.output_types[0] == OutputType.dataframe:
                chunks.append(
                    new_op.new_chunk(
                        [c],
                        index=c.index,
                        shape=(np.nan, len(out_df.dtypes)),
                        dtypes=out_df.dtypes,
                        columns_value=out_df.columns_value,
                        index_value=new_index,
                    )
                )
        new_op = op.copy().reset_key()
        kw = out_df.params.copy()
        kw["chunks"] = chunks
        if op.output_types[0] == OutputType.dataframe:
            kw["nsplits"] = ((np.nan,) * len(chunks), (len(out_df.dtypes),))
        return new_op.new_tileables([in_groupby], **kw)

    @classmethod
    def execute(cls, ctx, op: "GroupByFillOperand"):
        in_data = ctx[op.inputs[0].key]
        out_chunk = op.outputs[0]

        if not in_data or in_data.empty:
            ctx[out_chunk.key] = (
                build_empty_df(out_chunk.dtypes)
                if op.output_types[0] == OutputType.dataframe
                else build_empty_series(out_chunk.dtype, name=out_chunk.name)
            )
            return

        func_name = getattr(op, "_func_name")
        if func_name == "fillna":
            ctx[out_chunk.key] = in_data.fillna(
                value=op.value,
                method=op.method,
                axis=op.axis,
                limit=op.limit,
                downcast=op.downcast,
            )
        else:
            result = getattr(in_data, func_name)(limit=op.limit)
            if result.ndim == 2:
                ctx[out_chunk.key] = result.astype(out_chunk.dtypes, copy=False)
            else:
                ctx[out_chunk.key] = result.astype(out_chunk.dtype, copy=False)


class GroupByFFill(GroupByFillOperand):
    _op_type_ = opcodes.FILL_NA
    _func_name = "ffill"


class GroupByBFill(GroupByFillOperand):
    _op_type = opcodes.FILL_NA
    _func_name = "bfill"


class GroupByFillNa(GroupByFillOperand):
    _op_type = opcodes.FILL_NA
    _func_name = "fillna"


def ffill(groupby, limit=None):
    op = GroupByFFill(limit=limit)
    return op(groupby)


def bfill(groupby, limit=None):
    op = GroupByBFill(limit=limit)
    return op(groupby)


def fillna(groupby, value=None, method=None, axis=None, limit=None, downcast=None):
    op = GroupByFillNa(
        value=value, method=method, axis=axis, limit=limit, downcast=downcast
    )
    return op(groupby)
