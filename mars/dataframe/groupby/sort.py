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

from ... import opcodes as OperandDef
from ...core import OutputType
from ...core.operand import MapReduceOperand, OperandStage
from ...serialization.serializables import (
    Int32Field,
    ListField,
)
from ...utils import (
    lazy_import,
)
from ..operands import DataFrameOperandMixin
from ..sort.psrs import DataFramePSRSChunkOperand

cudf = lazy_import("cudf", globals=globals())


def _series_to_df(in_series, xdf):
    in_df = in_series.to_frame()
    if in_series.name is not None:
        in_df.columns = xdf.Index([in_series.name])
    return in_df


class DataFrameGroupbyConcatPivot(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_PIVOT

    @property
    def output_limit(self):
        return 1

    @classmethod
    def execute(cls, ctx, op: "DataFrameGroupbyConcatPivot"):
        inputs = [ctx[c.key] for c in op.inputs if len(ctx[c.key]) > 0]

        xdf = pd if isinstance(inputs[0], (pd.DataFrame, pd.Series)) else cudf

        a = xdf.concat(inputs, axis=0)
        a = a.sort_index()
        index = a.index.drop_duplicates()

        p = len(inputs)
        if len(index) < p:
            num = p // len(index) + 1
            index = index.append([index] * (num - 1))

        index = index.sort_values()

        values = index.values

        slc = np.linspace(
            p - 1, len(index) - 1, num=len(op.inputs) - 1, endpoint=False
        ).astype(int)
        out = values[slc]
        ctx[op.outputs[-1].key] = out


class DataFramePSRSGroupbySample(DataFramePSRSChunkOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_REGULAR_SAMPLE

    @property
    def output_limit(self):
        return 1

    @classmethod
    def execute(cls, ctx, op: "DataFramePSRSGroupbySample"):
        a = ctx[op.inputs[0].key][0]
        xdf = pd if isinstance(a, (pd.DataFrame, pd.Series)) else cudf
        if isinstance(a, xdf.Series) and op.output_types[0] == OutputType.dataframe:
            a = _series_to_df(a, xdf)

        n = op.n_partition
        if a.shape[0] < n:
            num = n // a.shape[0] + 1
            a = xdf.concat([a] * num).sort_index()

        w = a.shape[0] * 1.0 / (n + 1)

        slc = np.linspace(max(w - 1, 0), a.shape[0] - 1, num=n, endpoint=False).astype(
            int
        )

        out = a.iloc[slc]
        if op.output_types[0] == OutputType.series and out.ndim == 2:
            assert out.shape[1] == 1
            out = out.iloc[:, 0]
        ctx[op.outputs[-1].key] = out


class DataFrameGroupbySortShuffle(MapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_SORT_SHUFFLE

    # for shuffle map
    by = ListField("by")
    n_partition = Int32Field("n_partition")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @property
    def output_limit(self):
        return 1

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameGroupbySortShuffle"):
        df, pivots = [ctx[c.key] for c in op.inputs]
        out = op.outputs[0]

        def _get_out_df(p_index, in_df):
            if p_index == 0:
                out_df = in_df.loc[: pivots[p_index]]
            elif p_index == op.n_partition - 1:
                out_df = in_df.loc[pivots[p_index - 1] :].drop(
                    index=pivots[p_index - 1], errors="ignore"
                )
            else:
                out_df = in_df.loc[pivots[p_index - 1] : pivots[p_index]].drop(
                    index=pivots[p_index - 1], errors="ignore"
                )
            return out_df

        for i in range(op.n_partition):
            index = (i, 0)
            out_df = tuple(_get_out_df(i, x) for x in df)
            ctx[out.key, index] = out_df

    @classmethod
    def _execute_reduce(cls, ctx, op: "DataFrameGroupbySortShuffle"):
        raw_inputs = list(op.iter_mapper_data(ctx, pop=False))
        by = op.by
        xdf = cudf if op.gpu else pd

        r = []

        tuple_len = len(raw_inputs[0])
        for i in range(tuple_len):
            r.append(xdf.concat([inp[i] for inp in raw_inputs], axis=0))
        r = tuple(r)

        ctx[op.outputs[0].key] = r + (by,)

    @classmethod
    def estimate_size(cls, ctx, op: "DataFrameGroupbySortShuffle"):
        super().estimate_size(ctx, op)
        result = ctx[op.outputs[0].key]
        if op.stage == OperandStage.reduce:
            ctx[op.outputs[0].key] = (result[0], result[1] * 1.5)
        else:
            ctx[op.outputs[0].key] = result

    @classmethod
    def execute(cls, ctx, op: "DataFrameGroupbySortShuffle"):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)
