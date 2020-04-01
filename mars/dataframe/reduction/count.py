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

import pandas as pd

from ... import opcodes as OperandDef
from ...config import options
from ...utils import lazy_import
from .core import DataFrameReductionOperand, DataFrameReductionMixin, ObjectType

cudf = lazy_import('cudf', globals=globals())


class DataFrameCount(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.COUNT
    _func_name = 'count'

    @classmethod
    def _execute_map(cls, ctx, op):
        cls._execute_without_count(ctx, op)

    @classmethod
    def _execute_combine(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        count_sum = in_data.sum(axis=op.axis)
        if isinstance(in_data, xdf.Series):
            ctx[op.outputs[0].key] = count_sum
        else:
            ctx[op.outputs[0].key] = xdf.DataFrame(count_sum) if op.axis == 1 else xdf.DataFrame(count_sum).transpose()

    @classmethod
    def _execute_agg(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = in_data.sum(axis=op.axis)


def count_series(series, level=None, combine_size=None, **kw):
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    op = DataFrameCount(level=level, combine_size=combine_size, object_type=ObjectType.scalar,
                        use_inf_as_na=use_inf_as_na)
    return op(series)


def count_dataframe(df, axis=0, level=None, numeric_only=False, combine_size=None, **kw):
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    op = DataFrameCount(axis=axis, level=level, numeric_only=numeric_only, combine_size=combine_size,
                        object_type=ObjectType.series, use_inf_as_na=use_inf_as_na)
    return op(df)
