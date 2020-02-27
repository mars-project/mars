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

from ... import opcodes as OperandDef
from ...config import options
from ...utils import lazy_import
from .core import DataFrameReductionOperand, DataFrameReductionMixin, ObjectType

cudf = lazy_import('cudf', globals=globals())


class DataFrameMean(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.MEAN
    _func_name = 'mean'

    @classmethod
    def _execute_map(cls, ctx, op):
        cls._execute_map_with_count(ctx, op, reduction_func='sum')

    @classmethod
    def _execute_combine(cls, ctx, op):
        cls._execute_combine_with_count(ctx, op, reduction_func='sum')

    @classmethod
    def _execute_agg(cls, ctx, op):
        in_data, concat_count = ctx[op.inputs[0].key]
        count = concat_count.sum(axis=op.axis)
        r = cls._execute_reduction(in_data, op, reduction_func='sum')
        ctx[op.outputs[0].key] = r / count


def mean_series(df, axis=None, skipna=None, level=None, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameMean(axis=axis, skipna=skipna, level=level, combine_size=combine_size,
                       object_type=ObjectType.scalar, use_inf_as_na=use_inf_as_na)
    return op(df)


def mean_dataframe(df, axis=None, skipna=None, level=None, numeric_only=None, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameMean(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                       combine_size=combine_size, object_type=ObjectType.series,
                       use_inf_as_na=use_inf_as_na)
    return op(df)
