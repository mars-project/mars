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

from ... import opcodes as OperandDef
from ...config import options
from ...core import OutputType
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameCount(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.COUNT
    _func_name = 'count'

    @property
    def is_atomic(self):
        return True

    @classmethod
    def get_reduction_callable(cls, op):
        skipna, numeric_only = op.skipna, op.numeric_only

        def count(value):
            if value.ndim == 1:
                return value.count()
            return value.count(skipna=skipna, numeric_only=numeric_only)

        return count


def count_series(series, level=None, combine_size=None, **kw):
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    method = kw.pop('method', None)
    op = DataFrameCount(level=level, combine_size=combine_size, output_types=[OutputType.scalar],
                        use_inf_as_na=use_inf_as_na, method=method)
    return op(series)


def count_dataframe(df, axis=0, level=None, numeric_only=False, combine_size=None, **kw):
    use_inf_as_na = kw.pop('_use_inf_as_na', options.dataframe.mode.use_inf_as_na)
    method = kw.pop('method', None)
    op = DataFrameCount(axis=axis, level=level, numeric_only=numeric_only, combine_size=combine_size,
                        output_types=[OutputType.series], use_inf_as_na=use_inf_as_na, method=method)
    return op(df)
