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

import functools

import numpy as np

from ... import opcodes
from ...config import options
from ...core import OutputType
from .aggregation import where_function
from .core import DataFrameReductionOperand, DataFrameReductionMixin


def _sum_with_count(value, skipna=True, min_count=0):
    if min_count == 0:
        return value.sum(skipna=skipna)
    else:
        return where_function(value.count() >= min_count, value.sum(skipna=skipna), np.nan)


class DataFrameSum(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = opcodes.SUM
    _func_name = 'sum'

    @classmethod
    def _make_agg_object(cls, op):
        fn = functools.partial(_sum_with_count, skipna=op.skipna, min_count=op.min_count)
        fn.__name__ = cls._func_name
        return fn


def sum_series(df, axis=None, skipna=None, level=None, min_count=0, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSum(axis=axis, skipna=skipna, level=level, min_count=min_count,
                      combine_size=combine_size, output_types=[OutputType.scalar],
                      use_inf_as_na=use_inf_as_na)
    return op(df)


def sum_dataframe(df, axis=None, skipna=None, level=None, min_count=0, numeric_only=None, combine_size=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSum(axis=axis, skipna=skipna, level=level, min_count=min_count,
                      numeric_only=numeric_only, combine_size=combine_size,
                      output_types=[OutputType.series], use_inf_as_na=use_inf_as_na)
    return op(df)
