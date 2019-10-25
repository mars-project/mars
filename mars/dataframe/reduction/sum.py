# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
from ...serialize import BoolField
from ..operands import ObjectType
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameSum(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.SUM
    _func_name = 'sum'

    def __init__(self, numeric_only=None, object_type=ObjectType.series, **kw):
        super(DataFrameSum, self).__init__(_numeric_only=numeric_only,
                                           _object_type=object_type, **kw)


def sum_series(df, axis=None, skipna=None, level=None, min_count=0, combine_size=None):
    op = DataFrameSum(axis=axis, skipna=skipna, level=level, min_count=min_count, combine_size=combine_size)
    return op(df)


def sum_dataframe(df, axis=None, skipna=None, level=None, min_count=0, numeric_only=None, combine_size=None):
    op = DataFrameSum(axis=axis, skipna=skipna, level=level, min_count=min_count,
                      numeric_only=numeric_only, combine_size=combine_size)
    return op(df)
