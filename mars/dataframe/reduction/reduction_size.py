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
from ...core import OutputType
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameSize(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.REDUCTION_SIZE
    _func_name = 'size'

    @property
    def is_atomic(self):
        return True


def size_series(df):
    op = DataFrameSize(output_types=[OutputType.scalar])
    return op(df)


def size_dataframe(df):
    op = DataFrameSize(output_types=[OutputType.series])
    return op(df)
