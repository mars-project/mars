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
from ...serialization.serializables import AnyField
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameCustomReduction(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.CUSTOM_REDUCTION
    _func_name = 'custom_reduction'

    _custom_reduction = AnyField('custom_reduction')

    def __init__(self, custom_reduction=None, **kw):
        super().__init__(_custom_reduction=custom_reduction, **kw)

    @property
    def custom_reduction(self):
        return self._custom_reduction

    @property
    def is_atomic(self):
        return True

    def get_reduction_args(self, axis=None):
        return dict()


def build_custom_reduction_result(df, custom_reduction_obj, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    output_type = OutputType.series if df.ndim == 2 else OutputType.scalar
    op = DataFrameCustomReduction(custom_reduction=custom_reduction_obj, output_types=[output_type],
                                  use_inf_as_na=use_inf_as_na, method=method)
    return op(df)
