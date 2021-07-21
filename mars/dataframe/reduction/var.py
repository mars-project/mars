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
from ...serialization.serializables import Int32Field
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameVar(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.VAR
    _func_name = 'var'

    _ddof = Int32Field('ddof')

    def __init__(self, ddof=None, **kw):
        super().__init__(_ddof=ddof, **kw)

    @property
    def ddof(self):
        return self._ddof

    @classmethod
    def get_reduction_callable(cls, op):
        skipna, ddof = op.skipna, op.ddof

        def var(x):
            cnt = x.count()
            if ddof == 0:
                return (x ** 2).mean(skipna=skipna) - (x.mean(skipna=skipna)) ** 2
            return ((x ** 2).sum(skipna=skipna) - x.sum(skipna=skipna) ** 2 / cnt) / (cnt - ddof)

        return var


def var_series(series, axis=None, skipna=None, level=None, ddof=1,
               combine_size=None, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameVar(axis=axis, skipna=skipna, level=level, ddof=ddof,
                      combine_size=combine_size, output_types=[OutputType.scalar],
                      use_inf_as_na=use_inf_as_na, method=method)
    return op(series)


def var_dataframe(df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None,
                  combine_size=None, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameVar(axis=axis, skipna=skipna, level=level, ddof=ddof,
                      numeric_only=numeric_only, combine_size=combine_size,
                      output_types=[OutputType.series], use_inf_as_na=use_inf_as_na,
                      method=method)
    return op(df)
