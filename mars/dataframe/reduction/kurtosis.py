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

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...serialization.serializables import BoolField
from .core import DataFrameReductionOperand, DataFrameReductionMixin


class DataFrameKurtosis(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = opcodes.KURTOSIS
    _func_name = 'kurt'

    _bias = BoolField('bias')
    _fisher = BoolField('fisher')

    def __init__(self, bias=None, fisher=None, **kw):
        super().__init__(_bias=bias, _fisher=fisher, **kw)

    @property
    def bias(self):
        return self._bias

    @property
    def fisher(self):
        return self._fisher

    @classmethod
    def get_reduction_callable(cls, op):
        from .aggregation import where_function
        skipna, bias, fisher = op.skipna, op.bias, op.fisher

        def kurt(x):
            cnt = x.count()
            mean = x.mean(skipna=skipna)
            divided = (x ** 4).mean(skipna=skipna) \
                - 4 * (x ** 3).mean(skipna=skipna) * mean \
                + 6 * (x ** 2).mean(skipna=skipna) * mean ** 2 \
                - 3 * mean ** 4
            var = x.var(skipna=skipna, ddof=0)
            val = where_function(var > 0, divided / var ** 2, np.nan)
            if not bias:
                val = where_function((var > 0) & (cnt > 3),
                                     (val * (cnt ** 2 - 1) - 3 * (cnt - 1) ** 2) / (cnt - 2) / (cnt - 3), np.nan)
            if not fisher:
                val += 3
            return val

        return kurt


def kurt_series(df, axis=None, skipna=None, level=None, combine_size=None, bias=False,
                fisher=True, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameKurtosis(axis=axis, skipna=skipna, level=level, combine_size=combine_size,
                           bias=bias, fisher=fisher, output_types=[OutputType.scalar],
                           use_inf_as_na=use_inf_as_na, method=method)
    return op(df)


def kurt_dataframe(df, axis=None, skipna=None, level=None, numeric_only=None, combine_size=None,
                   bias=False, fisher=True, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameKurtosis(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                           bias=bias, fisher=fisher, combine_size=combine_size,
                           output_types=[OutputType.series], use_inf_as_na=use_inf_as_na,
                           method=method)
    return op(df)
