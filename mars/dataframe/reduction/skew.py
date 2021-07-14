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


class DataFrameSkew(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = opcodes.SKEW
    _func_name = 'skew'

    _bias = BoolField('bias')

    def __init__(self, bias=None, **kw):
        super().__init__(_bias=bias, **kw)

    @property
    def bias(self):
        return self._bias

    @classmethod
    def get_reduction_callable(cls, op):
        from .aggregation import where_function
        skipna, bias = op.skipna, op.bias

        def skew(x):
            cnt = x.count()
            mean = x.mean(skipna=skipna)
            divided = (x ** 3).mean(skipna=skipna) \
                - 3 * (x ** 2).mean(skipna=skipna) * mean \
                + 2 * mean ** 3
            var = x.var(skipna=skipna, ddof=0)
            val = where_function(var > 0, divided / var ** 1.5, np.nan)
            if not bias:
                val = where_function((var > 0) & (cnt > 2), val * ((cnt * (cnt - 1)) ** 0.5 / (cnt - 2)), np.nan)
            return val

        return skew


def skew_series(df, axis=None, skipna=None, level=None, combine_size=None, bias=False, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSkew(axis=axis, skipna=skipna, level=level, combine_size=combine_size,
                       bias=bias, output_types=[OutputType.scalar], use_inf_as_na=use_inf_as_na,
                       method=method)
    return op(df)


def skew_dataframe(df, axis=None, skipna=None, level=None, numeric_only=None, combine_size=None,
                   bias=False, method=None):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSkew(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                       bias=bias, combine_size=combine_size, output_types=[OutputType.series],
                       use_inf_as_na=use_inf_as_na, method=method)
    return op(df)
