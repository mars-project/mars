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

from ... import opcodes
from ...config import options
from ...core import OutputType
from ...serialize import BoolField
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
    def _make_agg_object(cls, op):
        from .aggregation import skew_function
        pf = functools.partial(skew_function, bias=op.bias, skipna=op.skipna)
        pf.__name__ = cls._func_name
        return pf


def skew_series(df, axis=None, skipna=None, level=None, combine_size=None, bias=False):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSkew(axis=axis, skipna=skipna, level=level, combine_size=combine_size,
                       bias=bias, output_types=[OutputType.scalar], use_inf_as_na=use_inf_as_na)
    return op(df)


def skew_dataframe(df, axis=None, skipna=None, level=None, numeric_only=None, combine_size=None,
                   bias=False):
    use_inf_as_na = options.dataframe.mode.use_inf_as_na
    op = DataFrameSkew(axis=axis, skipna=skipna, level=level, numeric_only=numeric_only,
                       bias=bias, combine_size=combine_size, output_types=[OutputType.series],
                       use_inf_as_na=use_inf_as_na)
    return op(df)
