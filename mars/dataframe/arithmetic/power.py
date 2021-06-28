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

import operator

from ... import opcodes as OperandDef
from ...utils import classproperty
from .core import DataFrameBinopUfunc
from .docstring import bin_arithmetic_doc


class DataFramePower(DataFrameBinopUfunc):
    _op_type_ = OperandDef.POW

    _func_name = 'pow'
    _rfunc_name = 'rpow'

    @classproperty
    def _operator(self):
        return operator.pow

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorPower
        return TensorPower


_pow_example = """
>>> a.pow(b, fill_value=0).execute()
a    1.0
b    1.0
c    1.0
d    0.0
e    NaN
dtype: float64
"""


@bin_arithmetic_doc('Exponential power', op_name='pow', equiv='**', series_example=_pow_example)
def power(df, other, axis='columns', level=None, fill_value=None):
    op = DataFramePower(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


@bin_arithmetic_doc('Exponential power', op_name='rpow', equiv='**', series_example=_pow_example)
def rpower(df, other, axis='columns', level=None, fill_value=None):
    op = DataFramePower(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
