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


class DataFrameSubtract(DataFrameBinopUfunc):
    _op_type_ = OperandDef.SUB

    _func_name = 'sub'
    _rfunc_name = 'rsub'

    @classproperty
    def _operator(self):
        return operator.sub

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorSubtract
        return TensorSubtract


_sub_example = """
>>> a.subtract(b, fill_value=0).execute()
a    0.0
b    1.0
c    1.0
d   -1.0
e    NaN
dtype: float64
"""


@bin_arithmetic_doc('Subtraction', equiv='-', series_example=_sub_example)
def subtract(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameSubtract(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


@bin_arithmetic_doc('Subtraction', equiv='-', series_example=_sub_example)
def rsubtract(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameSubtract(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
