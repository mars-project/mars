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


class DataFrameFloorDiv(DataFrameBinopUfunc):
    _op_type_ = OperandDef.FLOORDIV

    _func_name = 'floordiv'
    _rfunc_name = 'rfloordiv'

    @classproperty
    def _operator(self):
        return operator.floordiv

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorFloorDiv
        return TensorFloorDiv


_floordiv_example = """
>>> a.floordiv(b, fill_value=0).execute()
a    1.0
b    NaN
c    NaN
d    0.0
e    NaN
dtype: float64
"""


@bin_arithmetic_doc('Integer division', equiv='//', series_example=_floordiv_example)
def floordiv(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameFloorDiv(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


@bin_arithmetic_doc('Integer division', equiv='//', series_example=_floordiv_example)
def rfloordiv(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameFloorDiv(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
