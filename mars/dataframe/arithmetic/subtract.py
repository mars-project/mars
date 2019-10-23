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

import operator

from ... import opcodes as OperandDef
from ...utils import classproperty
from ..operands import DataFrameBinOp
from ..utils import wrap_sequence
from .core import DataFrameBinOpMixin


class DataFrameSubtract(DataFrameBinOp, DataFrameBinOpMixin):
    _op_type_ = OperandDef.SUB

    _func_name = 'sub'
    _rfunc_name = 'rsub'

    @classproperty
    def _operator(self):
        return operator.sub


def subtract(df, other, axis='columns', level=None, fill_value=None):
    other = wrap_sequence(other)
    op = DataFrameSubtract(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


def rsubtract(df, other, axis='columns', level=None, fill_value=None):
    other = wrap_sequence(other)
    op = DataFrameSubtract(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
