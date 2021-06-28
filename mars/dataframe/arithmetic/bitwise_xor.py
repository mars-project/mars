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


class DataFrameXor(DataFrameBinopUfunc):
    _op_type_ = OperandDef.XOR

    _bit_func_name = '__xor__'
    _bit_rfunc_name = '__rxor__'

    @classproperty
    def _operator(self):
        return operator.xor

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorBitxor
        return TensorBitxor


def bitxor(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameXor(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


def rbitxor(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameXor(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
