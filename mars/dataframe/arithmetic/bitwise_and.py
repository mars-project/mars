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

import operator

from ... import opcodes as OperandDef
from ...utils import classproperty
from .core import DataFrameBinopUfunc


class DataFrameAnd(DataFrameBinopUfunc):
    _op_type_ = OperandDef.AND

    _bit_func_name = '__and__'
    _bit_rfunc_name = '__rand__'

    @classproperty
    def _operator(self):
        return operator.and_

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorBitand
        return TensorBitand


def bitand(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameAnd(axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    return op(df, other)


def rbitand(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameAnd(axis=axis, level=level, fill_value=fill_value, lhs=other, rhs=df)
    return op.rcall(df, other)
