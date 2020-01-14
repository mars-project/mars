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

from ... import opcodes as OperandDef
from ...utils import classproperty
from .core import DataFrameBinOpMixin, DataFrameBinOp


class DataFrameGreaterEqual(DataFrameBinOp, DataFrameBinOpMixin):
    _op_type_ = OperandDef.GE

    _func_name = 'ge'
    _rfunc_name = 'le'

    @classproperty
    def _operator(self):
        return lambda lhs, rhs: lhs.ge(rhs)


def ge(df, other, axis='columns', level=None):
    op = DataFrameGreaterEqual(axis=axis, level=level, lhs=df, rhs=other)
    return op(df, other)
