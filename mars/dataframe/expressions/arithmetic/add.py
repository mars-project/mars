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

from .... import opcodes as OperandDef
from ....serialize import AnyField, Float64Field
from ....utils import classproperty
from ..core import DataFrameOperand
from .core import DataFrameBinOpMixin


class DataFrameAdd(DataFrameOperand, DataFrameBinOpMixin):
    _op_type_ = OperandDef.ADD

    _axis = AnyField('axis')
    _level = AnyField('level')
    _fill_value = Float64Field('fill_value')

    def __init__(self, axis=None, level=None, fill_value=None, **kw):
        super(DataFrameAdd, self).__init__(_axis=axis, _level=level,
                                           _fill_value=fill_value, **kw)

    @classproperty
    def _operator(self):
        return operator.add

    @property
    def axis(self):
        return self._axis

    @property
    def level(self):
        return self._level

    @property
    def fill_value(self):
        return self._fill_value


def add(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameAdd(axis=axis, level=level, fill_value=fill_value)
    return op(df, other)


def radd(df, other, axis='columns', level=None, fill_value=None):
    op = DataFrameAdd(axis=axis, level=level, fill_value=fill_value)
    return op.rcall(df, other)
