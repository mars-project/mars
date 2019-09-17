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
import numpy as np

from ... import opcodes as OperandDef
from ...serialize import AnyField, Float64Field
from ...utils import classproperty
from ..operands import DataFrameOperand
from .core import DataFrameBinOpMixin, DATAFRAME_TYPE, SERIES_TYPE


class DataFrameDiv(DataFrameOperand, DataFrameBinOpMixin):
    _op_type_ = OperandDef.DIV

    _axis = AnyField('axis')
    _level = AnyField('level')
    _fill_value = Float64Field('fill_value')
    _lhs = AnyField('lhs')
    _rhs = AnyField('rhs')
    _func_name = AnyField('func_name')

    def __init__(self, func_name=None, axis=None, level=None, fill_value=None, object_type=None, lhs=None, rhs=None, **kw):
        super(DataFrameDiv, self).__init__(_func_name=func_name, _axis=axis, _level=level,
                                           _fill_value=fill_value,
                                           _object_type=object_type, _lhs=lhs, _rhs=rhs, **kw)

    @classproperty
    def _operator(self):
        return operator.div

    @property
    def func_name(self):
        return self._func_name

    @property
    def axis(self):
        return self._axis

    @property
    def level(self):
        return self._level

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def _set_inputs(self, inputs):
        super(DataFrameDiv, self)._set_inputs(inputs)
        if len(self._inputs) == 2:
            self._lhs = self._inputs[0]
            self._rhs = self._inputs[1]
        else:
            if isinstance(self._lhs, (DATAFRAME_TYPE, SERIES_TYPE)):
                self._lhs = self._inputs[0]
            elif np.isscalar(self._lhs):
                self._rhs = self._inputs[0]


def div(df, other, axis='columns', level=None, fill_value=None):
    if isinstance(other, (DATAFRAME_TYPE, SERIES_TYPE)) or np.isscalar(other):
        op = DataFrameDiv(func_name='div', axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    else:
        raise NotImplementedError('Only support div with dataframe, series or scalar!')
    return op(df, other)


def rdiv(df, other, axis='columns', level=None, fill_value=None):
    if isinstance(other, (DATAFRAME_TYPE, SERIES_TYPE)) or np.isscalar(other):
        op = DataFrameDiv(func_name='rdiv', axis=axis, level=level, fill_value=fill_value, lhs=df, rhs=other)
    else:
        raise NotImplementedError('Only support div with dataframe, series or scalar!')
    return op(df, other)
