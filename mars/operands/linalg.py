#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


from .base import HasInput
from ..operands.core import Operand
from .. import opcodes as OperandDef
from ..serialize import ValueType, KeyField, StringField, TupleField, BoolField, AnyField


class QR(HasInput):
    _op_type_ = OperandDef.QR

    _input = KeyField('input')
    _method = StringField('method')

    @property
    def method(self):
        return self._method

    @property
    def output_limit(self):
        return 2


class SVD(HasInput):
    _op_type_ = OperandDef.SVD

    _input = KeyField('input')
    _method = StringField('method')

    @property
    def method(self):
        return self._method

    @property
    def output_limit(self):
        return 3


class Cholesky(HasInput):
    _op_type_ = OperandDef.CHOLESKY

    _input = KeyField('input')
    _lower = BoolField('lower')

    @property
    def lower(self):
        return self._lower


class SolveTriangular(Operand):
    _op_type_ = OperandDef.SOLVE_TRIANGULAR

    _a = KeyField('a')
    _b = KeyField('b')
    _lower = BoolField('lower')

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def lower(self):
        return self._lower


class Inv(HasInput):
    _op_type_ = OperandDef.INV

    _input = KeyField('input')


class LU(HasInput):
    _op_type_ = OperandDef.LU

    _input = KeyField('input')

    @property
    def output_limit(self):
        return 3


class Norm(HasInput):
    _op_type_ = OperandDef.NORM

    _input = KeyField('input')
    _ord = AnyField('ord')
    _axis = TupleField('axis', ValueType.int32)
    _keepdims = BoolField('keepdims')

    @property
    def ord(self):
        return getattr(self, '_ord', None)

    @property
    def axis(self):
        return self._axis

    @property
    def keepdims(self):
        return self._keepdims
