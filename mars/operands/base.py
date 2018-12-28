#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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


from .core import Operand
from .. import opcodes as OperandDef
from ..serialize import ValueType, AnyField, KeyField, ListField, TupleField, DataTypeField, \
    StringField, Int32Field, Int64Field, BoolField


class HasInput(Operand):
    __slots__ = ()

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super(HasInput, self)._set_inputs(inputs)
        self._input = self._inputs[0]


class Reshape(HasInput):
    _op_type_ = OperandDef.RESHAPE

    _input = KeyField('input')
    _newshape = TupleField('newshape', ValueType.int64)

    @property
    def newshape(self):
        return self._newshape


class CopyTo(Operand):
    _op_type_ = OperandDef.COPYTO

    _src = KeyField('src')
    _dst = KeyField('dest')
    _casting = StringField('casting')
    _where = KeyField('where')

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def casting(self):
        return self._casting

    @property
    def where(self):
        return self._where


class Astype(HasInput):
    _op_type_ = OperandDef.ASTYPE

    _input = KeyField('input')
    _dtype = DataTypeField('dtype')
    _casting = StringField('casting')

    @property
    def dtype(self):
        return self._dtype

    @property
    def casting(self):
        return self._casting


class Transpose(HasInput):
    _op_type_ = OperandDef.TRANSPOSE

    _input = KeyField('input')
    _axes = ListField('axes', ValueType.int32)

    @property
    def axes(self):
        return getattr(self, '_axes', None)


class Slice(HasInput):
    _op_type_ = OperandDef.SLICE

    _input = KeyField('input')
    _slices = ListField('slices')

    @property
    def slices(self):
        return self._slices


class Index(HasInput):
    _op_type_ = OperandDef.INDEX

    _input = KeyField('input')
    _indexes = ListField('indexes')

    @property
    def indexes(self):
        return self._indexes


class IndexSetValue(HasInput):
    _op_type_ = OperandDef.INDEXSETVALUE

    _input = KeyField('input')
    _indexes = ListField('indexes')
    _value = AnyField('value')

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value


class UnravelIndex(HasInput):
    _op_type_ = OperandDef.UNRAVEL_INDEX

    _input = KeyField('input')
    _dims = TupleField('dims', ValueType.int32)

    @property
    def dims(self):
        return self._dims

    @property
    def output_limit(self):
        return float('inf')


class Split(HasInput):
    _op_type_ = OperandDef.ARRAY_SPLIT

    _input = KeyField('input')
    _indices_or_sections = AnyField('indices_or_sections')
    _axis = Int32Field('axis')

    @property
    def indices_or_sections(self):
        return self._indices_or_sections

    @property
    def axis(self):
        return getattr(self, '_axis', 0)

    @property
    def output_limit(self):
        return float('inf')


class BroadcastTo(HasInput):
    _op_type_ = OperandDef.BROADCAST_TO

    _input = KeyField('input')
    _shape = TupleField('shape')

    @property
    def shape(self):
        return self._shape


class Where(Operand):
    _op_type_ = OperandDef.WHERE

    _condition = KeyField('condition')
    _x = KeyField('x')
    _y = KeyField('y')

    @property
    def condition(self):
        return self._condition

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class Argwhere(HasInput):
    _op_type_ = OperandDef.ARGWHERE

    _input = KeyField('input')


class Squeeze(HasInput):
    _op_type_ = OperandDef.SQUEEZE

    _input = KeyField('input')
    _axis = TupleField('axis', ValueType.int32)

    @property
    def axis(self):
        return self._axis


class Digitize(HasInput):
    _op_type_ = OperandDef.DIGITIZE

    _input = KeyField('input')
    _bins = AnyField('bins')
    _right = BoolField('right')

    @property
    def bins(self):
        return self._bins

    @property
    def right(self):
        return self._right


class Repeat(HasInput):
    _op_type_ = OperandDef.REPEAT

    _input = KeyField('input')
    _repeats = AnyField('repeats')
    _axis = Int32Field('axis')

    @property
    def repeats(self):
        return self._repeats

    @property
    def axis(self):
        return self._axis


class IsIn(Operand):
    _op_type_ = OperandDef.ISIN

    _element = KeyField('element')
    _test_elements = KeyField('test_elements')
    _assume_unique = BoolField('assume_unique')
    _invert = BoolField('invert')

    @property
    def element(self):
        return self._element

    @property
    def test_elements(self):
        return self._test_elements

    @property
    def assume_unique(self):
        return self._assume_unique

    @property
    def invert(self):
        return self._invert


class SwapAxes(HasInput):
    _op_type_ = OperandDef.SWAPAXES

    _input = KeyField('input')
    _axis1 = Int32Field('axis1')
    _axis2 = Int32Field('axis2')

    @property
    def axis1(self):
        return self._axis1

    @property
    def axis2(self):
        return self._axis2


class Stack(Operand):
    _op_type_ = OperandDef.STACK

    _axis = Int32Field('axis')

    @property
    def axis(self):
        return self._axis


class Rechunk(HasInput):
    _op_type_ = OperandDef.RECHUNK

    _input = KeyField('input')
    _chunk_size = AnyField('chunk_size')
    _threshold = Int32Field('threshold')
    _chunk_size_limit = Int64Field('chunk_size_limit')

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def threshold(self):
        return self._threshold

    @property
    def chunk_size_limit(self):
        return self._chunk_size_limit
