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

import json

from mars.compat import OrderedDict
from mars.operands import AddConstant, MulConstant, Index, Log, Operand
from mars.serialize import AttributeAsDictKey, serializes, deserializes, \
    ProtobufSerializeProvider, JsonSerializeProvider, Int64Field
from mars.tests.core import TestBase


class MockKeyObject(AttributeAsDictKey):
    _data = Int64Field('data')

    def __eq__(self, other):
        if not isinstance(other, MockKeyObject):
            return False

        return self._data == other.data

    @property
    def data(self):
        return self._data


class MockOperandMixin(object):
    __slots__ = ()
    _op_module_ = 'mock'


class MockAddConstantMixin(AddConstant, MockOperandMixin):
    _init_update_key_ = True


class MockMulConstantMixin(MulConstant, MockOperandMixin):
    _init_update_key_ = True


class MockIndexMixin(Index, MockOperandMixin):
    _init_update_key_ = True


class MockLog(Log, MockOperandMixin):
    _init_update_key_ = True


class Test(TestBase):
    def setUp(self):
        self.attrs = dict()
        self.j = JsonSerializeProvider()
        self.p = ProtobufSerializeProvider()

    def pickle_unpickle(self, provider, *objs):
        clses = [type(obj) for obj in objs]
        serials = serializes(provider, objs)
        if provider == self.j:
            serials = [json.loads(json.dumps(s), object_hook=OrderedDict) for s in serials]
        res = deserializes(provider, clses, serials)
        if len(objs) == 1:
            return res[0]
        return res

    def testConstantOp(self):
        data = MockKeyObject(_data=64)
        op = MockAddConstantMixin(_lhs=data, _rhs=1, _inputs=[data])

        self.assertEqual(op.inputs, [data])

        _, op2 = self.pickle_unpickle(self.p, data, op)

        self.assertEqual(op.lhs.data, op2.lhs.data)
        self.assertEqual(op.rhs, op2.rhs)
        self.assertEqual(op2.constant, [op.rhs])
        self.assertIs(op2.reverse, False)
        self.assertEqual(op.inputs, op2.inputs)

        _, op2 = self.pickle_unpickle(self.j, data, op)

        self.assertEqual(op.lhs.data, op2.lhs.data)
        self.assertEqual(op.rhs, op2.rhs)
        self.assertEqual(op2.constant, [op.rhs])
        self.assertIs(op2.reverse, False)

        op = MockMulConstantMixin(_lhs=100, _rhs=data)

        _, op2 = self.pickle_unpickle(self.p, data, op)

        self.assertEqual(op.lhs, op2.lhs)
        self.assertEqual(op.rhs.data, op2.rhs.data)
        self.assertEqual(op2.constant, [op.lhs])
        self.assertIs(op2.reverse, True)

        _, op2 = self.pickle_unpickle(self.j, data, op)

        self.assertEqual(op.lhs, op2.lhs)
        self.assertEqual(op.rhs.data, op2.rhs.data)
        self.assertEqual(op2.constant, [op.lhs])
        self.assertIs(op2.reverse, True)

    def testIndex(self):
        data = MockKeyObject(_data=10)
        mask = MockKeyObject(_data=20)
        idx = MockIndexMixin(_input=data, _indexes=[slice(5), mask])

        _, _, idx2 = self.pickle_unpickle(self.p, data, mask, idx)

        self.assertEqual(idx.input.data, idx2.input.data)
        self.assertEqual(idx.indexes[0], idx2.indexes[0])
        self.assertEqual(idx.indexes[1].data, idx2.indexes[1].data)

    def testDeserialzeCls(self):
        data = MockKeyObject(_data=10)
        op = MockLog(_input=data)

        serials = serializes(self.p, [data, op])
        _, op2 = deserializes(self.p, [MockKeyObject, Operand], serials)

        self.assertIsInstance(op2, Log)
        self.assertBaseEqual(op, op2)
