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


from mars.operands.tests.test_core import Test as TestBase, MockKeyObject
from mars.operands import AddConstant


class MockOperandMixin(object):
    _op_module_ = 'mock'


class MockNeAddConst(AddConstant, MockOperandMixin):
    _init_update_key_ = True


class Test(TestBase):
    def testNeAdd(self):
        data = MockKeyObject(_data=64)
        op = MockNeAddConst(_lhs=data, _rhs=1)

        _, op2 = self.pickle_unpickle(self.p, data, op)

        self.assertEqual(op.lhs.data, op2.lhs.data)
        self.assertEqual(op.rhs, op2.rhs)
        self.assertEqual(op2.constant, [op.rhs])
        self.assertIs(op2.reverse, False)

        _, op2 = self.pickle_unpickle(self.j, data, op)

        self.assertEqual(op.lhs.data, op2.lhs.data)
        self.assertEqual(op.rhs, op2.rhs)
        self.assertEqual(op2.constant, [op.rhs])
        self.assertIs(op2.reverse, False)
