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

import numpy as np

from mars.operands.random import Randn
from mars.operands.tests.test_core import Test as TestBase


class MockOperandMixin(object):
    _op_module_ = 'mock'


class MockRandn(Randn, MockOperandMixin):
    _init_update_key_ = True


class Test(TestBase):
    def testRandomOp(self):
        state = np.random.RandomState()
        r = MockRandn(_state=state.get_state(), _size=(500, 300))

        r2 = self.pickle_unpickle(self.p, r)

        self.assertBaseEqual(r, r2)

        r2 = self.pickle_unpickle(self.j, r)

        self.assertBaseEqual(r, r2)
