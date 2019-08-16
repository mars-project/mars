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
import unittest
import numpy as np

from mars.executor import Executor
import mars.tensor as mt
from mars.tensor.fuse.jx import JAX_INSTALLED


@unittest.skipIf(not JAX_INSTALLED, 'jax not installed')
class Test(unittest.TestCase):
    # test multiple engines execution
    def setUp(self):
        self.executor = Executor(['jax', 'numexpr'])

    def testUnaryExecution(self):
        executor_numpy = Executor('numpy')
        a = mt.ones((2, 2))
        # a = a * (-1)
        c = mt.abs(a)
        d = mt.abs(c)
        e = mt.abs(d)
        f = mt.abs(e)
        result = self.executor.execute_tensor(f, concat=True)
        expected = executor_numpy.execute_tensor(f, concat=True)
        np.testing.assert_equal(result, expected)
