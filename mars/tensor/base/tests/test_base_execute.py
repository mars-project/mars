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
from mars.tensor.datasource import tensor, ones
from mars.tensor.base import broadcast_to, atleast_1d, atleast_2d, atleast_3d


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testBroadcastToExecution(self):
        raw = np.random.random((10, 5, 1))
        arr = tensor(raw, chunk_size=2)
        arr2 = broadcast_to(arr, (5, 10, 5, 6))

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], np.broadcast_to(raw, (5, 10, 5, 6))))

    def testAtleast1dExecution(self):
        x = 1
        y = ones(3, chunk_size=2)
        z = ones((3, 4), chunk_size=2)

        t = atleast_1d(x, y, z)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in t]

        self.assertTrue(np.array_equal(res[0], np.array([1])))
        self.assertTrue(np.array_equal(res[1], np.ones(3)))
        self.assertTrue(np.array_equal(res[2], np.ones((3, 4))))

    def testAtleast2dExecution(self):
        x = 1
        y = ones(3, chunk_size=2)
        z = ones((3, 4), chunk_size=2)

        t = atleast_2d(x, y, z)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in t]

        self.assertTrue(np.array_equal(res[0], np.array([[1]])))
        self.assertTrue(np.array_equal(res[1], np.atleast_2d(np.ones(3))))
        self.assertTrue(np.array_equal(res[2], np.ones((3, 4))))

    def testAtleast3dExecution(self):
        x = 1
        y = ones(3, chunk_size=2)
        z = ones((3, 4), chunk_size=2)

        t = atleast_3d(x, y, z)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in t]

        self.assertTrue(np.array_equal(res[0], np.atleast_3d(x)))
        self.assertTrue(np.array_equal(res[1], np.atleast_3d(np.ones(3))))
        self.assertTrue(np.array_equal(res[2], np.atleast_3d(np.ones((3, 4)))))
