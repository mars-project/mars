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
import scipy.sparse as sps

from mars.executor import Executor
from mars.tensor.datasource import tensor, empty
from mars.tensor.merge import concatenate, stack, hstack, vstack, dstack, column_stack


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testConcatenateExecution(self):
        a_data = np.random.rand(10, 20, 30)
        b_data = np.random.rand(10, 20, 40)
        c_data = np.random.rand(10, 20, 50)

        a = tensor(a_data, chunk_size=5)
        b = tensor(b_data, chunk_size=6)
        c = tensor(c_data, chunk_size=7)

        d = concatenate([a, b, c], axis=-1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.concatenate([a_data, b_data, c_data], axis=-1)
        self.assertTrue(np.array_equal(res, expected))

        a_data = sps.random(10, 30)
        b_data = sps.rand(10, 40)
        c_data = sps.rand(10, 50)

        a = tensor(a_data, chunk_size=5)
        b = tensor(b_data, chunk_size=6)
        c = tensor(c_data, chunk_size=7)

        d = concatenate([a, b, c], axis=-1)
        res = self.executor.execute_tensor(d, concat=True)[0]
        expected = np.concatenate([a_data.A, b_data.A, c_data.A], axis=-1)
        self.assertTrue(np.array_equal(res.toarray(), expected))

    def testStackExecution(self):
        raw = [np.random.randn(3, 4) for _ in range(10)]
        arrs = [tensor(a, chunk_size=3) for a in raw]

        arr2 = stack(arrs)
        res = self.executor.execute_tensor(arr2, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw)))

        arr3 = stack(arrs, axis=1)
        res = self.executor.execute_tensor(arr3, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw, axis=1)))

        arr4 = stack(arrs, axis=2)
        res = self.executor.execute_tensor(arr4, concat=True)
        self.assertTrue(np.array_equal(res[0], np.stack(raw, axis=2)))

        raw2 = [np.asfortranarray(np.random.randn(3, 4)) for _ in range(10)]
        arr5 = [tensor(a, chunk_size=3) for a in raw2]

        arr6 = stack(arr5)
        res = self.executor.execute_tensor(arr6, concat=True)[0]
        expected = np.stack(raw2).copy('A')
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

        arr7 = stack(arr5, out=empty((10, 3, 4), order='F'))
        res = self.executor.execute_tensor(arr7, concat=True)[0]
        expected = np.stack(raw2, out=np.empty((10, 3, 4), order='F')).copy('A')
        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])

    def testHStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(20)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = hstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.hstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(10, 5)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = hstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.hstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testVStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(10)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = vstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.vstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(5, 20)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = vstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.vstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testDStackExecution(self):
        a_data = np.random.rand(10)
        b_data = np.random.rand(10)

        a = tensor(a_data, chunk_size=4)
        b = tensor(b_data, chunk_size=4)

        c = dstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.dstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

        a_data = np.random.rand(10, 20)
        b_data = np.random.rand(10, 20)

        a = tensor(a_data, chunk_size=3)
        b = tensor(b_data, chunk_size=4)

        c = dstack([a, b])
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.dstack([a_data, b_data])
        self.assertTrue(np.array_equal(res, expected))

    def testColumnStackExecution(self):
        a_data = np.array((1, 2, 3))
        b_data = np.array((2, 3, 4))
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=2)

        c = column_stack((a, b))
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.column_stack((a_data, b_data))
        np.testing.assert_equal(res, expected)

        a_data = np.random.rand(4, 2, 3)
        b_data = np.random.rand(4, 2, 3)
        a = tensor(a_data, chunk_size=1)
        b = tensor(b_data, chunk_size=2)

        c = column_stack((a, b))
        res = self.executor.execute_tensor(c, concat=True)[0]
        expected = np.column_stack((a_data, b_data))
        np.testing.assert_equal(res, expected)
