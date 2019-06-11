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
from mars.tensor.expressions.datasource import tensor, arange
from mars.tensor.expressions.indexing import take, compress, extract, choose, \
    unravel_index, nonzero, flatnonzero
from mars.tensor import mod, stack, hstack
from mars.config import options


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')
        self.old_chunk = options.tensor.chunk_size
        options.tensor.chunk_size = 10

    def tearDown(self):
        options.tensor.chunk_size = self.old_chunk

    def testBoolIndexingExecution(self):
        raw = np.random.random((11, 8, 12, 14))
        arr = tensor(raw, chunk_size=3)

        index = arr < .5
        arr2 = arr[index]
        size_res = self.executor.execute_tensor(arr2, mock=True)
        res = self.executor.execute_tensor(arr2)

        self.assertEqual(sum(s[0] for s in size_res), arr.nbytes)
        self.assertTrue(np.array_equal(np.sort(np.concatenate(res)), np.sort(raw[raw < .5])))

        index2 = tensor(raw[:, :, 0, 0], chunk_size=3) < .5
        arr3 = arr[index2]
        res = self.executor.execute_tensor(arr3)

        self.assertEqual(sum(it.size for it in res), raw[raw[:, :, 0, 0] < .5].size)

    def testFancyIndexingExecution(self):
        raw = np.random.random((11, 8, 12, 14))
        arr = tensor(raw, chunk_size=(2, 3, 2, 3))

        index = [8, 10, 3, 1, 9, 10]
        arr2 = arr[index]

        res = self.executor.execute_tensor(arr2, concat=True)
        np.testing.assert_array_equal(res[0], raw[index])

        index = np.random.permutation(8)
        arr3 = arr[:2, ..., index]

        res = self.executor.execute_tensor(arr3, concat=True)
        self.assertTrue(np.array_equal(res[0], raw[:2, ..., index]))

        index = [1, 3, 9, 10]
        arr4 = arr[..., index, :5]

        res = self.executor.execute_tensor(arr4, concat=True)
        self.assertTrue(np.array_equal(res[0], raw[..., index, :5]))

        index1 = [8, 10, 3, 1, 9, 10]
        index2 = [1, 3, 9, 10, 2, 7]
        arr5 = arr[index1, :, index2]

        res = self.executor.execute_tensor(arr5, concat=True)
        np.testing.assert_array_equal(res[0], raw[index1, :, index2])

        index1 = [1, 3, 5, 7, 9, 10]
        index2 = [1, 9, 9, 10, 2, 7]
        arr6 = arr[index1, :, index2]

        res = self.executor.execute_tensor(arr6, concat=True)
        np.testing.assert_array_equal(res[0], raw[index1, :, index2])
        # fancy index is ordered, no concat required
        self.assertGreater(len(arr6.nsplits[0]), 1)

        index1 = [[8, 10, 3], [1, 9, 10]]
        index2 = [[1, 3, 9], [10, 2, 7]]
        arr7 = arr[index1, :, index2]

        res = self.executor.execute_tensor(arr7, concat=True)
        np.testing.assert_array_equal(res[0], raw[index1, :, index2])

        index1 = [[1, 3], [3, 7], [7, 7]]
        index2 = [1, 9]
        arr8 = arr[0, index1, :, index2]

        res = self.executor.execute_tensor(arr8, concat=True)
        np.testing.assert_array_equal(res[0], raw[0, index1, :, index2])

    def testSliceExecution(self):
        raw = np.random.random((11, 8, 12, 14))
        arr = tensor(raw, chunk_size=3)

        arr2 = arr[2:9:2, 3:7, -1:-9:-2, 12:-11:-4]
        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], raw[2:9:2, 3:7, -1:-9:-2, 12:-11:-4]))

        raw = sps.random(12, 14, density=.1)
        arr = tensor(raw, chunk_size=3)

        arr2 = arr[-1:-9:-2, 12:-11:-4]
        res = self.executor.execute_tensor(arr2, concat=True)[0]

        np.testing.assert_equal(res.toarray(), raw.toarray()[-1:-9:-2, 12:-11:-4])

    def testMixedIndexingExecution(self):
        raw = np.random.random((11, 8, 12, 13))
        arr = tensor(raw, chunk_size=3)

        raw_cond = raw[0, :, 0, 0] < .5
        cond = tensor(raw[0, :, 0, 0], chunk_size=3) < .5
        arr2 = arr[10::-2, cond, None, ..., :5]
        size_res = self.executor.execute_tensor(arr2, mock=True)
        res = self.executor.execute_tensor(arr2, concat=True)

        new_shape = list(arr2.shape)
        new_shape[1] = cond.shape[0]
        self.assertEqual(sum(s[0] for s in size_res), int(np.prod(new_shape) * arr2.dtype.itemsize))
        self.assertTrue(np.array_equal(res[0], raw[10::-2, raw_cond, None, ..., :5]))

        b_raw = np.random.random(8)
        cond = tensor(b_raw, chunk_size=2) < .5
        arr3 = arr[-2::-3, cond, ...]
        res = self.executor.execute_tensor(arr3, concat=True)

        self.assertTrue(np.array_equal(res[0], raw[-2::-3, b_raw < .5, ...]))

    def testSetItemExecution(self):
        raw = data = np.random.randint(0, 10, size=(11, 8, 12, 13))
        arr = tensor(raw.copy(), chunk_size=3)
        raw = raw.copy()

        idx = slice(2, 9, 2), slice(3, 7), slice(-1, -9, -2), 2
        arr[idx] = 20
        res = self.executor.execute_tensor(arr, concat=True)

        raw[idx] = 20
        self.assertTrue(np.array_equal(res[0], raw))

        raw = data
        shape = raw[idx].shape

        arr2 = tensor(raw.copy(), chunk_size=3)
        raw = raw.copy()

        replace = np.random.randint(10, 20, size=shape[:-1] + (1,)).astype('f4')
        arr2[idx] = tensor(replace, chunk_size=4)
        res = self.executor.execute_tensor(arr2, concat=True)

        raw[idx] = replace
        self.assertTrue(np.array_equal(res[0], raw))

    def testTakeExecution(self):
        data = np.random.rand(10, 20, 30)
        t = tensor(data, chunk_size=10)

        a = t.take([4, 1, 2, 6, 200])

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.take(data, [4, 1, 2, 6, 200])
        self.assertTrue(np.array_equal(res, expected))

        a = take(t, [5, 19, 2, 13], axis=1)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.take(data, [5, 19, 2, 13], axis=1)
        self.assertTrue(np.array_equal(res, expected))

    def testCompressExecution(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        a = tensor(data, chunk_size=1)

        t = compress([0, 1], a, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.compress([0, 1], data, axis=0)
        self.assertTrue(np.array_equal(res, expected))

        t = compress([0, 1], a, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.compress([0, 1], data, axis=1)
        self.assertTrue(np.array_equal(res, expected))

        t = a.compress([0, 1, 1])

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.compress([0, 1, 1], data)
        self.assertTrue(np.array_equal(res, expected))

        t = compress([False, True, True], a, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.compress([False, True, True], data, axis=0)
        self.assertTrue(np.array_equal(res, expected))

        t = compress([False, True], a, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.compress([False, True], data, axis=1)
        self.assertTrue(np.array_equal(res, expected))

        with self.assertRaises(np.AxisError):
            compress([0, 1, 1], a, axis=1)

    def testExtractExecution(self):
        data = np.arange(12).reshape((3, 4))
        a = tensor(data, chunk_size=2)
        condition = mod(a, 3) == 0

        t = extract(condition, a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.extract(np.mod(data, 3) == 0, data)
        self.assertTrue(np.array_equal(res, expected))

    def testChooseExecution(self):
        options.tensor.chunk_size = 2

        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        a = choose([2, 3, 1, 0], choices)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.choose([2, 3, 1, 0], choices)

        self.assertTrue(np.array_equal(res, expected))

        a = choose([2, 4, 1, 0], choices, mode='clip')  # 4 goes to 3 (4-1)
        expected = np.choose([2, 4, 1, 0], choices, mode='clip')

        res = self.executor.execute_tensor(a, concat=True)[0]
        self.assertTrue(np.array_equal(res, expected))

        a = choose([2, 4, 1, 0], choices, mode='wrap')  # 4 goes to (4 mod 4)
        expected = np.choose([2, 4, 1, 0], choices, mode='wrap')  # 4 goes to (4 mod 4)

        res = self.executor.execute_tensor(a, concat=True)[0]
        self.assertTrue(np.array_equal(res, expected))

        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]

        b = choose(a, choices)
        expected = np.choose(a, choices)

        res = self.executor.execute_tensor(b, concat=True)[0]
        self.assertTrue(np.array_equal(res, expected))

        a = np.array([0, 1]).reshape((2, 1, 1))
        c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
        c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))

        b = choose(a, (c1, c2))
        expected = np.choose(a, (c1, c2))

        res = self.executor.execute_tensor(b, concat=True)[0]
        self.assertTrue(np.array_equal(res, expected))

    def testUnravelExecution(self):
        a = tensor([22, 41, 37], chunk_size=1)
        t = stack(unravel_index(a, (7, 6)))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.stack(np.unravel_index([22, 41, 37], (7, 6)))

        self.assertTrue(np.array_equal(res, expected))

    def testNonzeroExecution(self):
        data = np.array([[1, 0, 0], [0, 2, 0], [1, 1, 0]])
        x = tensor(data, chunk_size=2)
        t = hstack(nonzero(x))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.hstack(np.nonzero(data))

        self.assertTrue(np.array_equal(res, expected))

    def testFlatnonzeroExecution(self):
        x = arange(-2, 3, chunk_size=2)

        t = flatnonzero(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.flatnonzero(np.arange(-2, 3))

        np.testing.assert_equal(res, expected)
