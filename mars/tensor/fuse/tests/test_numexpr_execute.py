#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import itertools
import numpy as np

from mars.executor import Executor
from mars.tensor.datasource import tensor


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()

    def testBaseExecution(self):
        executor_numpy = Executor('numpy')

        raw1 = np.random.randint(10, size=(10, 10, 10))
        raw2 = np.random.randint(10, size=(10, 10, 10))
        arr1 = tensor(raw1, chunk_size=3)
        arr2 = tensor(raw2, chunk_size=3)

        arr3 = arr1 + arr2 + 10
        arr4 = 10 + arr1 + arr2
        res3 = executor_numpy.execute_tensor(arr3, concat=True)
        res3_cmp = self.executor.execute_tensor(arr4, concat=True)
        self.assertTrue(np.array_equal(res3[0], res3_cmp[0]))

        res5 = executor_numpy.execute_tensor((arr1 + arr1), concat=True)
        res5_cmp = self.executor.execute_tensor((arr1 + arr1), concat=True)
        self.assertTrue(np.array_equal(res5[0], res5_cmp[0]))

    def testFuseSizeExecution(self):
        executor_size = Executor()
        executor_numpy = Executor()

        raw1 = np.random.randint(10, size=(10, 10, 10))
        arr1 = tensor(raw1, chunk_size=3)
        arr2 = arr1 + 10
        arr3 = arr2 * 3
        arr4 = arr3 + 5

        res4_size = executor_size.execute_tensor(arr4, mock=True)
        res4 = executor_numpy.execute_tensor(arr4, concat=True)
        res4_cmp = self.executor.execute_tensor(arr4, concat=True)
        self.assertEqual(sum(s[0] for s in res4_size), arr4.nbytes)
        self.assertTrue(np.array_equal(res4[0], res4_cmp[0]))

    def testUnaryExecution(self):
        from mars.tensor.arithmetic import UNARY_UFUNC, arccosh, invert, sin, conj

        _sp_unary_ufunc = {arccosh, invert, conj}
        _new_unary_ufunc = list(UNARY_UFUNC - _sp_unary_ufunc)
        executor_numexpr = Executor()

        def _normalize_by_sin(func1, func2, arr):
            return func1(abs(sin((func2(arr)))))

        for i, j in itertools.permutations(range(len(_new_unary_ufunc)), 2):
            raw = np.random.random((8, 8, 8))
            arr1 = tensor(raw, chunk_size=4)

            func1 = _new_unary_ufunc[i]
            func2 = _new_unary_ufunc[j]
            arr2 = _normalize_by_sin(func1, func2, arr1)
            res = executor_numexpr.execute_tensor(arr2, concat=True)
            res_cmp = self.executor.execute_tensor(arr2, concat=True)
            np.testing.assert_allclose(res[0], res_cmp[0])

        raw = np.random.randint(100, size=(8, 8, 8))
        arr1 = tensor(raw, chunk_size=4)
        arr2 = arccosh(1 + abs(invert(arr1)))
        res = executor_numexpr.execute_tensor(arr2, concat=True)
        res_cmp = self.executor.execute_tensor(arr2, concat=True)
        self.assertTrue(np.allclose(res[0], res_cmp[0]))

    def testBinExecution(self):
        from mars.tensor.arithmetic import BIN_UFUNC, mod, fmod, \
            bitand, bitor, bitxor, lshift, rshift, ldexp

        _sp_bin_ufunc = [mod, fmod, bitand, bitor, bitxor, lshift, rshift]
        _new_bin_ufunc = list(BIN_UFUNC - set(_sp_bin_ufunc) - {ldexp})
        executor_numexpr = Executor()

        for i, j in itertools.permutations(range(len(_new_bin_ufunc)), 2):
            raw = np.random.random((9, 9, 9))
            arr1 = tensor(raw, chunk_size=5)

            func1 = _new_bin_ufunc[i]
            func2 = _new_bin_ufunc[j]
            arr2 = func1(1, func2(2, arr1))
            res = executor_numexpr.execute_tensor(arr2, concat=True)
            res_cmp = self.executor.execute_tensor(arr2, concat=True)
            self.assertTrue(np.allclose(res[0], res_cmp[0]))

        for i, j in itertools.permutations(range(len(_sp_bin_ufunc)), 2):
            raw = np.random.randint(1, 100, size=(10, 10, 10))
            arr1 = tensor(raw, chunk_size=3)

            func1 = _sp_bin_ufunc[i]
            func2 = _sp_bin_ufunc[j]
            arr2 = func1(10, func2(arr1, 5))
            res = executor_numexpr.execute_tensor(arr2, concat=True)
            res_cmp = self.executor.execute_tensor(arr2, concat=True)
            self.assertTrue(np.allclose(res[0], res_cmp[0]))

    def testReductionExecution(self):
        raw1 = np.random.randint(5, size=(8, 8, 8))
        raw2 = np.random.randint(5, size=(8, 8, 8))
        arr1 = tensor(raw1, chunk_size=3)
        arr2 = tensor(raw2, chunk_size=3)

        res1 = self.executor.execute_tensor((arr1 + 1).sum(keepdims=True))
        res2 = self.executor.execute_tensor((arr1 + 1).prod(keepdims=True))
        self.assertTrue(np.array_equal((raw1 + 1).sum(keepdims=True), res1[0]))
        self.assertTrue(np.array_equal((raw1 + 1).prod(keepdims=True), res2[0]))

        res1 = self.executor.execute_tensor((arr1 + 1).sum(axis=1), concat=True)
        res2 = self.executor.execute_tensor((arr1 + 1).prod(axis=1), concat=True)
        res3 = self.executor.execute_tensor((arr1 + 1).max(axis=1), concat=True)
        res4 = self.executor.execute_tensor((arr1 + 1).min(axis=1), concat=True)
        self.assertTrue(np.array_equal((raw1 + 1).sum(axis=1), res1[0]))
        self.assertTrue(np.array_equal((raw1 + 1).prod(axis=1), res2[0]))
        self.assertTrue(np.array_equal((raw1 + 1).max(axis=1), res3[0]))
        self.assertTrue(np.array_equal((raw1 + 1).min(axis=1), res4[0]))

        raw3 = raw2 - raw1 + 10
        arr3 = -arr1 + arr2 + 10

        res1 = self.executor.execute_tensor(arr3.sum(axis=(0, 1)), concat=True)
        res2 = self.executor.execute_tensor(arr3.prod(axis=(0, 1)), concat=True)
        res3 = self.executor.execute_tensor(arr3.max(axis=(0, 1)), concat=True)
        res4 = self.executor.execute_tensor(arr3.min(axis=(0, 1)), concat=True)
        self.assertTrue(np.array_equal(raw3.sum(axis=(0, 1)), res1[0]))
        self.assertTrue(np.array_equal(raw3.prod(axis=(0, 1)), res2[0]))
        self.assertTrue(np.array_equal(raw3.max(axis=(0, 1)), res3[0]))
        self.assertTrue(np.array_equal(raw3.min(axis=(0, 1)), res4[0]))

    def testBoolReductionExecution(self):
        raw = np.random.randint(5, size=(8, 8, 8))
        arr = tensor(raw, chunk_size=2)

        res = self.executor.execute_tensor((arr > 3).sum(axis=1), concat=True)
        np.testing.assert_array_equal(res[0], (raw > 3).sum(axis=1))

        res = self.executor.execute_tensor((arr > 3).sum())
        np.testing.assert_array_equal(res, (raw > 3).sum())

    def testOrderExecution(self):
        raw = np.asfortranarray(np.random.rand(4, 5, 6))
        arr = tensor(raw, chunk_size=2)

        res = self.executor.execute_tensor(arr * 3 + 1, concat=True)[0]
        expected = raw * 3 + 1

        np.testing.assert_array_equal(res, expected)
        self.assertEqual(res.flags['C_CONTIGUOUS'], expected.flags['C_CONTIGUOUS'])
        self.assertEqual(res.flags['F_CONTIGUOUS'], expected.flags['F_CONTIGUOUS'])
