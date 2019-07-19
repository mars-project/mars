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
from mars.tensor.datasource import ones, tensor
from mars.tensor.reduction import mean, nansum, nanmax, nanmin, nanmean, nanprod, nanargmax, \
    nanargmin, nanvar, nanstd, count_nonzero, allclose, array_equal, var, std, nancumsum, nancumprod


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testSumProdExecution(self):
        arr = ones((10, 8), chunk_size=3)
        self.assertEqual([80], self.executor.execute_tensor(arr.sum()))
        self.assertEqual((10,) * 8,
                         tuple(np.concatenate(self.executor.execute_tensor(arr.sum(axis=0)))))

        arr = ones((3, 3), chunk_size=2)
        self.assertEqual([512], self.executor.execute_tensor((arr * 2).prod()))
        self.assertEqual((8,) * 3,
                         tuple(np.concatenate(self.executor.execute_tensor((arr * 2).prod(axis=0)))))

        raw = sps.random(10, 20, density=.1)
        arr = tensor(raw, chunk_size=3)
        res = self.executor.execute_tensor(arr.sum())[0]

        self.assertAlmostEqual(res, raw.sum())

    def testMaxMinExecution(self):
        raw = np.random.randint(10000, size=(10, 10, 10))

        arr = tensor(raw, chunk_size=3)

        self.assertEqual([raw.max()], self.executor.execute_tensor(arr.max()))
        self.assertEqual([raw.min()], self.executor.execute_tensor(arr.min()))

        np.testing.assert_array_equal(
            raw.max(axis=0), self.executor.execute_tensor(arr.max(axis=0), concat=True)[0])
        np.testing.assert_array_equal(
            raw.min(axis=0), self.executor.execute_tensor(arr.min(axis=0), concat=True)[0])

        np.testing.assert_array_equal(
            raw.max(axis=(1, 2)), self.executor.execute_tensor(arr.max(axis=(1, 2)), concat=True)[0])
        np.testing.assert_array_equal(
            raw.min(axis=(1, 2)), self.executor.execute_tensor(arr.min(axis=(1, 2)), concat=True)[0])

        raw = sps.random(10, 10, density=.5)

        arr = tensor(raw, chunk_size=3)

        self.assertEqual([raw.max()], self.executor.execute_tensor(arr.max()))
        self.assertEqual([raw.min()], self.executor.execute_tensor(arr.min()))

    def testAllAnyExecution(self):
        raw1 = np.zeros((10, 15))
        raw2 = np.ones((10, 15))
        raw3 = np.array([[True, False, True, False], [True, True, True, True],
                         [False, False, False, False], [False, True, False, True]])

        arr1 = tensor(raw1, chunk_size=3)
        arr2 = tensor(raw2, chunk_size=3)
        arr3 = tensor(raw3, chunk_size=4)

        self.assertFalse(self.executor.execute_tensor(arr1.all())[0])
        self.assertTrue(self.executor.execute_tensor(arr2.all())[0])
        self.assertFalse(self.executor.execute_tensor(arr1.any())[0])
        self.assertTrue(self.executor.execute_tensor(arr1.any()))
        np.testing.assert_array_equal(raw3.all(axis=1),
                                      self.executor.execute_tensor(arr3.all(axis=1))[0])
        np.testing.assert_array_equal(raw3.any(axis=0),
                                      self.executor.execute_tensor(arr3.any(axis=0))[0])

        raw = sps.random(10, 10, density=.5) > .5

        arr = tensor(raw, chunk_size=3)

        self.assertEqual(raw.A.all(), self.executor.execute_tensor(arr.all())[0])
        self.assertEqual(raw.A.any(), self.executor.execute_tensor(arr.any())[0])

    def testMeanExecution(self):
        raw1 = np.random.random((20, 25))
        raw2 = np.random.randint(10, size=(20, 25))

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.mean())
        expected1 = raw1.mean()
        self.assertTrue(np.allclose(res1[0], expected1))

        res2 = self.executor.execute_tensor(arr1.mean(axis=0))
        expected2 = raw1.mean(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr1.mean(axis=1, keepdims=True))
        expected3 = raw1.mean(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        arr2 = tensor(raw2, chunk_size=3)

        res1 = self.executor.execute_tensor(arr2.mean())
        expected1 = raw2.mean()
        self.assertEqual(res1[0], expected1)

        res2 = self.executor.execute_tensor(arr2.mean(axis=0))
        expected2 = raw2.mean(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr2.mean(axis=1, keepdims=True))
        expected3 = raw2.mean(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        raw1 = sps.random(20, 25, density=.1)

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.mean())
        expected1 = raw1.mean()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr2 = tensor(raw1, chunk_size=30)

        res1 = self.executor.execute_tensor(arr2.mean())
        expected1 = raw1.mean()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr = mean(1)
        self.assertEqual(self.executor.execute_tensor(arr)[0], 1)

    def testVarExecution(self):
        raw1 = np.random.random((20, 25))
        raw2 = np.random.randint(10, size=(20, 25))

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.var())
        expected1 = raw1.var()
        self.assertTrue(np.allclose(res1[0], expected1))

        res2 = self.executor.execute_tensor(arr1.var(axis=0))
        expected2 = raw1.var(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr1.var(axis=1, keepdims=True))
        expected3 = raw1.var(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        arr2 = tensor(raw2, chunk_size=3)

        res1 = self.executor.execute_tensor(arr2.var())
        expected1 = raw2.var()
        self.assertAlmostEqual(res1[0], expected1)

        res2 = self.executor.execute_tensor(arr2.var(axis=0))
        expected2 = raw2.var(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr2.var(axis=1, keepdims=True))
        expected3 = raw2.var(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        res4 = self.executor.execute_tensor(arr2.var(ddof=1))
        expected4 = raw2.var(ddof=1)
        self.assertAlmostEqual(res4[0], expected4)

        raw1 = sps.random(20, 25, density=.1)

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.var())
        expected1 = raw1.toarray().var()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr2 = tensor(raw1, chunk_size=30)

        res1 = self.executor.execute_tensor(arr2.var())
        expected1 = raw1.toarray().var()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr = var(1)
        self.assertEqual(self.executor.execute_tensor(arr)[0], 0)

    def testStdExecution(self):
        raw1 = np.random.random((20, 25))
        raw2 = np.random.randint(10, size=(20, 25))

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.std())
        expected1 = raw1.std()
        self.assertTrue(np.allclose(res1[0], expected1))

        res2 = self.executor.execute_tensor(arr1.std(axis=0))
        expected2 = raw1.std(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr1.std(axis=1, keepdims=True))
        expected3 = raw1.std(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        arr2 = tensor(raw2, chunk_size=3)

        res1 = self.executor.execute_tensor(arr2.std())
        expected1 = raw2.std()
        self.assertAlmostEqual(res1[0], expected1)

        res2 = self.executor.execute_tensor(arr2.std(axis=0))
        expected2 = raw2.std(axis=0)
        self.assertTrue(np.allclose(np.concatenate(res2), expected2))

        res3 = self.executor.execute_tensor(arr2.std(axis=1, keepdims=True))
        expected3 = raw2.std(axis=1, keepdims=True)
        self.assertTrue(np.allclose(np.concatenate(res3), expected3))

        res4 = self.executor.execute_tensor(arr2.std(ddof=1))
        expected4 = raw2.std(ddof=1)
        self.assertAlmostEqual(res4[0], expected4)

        raw1 = sps.random(20, 25, density=.1)

        arr1 = tensor(raw1, chunk_size=3)

        res1 = self.executor.execute_tensor(arr1.std())
        expected1 = raw1.toarray().std()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr2 = tensor(raw1, chunk_size=30)

        res1 = self.executor.execute_tensor(arr2.std())
        expected1 = raw1.toarray().std()
        self.assertTrue(np.allclose(res1[0], expected1))

        arr = std(1)
        self.assertEqual(self.executor.execute_tensor(arr)[0], 0)

    def testArgReduction(self):
        raw = np.random.random((20, 20, 20))

        arr = tensor(raw, chunk_size=3)

        self.assertEqual(raw.argmax(),
                         self.executor.execute_tensor(arr.argmax())[0])
        self.assertEqual(raw.argmin(),
                         self.executor.execute_tensor(arr.argmin())[0])

        np.testing.assert_array_equal(
            raw.argmax(axis=0), self.executor.execute_tensor(arr.argmax(axis=0), concat=True)[0])
        np.testing.assert_array_equal(
            raw.argmin(axis=0), self.executor.execute_tensor(arr.argmin(axis=0), concat=True)[0])

        raw_format = sps.random(20, 20, density=.1, format='lil')

        random_min = np.random.randint(0, 200)
        random_max = np.random.randint(200, 400)
        raw_format[np.unravel_index(random_min, raw_format.shape)] = -1
        raw_format[np.unravel_index(random_max, raw_format.shape)] = 2

        raw = raw_format.tocoo()
        arr = tensor(raw, chunk_size=3)

        self.assertEqual(raw.argmax(),
                         self.executor.execute_tensor(arr.argmax())[0])
        self.assertEqual(raw.argmin(),
                         self.executor.execute_tensor(arr.argmin())[0])

    def testNanReduction(self):
        raw = np.random.choice(a=[0, 1, np.nan], size=(10, 10), p=[0.3, 0.4, 0.3])

        arr = tensor(raw, chunk_size=3)

        self.assertEqual(np.nansum(raw), self.executor.execute_tensor(nansum(arr))[0])
        self.assertEqual(np.nanprod(raw), self.executor.execute_tensor(nanprod(arr))[0])
        self.assertEqual(np.nanmax(raw), self.executor.execute_tensor(nanmax(arr))[0])
        self.assertEqual(np.nanmin(raw), self.executor.execute_tensor(nanmin(arr))[0])
        self.assertEqual(np.nanmean(raw), self.executor.execute_tensor(nanmean(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw), self.executor.execute_tensor(nanvar(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw, ddof=1), self.executor.execute_tensor(nanvar(arr, ddof=1))[0])
        self.assertAlmostEqual(np.nanstd(raw), self.executor.execute_tensor(nanstd(arr))[0])
        self.assertAlmostEqual(np.nanstd(raw, ddof=1), self.executor.execute_tensor(nanstd(arr, ddof=1))[0])

        arr = tensor(raw, chunk_size=10)

        self.assertEqual(np.nansum(raw), self.executor.execute_tensor(nansum(arr))[0])
        self.assertEqual(np.nanprod(raw), self.executor.execute_tensor(nanprod(arr))[0])
        self.assertEqual(np.nanmax(raw), self.executor.execute_tensor(nanmax(arr))[0])
        self.assertEqual(np.nanmin(raw), self.executor.execute_tensor(nanmin(arr))[0])
        self.assertEqual(np.nanmean(raw), self.executor.execute_tensor(nanmean(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw), self.executor.execute_tensor(nanvar(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw, ddof=1), self.executor.execute_tensor(nanvar(arr, ddof=1))[0])
        self.assertAlmostEqual(np.nanstd(raw), self.executor.execute_tensor(nanstd(arr))[0])
        self.assertAlmostEqual(np.nanstd(raw, ddof=1), self.executor.execute_tensor(nanstd(arr, ddof=1))[0])

        raw = np.random.random((10, 10))
        raw[:3, :3] = np.nan
        arr = tensor(raw, chunk_size=3)
        self.assertEqual(np.nanargmin(raw), self.executor.execute_tensor(nanargmin(arr))[0])
        self.assertEqual(np.nanargmax(raw), self.executor.execute_tensor(nanargmax(arr))[0])

        raw = np.full((10, 10), np.nan)
        arr = tensor(raw, chunk_size=3)

        self.assertEqual(0, self.executor.execute_tensor(nansum(arr))[0])
        self.assertEqual(1, self.executor.execute_tensor(nanprod(arr))[0])
        self.assertTrue(np.isnan(self.executor.execute_tensor(nanmax(arr))[0]))
        self.assertTrue(np.isnan(self.executor.execute_tensor(nanmin(arr))[0]))
        self.assertTrue(np.isnan(self.executor.execute_tensor(nanmean(arr))[0]))
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tensor(nanargmin(arr))[0]
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tensor(nanargmax(arr))[0]

        raw = sps.random(10, 10, density=.1, format='csr')
        raw[:3, :3] = np.nan
        arr = tensor(raw, chunk_size=3)

        self.assertAlmostEqual(np.nansum(raw.A), self.executor.execute_tensor(nansum(arr))[0])
        self.assertAlmostEqual(np.nanprod(raw.A), self.executor.execute_tensor(nanprod(arr))[0])
        self.assertAlmostEqual(np.nanmax(raw.A), self.executor.execute_tensor(nanmax(arr))[0])
        self.assertAlmostEqual(np.nanmin(raw.A), self.executor.execute_tensor(nanmin(arr))[0])
        self.assertAlmostEqual(np.nanmean(raw.A), self.executor.execute_tensor(nanmean(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw.A), self.executor.execute_tensor(nanvar(arr))[0])
        self.assertAlmostEqual(np.nanvar(raw.A, ddof=1), self.executor.execute_tensor(nanvar(arr, ddof=1))[0])
        self.assertAlmostEqual(np.nanstd(raw.A), self.executor.execute_tensor(nanstd(arr))[0])
        self.assertAlmostEqual(np.nanstd(raw.A, ddof=1), self.executor.execute_tensor(nanstd(arr, ddof=1))[0])

        arr = nansum(1)
        self.assertEqual(self.executor.execute_tensor(arr)[0], 1)

    def testCumReduction(self):
        raw = np.random.randint(5, size=(8, 8, 8))

        arr = tensor(raw, chunk_size=3)

        res1 = self.executor.execute_tensor(arr.cumsum(axis=1), concat=True)
        res2 = self.executor.execute_tensor(arr.cumprod(axis=1), concat=True)
        expected1 = raw.cumsum(axis=1)
        expected2 = raw.cumprod(axis=1)
        np.testing.assert_array_equal(res1[0], expected1)
        np.testing.assert_array_equal(res2[0], expected2)

        raw = sps.random(8, 8, density=.1)

        arr = tensor(raw, chunk_size=3)

        res1 = self.executor.execute_tensor(arr.cumsum(axis=1), concat=True)
        res2 = self.executor.execute_tensor(arr.cumprod(axis=1), concat=True)
        expected1 = raw.A.cumsum(axis=1)
        expected2 = raw.A.cumprod(axis=1)
        self.assertTrue(np.allclose(res1[0], expected1))
        self.assertTrue(np.allclose(res2[0], expected2))

    def testNanCumReduction(self):
        raw = np.random.randint(5, size=(8, 8, 8))
        raw[:2, 2:4, 4:6] = np.nan

        arr = tensor(raw, chunk_size=3)

        res1 = self.executor.execute_tensor(nancumsum(arr, axis=1), concat=True)
        res2 = self.executor.execute_tensor(nancumprod(arr, axis=1), concat=True)
        expected1 = np.nancumsum(raw, axis=1)
        expected2 = np.nancumprod(raw, axis=1)
        np.testing.assert_array_equal(res1[0], expected1)
        np.testing.assert_array_equal(res2[0], expected2)

        raw = sps.random(8, 8, density=.1, format='lil')
        raw[:2, 2:4] = np.nan

        arr = tensor(raw, chunk_size=3)

        res1 = self.executor.execute_tensor(nancumsum(arr, axis=1), concat=True)[0]
        res2 = self.executor.execute_tensor(nancumprod(arr, axis=1), concat=True)[0]
        expected1 = np.nancumsum(raw.A, axis=1)
        expected2 = np.nancumprod(raw.A, axis=1)
        self.assertTrue(np.allclose(res1, expected1))
        self.assertTrue(np.allclose(res2, expected2))

    def testOutReductionExecution(self):
        raw = np.random.randint(5, size=(8, 8, 8))

        arr = tensor(raw, chunk_size=3)
        arr2 = ones((8, 8), dtype='i8', chunk_size=3)
        arr.sum(axis=1, out=arr2)

        res = self.executor.execute_tensor(arr2, concat=True)[0]
        expected = raw.sum(axis=1)

        np.testing.assert_array_equal(res, expected)

    def testOutCumReductionExecution(self):
        raw = np.random.randint(5, size=(8, 8, 8))

        arr = tensor(raw, chunk_size=3)
        arr.cumsum(axis=0, out=arr)

        res = self.executor.execute_tensor(arr, concat=True)[0]
        expected = raw.cumsum(axis=0)

        np.testing.assert_array_equal(res, expected)

    def testCountNonzeroExecution(self):
        raw = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]

        arr = tensor(raw, chunk_size=2)
        t = count_nonzero(arr)

        res = self.executor.execute_tensor(t)[0]
        expected = np.count_nonzero(raw)
        np.testing.assert_equal(res, expected)

        t = count_nonzero(arr, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.count_nonzero(raw, axis=0)
        np.testing.assert_equal(res, expected)

        t = count_nonzero(arr, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.count_nonzero(raw, axis=1)
        np.testing.assert_equal(res, expected)

        raw = sps.csr_matrix(raw)

        arr = tensor(raw, chunk_size=2)
        t = count_nonzero(arr)

        res = self.executor.execute_tensor(t)[0]
        expected = np.count_nonzero(raw.A)
        np.testing.assert_equal(res, expected)

        t = count_nonzero(arr, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.count_nonzero(raw.A, axis=0)
        np.testing.assert_equal(res, expected)

        t = count_nonzero(arr, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.count_nonzero(raw.A, axis=1)
        np.testing.assert_equal(res, expected)

    def testAllcloseExecution(self):
        a = tensor([1e10, 1e-7], chunk_size=1)
        b = tensor([1.00001e10, 1e-8], chunk_size=1)

        t = allclose(a, b)

        res = self.executor.execute_tensor(t)[0]
        self.assertFalse(res)

        a = tensor([1e10, 1e-8], chunk_size=1)
        b = tensor([1.00001e10, 1e-9], chunk_size=1)

        t = allclose(a, b)

        res = self.executor.execute_tensor(t)[0]
        self.assertTrue(res)

        a = tensor([1.0, np.nan], chunk_size=1)
        b = tensor([1.0, np.nan], chunk_size=1)

        t = allclose(a, b, equal_nan=True)

        res = self.executor.execute_tensor(t)[0]
        self.assertTrue(res)

        a = tensor(sps.csr_matrix([[1e10, 1e-7], [0, 0]]), chunk_size=1)
        b = tensor(sps.csr_matrix([[1.00001e10, 1e-8], [0, 0]]), chunk_size=1)

        t = allclose(a, b)

        res = self.executor.execute_tensor(t)[0]
        self.assertFalse(res)

    def testArrayEqual(self):
        a = ones((10, 5), chunk_size=1)
        b = ones((10, 5), chunk_size=2)

        c = array_equal(a, b)

        res = bool(self.executor.execute_tensor(c)[0])
        self.assertTrue(res)
