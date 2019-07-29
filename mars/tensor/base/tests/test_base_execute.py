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

from mars import tensor as mt
from mars.executor import Executor
from mars.tensor.datasource import tensor, ones, zeros, arange
from mars.tensor.base import copyto, transpose, moveaxis, broadcast_to, broadcast_arrays, where, \
    expand_dims, rollaxis, atleast_1d, atleast_2d, atleast_3d, argwhere, array_split, split, \
    hsplit, vsplit, dsplit, roll, squeeze, ptp, diff, ediff1d, digitize, average, cov, corrcoef, \
    flip, flipud, fliplr, repeat, tile, isin
from mars.tensor.merge import stack
from mars.tensor.reduction import all as tall


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testRechunkExecution(self):
        raw = np.random.random((11, 8))
        arr = tensor(raw, chunk_size=3)
        arr2 = arr.rechunk(4)

        res = self.executor.execute_tensor(arr2)

        self.assertTrue(np.array_equal(res[0], raw[:4, :4]))
        self.assertTrue(np.array_equal(res[1], raw[:4, 4:]))
        self.assertTrue(np.array_equal(res[2], raw[4:8, :4]))
        self.assertTrue(np.array_equal(res[3], raw[4:8, 4:]))
        self.assertTrue(np.array_equal(res[4], raw[8:, :4]))
        self.assertTrue(np.array_equal(res[5], raw[8:, 4:]))

    def testCopytoExecution(self):
        a = ones((2, 3), chunk_size=1)
        b = tensor([3, -1, 3], chunk_size=2)

        copyto(a, b, where=b > 1)

        res = self.executor.execute_tensor(a, concat=True)[0]
        expected = np.array([[3, 1, 3], [3, 1, 3]])

        np.testing.assert_equal(res, expected)

    def testAstypeExecution(self):
        raw = np.random.random((10, 5))
        arr = tensor(raw, chunk_size=3)
        arr2 = arr.astype('i8')

        res = self.executor.execute_tensor(arr2, concat=True)
        self.assertTrue(np.array_equal(res[0], raw.astype('i8')))

        raw = sps.random(10, 5, density=.2)
        arr = tensor(raw, chunk_size=3)
        arr2 = arr.astype('i8')

        res = self.executor.execute_tensor(arr2, concat=True)
        self.assertTrue(np.array_equal(res[0].toarray(), raw.astype('i8').toarray()))

    def testTransposeExecution(self):
        raw = np.random.random((11, 8, 5))
        arr = tensor(raw, chunk_size=3)
        arr2 = transpose(arr)

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], raw.T))

        arr3 = transpose(arr, axes=(-2, -1, -3))

        res = self.executor.execute_tensor(arr3, concat=True)

        self.assertTrue(np.array_equal(res[0], raw.transpose(1, 2, 0)))

        raw = sps.random(11, 8)
        arr = tensor(raw, chunk_size=3)
        arr2 = transpose(arr)

        self.assertTrue(arr2.issparse())

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0].toarray(), raw.T.toarray()))

    def testSwapaxesExecution(self):
        raw = np.random.random((11, 8, 5))
        arr = tensor(raw, chunk_size=3)
        arr2 = arr.swapaxes(2, 0)

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], raw.swapaxes(2, 0)))

        raw = sps.random(11, 8, density=.2)
        arr = tensor(raw, chunk_size=3)
        arr2 = arr.swapaxes(1, 0)

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0].toarray(), raw.toarray().swapaxes(1, 0)))

    def testMoveaxisExecution(self):
        x = zeros((3, 4, 5), chunk_size=2)

        t = moveaxis(x, 0, -1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertEqual(res.shape, (4, 5, 3))

        t = moveaxis(x, -1, 0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertEqual(res.shape, (5, 3, 4))

        t = moveaxis(x, [0, 1], [-1, -2])

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertEqual(res.shape, (5, 4, 3))

        t = moveaxis(x, [0, 1, 2], [-1, -2, -3])

        res = self.executor.execute_tensor(t, concat=True)[0]
        self.assertEqual(res.shape, (5, 4, 3))

    def testBroadcastToExecution(self):
        raw = np.random.random((10, 5, 1))
        arr = tensor(raw, chunk_size=2)
        arr2 = broadcast_to(arr, (5, 10, 5, 6))

        res = self.executor.execute_tensor(arr2, concat=True)

        self.assertTrue(np.array_equal(res[0], np.broadcast_to(raw, (5, 10, 5, 6))))

    def testBroadcastArraysExecutions(self):
        x_data = [[1, 2, 3]]
        x = tensor(x_data, chunk_size=1)
        y_data = [[1], [2], [3]]
        y = tensor(y_data, chunk_size=2)

        a = broadcast_arrays(x, y)

        res = [self.executor.execute_tensor(arr, concat=True)[0] for arr in a]
        expected = np.broadcast_arrays(x_data, y_data)

        for r, e in zip(res, expected):
            np.testing.assert_equal(r, e)

    def testWhereExecution(self):
        raw_cond = np.random.randint(0, 2, size=(4, 4), dtype='?')
        raw_x = np.random.rand(4, 1)
        raw_y = np.random.rand(4, 4)

        cond, x, y = tensor(raw_cond, chunk_size=2), tensor(raw_x, chunk_size=2), tensor(raw_y, chunk_size=2)

        arr = where(cond, x, y)
        res = self.executor.execute_tensor(arr, concat=True)
        self.assertTrue(np.array_equal(res[0], np.where(raw_cond, raw_x, raw_y)))

        raw_cond = sps.csr_matrix(np.random.randint(0, 2, size=(4, 4), dtype='?'))
        raw_x = sps.random(4, 1, density=.1)
        raw_y = sps.random(4, 4, density=.1)

        cond, x, y = tensor(raw_cond, chunk_size=2), tensor(raw_x, chunk_size=2), tensor(raw_y, chunk_size=2)

        arr = where(cond, x, y)
        res = self.executor.execute_tensor(arr, concat=True)[0]
        self.assertTrue(np.array_equal(res.toarray(),
                                       np.where(raw_cond.toarray(), raw_x.toarray(), raw_y.toarray())))

    def testReshapeExecution(self):
        raw_data = np.random.rand(10, 20, 30)
        x = tensor(raw_data, chunk_size=6)

        y = x.reshape(-1, 30)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(-1, 30)))

        y2 = x.reshape(10, -1)

        res = self.executor.execute_tensor(y2, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(10, -1)))

        y3 = x.reshape(-1)

        res = self.executor.execute_tensor(y3, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(-1)))

        y4 = x.ravel()

        res = self.executor.execute_tensor(y4, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.ravel()))

        raw_data = np.random.rand(30, 100, 20)
        x = tensor(raw_data, chunk_size=6)

        y = x.reshape(-1, 20, 5, 5, 4)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(-1, 20, 5, 5, 4)))

        y2 = x.reshape(3000, 10, 2)

        res = self.executor.execute_tensor(y2, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(3000, 10, 2)))

        y3 = x.reshape(60, 25, 40)

        res = self.executor.execute_tensor(y3, concat=True)
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(60, 25, 40)))

        y4 = x.reshape(60, 25, 40)
        y4.op.extra_params['_reshape_with_shuffle'] = True

        size_res = self.executor.execute_tensor(y4, mock=True)
        res = self.executor.execute_tensor(y4, concat=True)
        self.assertEqual(res[0].nbytes, sum(v[0] for v in size_res))
        self.assertTrue(np.array_equal(res[0], raw_data.reshape(60, 25, 40)))

    def testExpandDimsExecution(self):
        raw_data = np.random.rand(10, 20, 30)
        x = tensor(raw_data, chunk_size=6)

        y = expand_dims(x, 1)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.expand_dims(raw_data, 1)))

        y = expand_dims(x, 0)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.expand_dims(raw_data, 0)))

        y = expand_dims(x, 3)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.expand_dims(raw_data, 3)))

        y = expand_dims(x, -1)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.expand_dims(raw_data, -1)))

        y = expand_dims(x, -4)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.expand_dims(raw_data, -4)))

        with self.assertRaises(np.AxisError):
            expand_dims(x, -5)

        with self.assertRaises(np.AxisError):
            expand_dims(x, 4)

    def testRollAxisExecution(self):
        x = ones((3, 4, 5, 6), chunk_size=1)
        y = rollaxis(x, 3, 1)

        res = self.executor.execute_tensor(y, concat=True)
        self.assertTrue(np.array_equal(res[0], np.rollaxis(np.ones((3, 4, 5, 6)), 3, 1)))

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

    def testArgwhereExecution(self):
        x = arange(6, chunk_size=2).reshape(2, 3)
        t = argwhere(x > 1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.argwhere(np.arange(6).reshape(2, 3) > 1)

        self.assertTrue(np.array_equal(res, expected))

    def testArraySplitExecution(self):
        x = arange(48, chunk_size=3).reshape(2, 3, 8)
        ss = array_split(x, 3, axis=2)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.array_split(np.arange(48).reshape(2, 3, 8), 3, axis=2)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        ss = array_split(x, [3, 5, 6, 10], axis=2)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.array_split(np.arange(48).reshape(2, 3, 8), [3, 5, 6, 10], axis=2)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    def testSplitExecution(self):
        x = arange(48, chunk_size=3).reshape(2, 3, 8)
        ss = split(x, 4, axis=2)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.split(np.arange(48).reshape(2, 3, 8), 4, axis=2)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        ss = split(x, [3, 5, 6, 10], axis=2)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.split(np.arange(48).reshape(2, 3, 8), [3, 5, 6, 10], axis=2)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        # hsplit
        x = arange(120, chunk_size=3).reshape(2, 12, 5)
        ss = hsplit(x, 4)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.hsplit(np.arange(120).reshape(2, 12, 5), 4)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        # vsplit
        x = arange(48, chunk_size=3).reshape(8, 3, 2)
        ss = vsplit(x, 4)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.vsplit(np.arange(48).reshape(8, 3, 2), 4)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        # dsplit
        x = arange(48, chunk_size=3).reshape(2, 3, 8)
        ss = dsplit(x, 4)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.dsplit(np.arange(48).reshape(2, 3, 8), 4)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

        x_data = sps.random(12, 8, density=.1)
        x = tensor(x_data, chunk_size=3)
        ss = split(x, 4, axis=0)

        res = [self.executor.execute_tensor(i, concat=True)[0] for i in ss]
        expected = np.split(x_data.toarray(), 4, axis=0)
        self.assertEqual(len(res), len(expected))
        [np.testing.assert_equal(r.toarray(), e) for r, e in zip(res, expected)]

    def testRollExecution(self):
        x = arange(10, chunk_size=2)

        t = roll(x, 2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.roll(np.arange(10), 2)
        np.testing.assert_equal(res, expected)

        x2 = x.reshape(2, 5)

        t = roll(x2, 1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.roll(np.arange(10).reshape(2, 5), 1)
        np.testing.assert_equal(res, expected)

        t = roll(x2, 1, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.roll(np.arange(10).reshape(2, 5), 1, axis=0)
        np.testing.assert_equal(res, expected)

        t = roll(x2, 1, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.roll(np.arange(10).reshape(2, 5), 1, axis=1)
        np.testing.assert_equal(res, expected)

    def testSqueezeExecution(self):
        data = np.array([[[0], [1], [2]]])
        x = tensor(data, chunk_size=1)

        t = squeeze(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.squeeze(data)
        np.testing.assert_equal(res, expected)

        t = squeeze(x, axis=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.squeeze(data, axis=2)
        np.testing.assert_equal(res, expected)

    def testPtpExecution(self):
        x = arange(4, chunk_size=1).reshape(2, 2)

        t = ptp(x, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.ptp(np.arange(4).reshape(2, 2), axis=0)
        np.testing.assert_equal(res, expected)

        t = ptp(x, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.ptp(np.arange(4).reshape(2, 2), axis=1)
        np.testing.assert_equal(res, expected)

        t = ptp(x)

        res = self.executor.execute_tensor(t)[0]
        expected = np.ptp(np.arange(4).reshape(2, 2))
        np.testing.assert_equal(res, expected)

    def testDiffExecution(self):
        data = np.array([1, 2, 4, 7, 0])
        x = tensor(data, chunk_size=2)

        t = diff(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.diff(data)
        np.testing.assert_equal(res, expected)

        t = diff(x, n=2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.diff(data, n=2)
        np.testing.assert_equal(res, expected)

        data = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
        x = tensor(data, chunk_size=2)

        t = diff(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.diff(data)
        np.testing.assert_equal(res, expected)

        t = diff(x, axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.diff(data, axis=0)
        np.testing.assert_equal(res, expected)

        x = mt.arange('1066-10-13', '1066-10-16', dtype=mt.datetime64)
        t = diff(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.diff(np.arange('1066-10-13', '1066-10-16', dtype=np.datetime64))
        np.testing.assert_equal(res, expected)

    def testEdiff1d(self):
        data = np.array([1, 2, 4, 7, 0])
        x = tensor(data, chunk_size=2)

        t = ediff1d(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.ediff1d(data)
        np.testing.assert_equal(res, expected)

        to_begin = tensor(-99, chunk_size=2)
        to_end = tensor([88, 99], chunk_size=2)
        t = ediff1d(x, to_begin=to_begin, to_end=to_end)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.ediff1d(data, to_begin=-99, to_end=np.array([88, 99]))
        np.testing.assert_equal(res, expected)

        data = [[1, 2, 4], [1, 6, 24]]

        t = ediff1d(tensor(data, chunk_size=2))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.ediff1d(data)
        np.testing.assert_equal(res, expected)

    def testDigitizeExecution(self):
        data = np.array([0.2, 6.4, 3.0, 1.6])
        x = tensor(data, chunk_size=2)
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        inds = digitize(x, bins)

        res = self.executor.execute_tensor(inds, concat=True)[0]
        expected = np.digitize(data, bins)
        np.testing.assert_equal(res, expected)

        b = tensor(bins, chunk_size=2)
        inds = digitize(x, b)

        res = self.executor.execute_tensor(inds, concat=True)[0]
        expected = np.digitize(data, bins)
        np.testing.assert_equal(res, expected)

        data = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        x = tensor(data, chunk_size=2)
        bins = np.array([0, 5, 10, 15, 20])
        inds = digitize(x, bins, right=True)

        res = self.executor.execute_tensor(inds, concat=True)[0]
        expected = np.digitize(data, bins, right=True)
        np.testing.assert_equal(res, expected)

        inds = digitize(x, bins, right=False)

        res = self.executor.execute_tensor(inds, concat=True)[0]
        expected = np.digitize(data, bins, right=False)
        np.testing.assert_equal(res, expected)

        data = sps.random(10, 1, density=.1) * 12
        x = tensor(data, chunk_size=2)
        bins = np.array([1.0, 2.0, 2.5, 4.0, 10.0])
        inds = digitize(x, bins)

        res = self.executor.execute_tensor(inds, concat=True)[0]
        expected = np.digitize(data.toarray(), bins, right=False)
        np.testing.assert_equal(res.toarray(), expected)

    def testAverageExecution(self):
        data = arange(1, 5, chunk_size=1)
        t = average(data)

        res = self.executor.execute_tensor(t)[0]
        expected = np.average(np.arange(1, 5))
        self.assertEqual(res, expected)

        t = average(arange(1, 11, chunk_size=2), weights=arange(10, 0, -1, chunk_size=2))

        res = self.executor.execute_tensor(t)[0]
        expected = np.average(range(1, 11), weights=range(10, 0, -1))
        self.assertEqual(res, expected)

        data = arange(6, chunk_size=2).reshape((3, 2))
        t = average(data, axis=1, weights=tensor([1./4, 3./4], chunk_size=2))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.average(np.arange(6).reshape(3, 2), axis=1, weights=(1./4, 3./4))
        np.testing.assert_equal(res, expected)

        with self.assertRaises(TypeError):
            average(data, weights=tensor([1./4, 3./4], chunk_size=2))

    def testCovExecution(self):
        data = np.array([[0, 2], [1, 1], [2, 0]]).T
        x = tensor(data, chunk_size=1)

        t = cov(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.cov(data)
        np.testing.assert_equal(res, expected)

        data_x = [-2.1, -1, 4.3]
        data_y = [3,  1.1,  0.12]
        x = tensor(data_x, chunk_size=1)
        y = tensor(data_y, chunk_size=1)

        X = stack((x, y), axis=0)
        t = cov(x, y)
        r = tall(t == cov(X))
        self.assertTrue(self.executor.execute_tensor(r)[0])

    def testCorrcoefExecution(self):
        data_x = [-2.1, -1, 4.3]
        data_y = [3, 1.1, 0.12]
        x = tensor(data_x, chunk_size=1)
        y = tensor(data_y, chunk_size=1)

        t = corrcoef(x, y)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.corrcoef(data_x, data_y)
        np.testing.assert_equal(res, expected)

    def testFlipExecution(self):
        a = arange(8, chunk_size=2).reshape((2, 2, 2))

        t = flip(a, 0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.flip(np.arange(8).reshape(2, 2, 2), 0)
        np.testing.assert_equal(res, expected)

        t = flip(a, 1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.flip(np.arange(8).reshape(2, 2, 2), 1)
        np.testing.assert_equal(res, expected)

        t = flipud(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.flipud(np.arange(8).reshape(2, 2, 2))
        np.testing.assert_equal(res, expected)

        t = fliplr(a)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.fliplr(np.arange(8).reshape(2, 2, 2))
        np.testing.assert_equal(res, expected)

    def testRepeatExecution(self):
        a = repeat(3, 4)

        res = self.executor.execute_tensor(a)[0]
        expected = np.repeat(3, 4)
        np.testing.assert_equal(res, expected)

        x_data = np.random.randn(20, 30)
        x = tensor(x_data, chunk_size=(3, 4))

        t = repeat(x, 2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.repeat(x_data, 2)
        np.testing.assert_equal(res, expected)

        t = repeat(x, 3, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.repeat(x_data, 3, axis=1)
        np.testing.assert_equal(res, expected)

        t = repeat(x, np.arange(20), axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.repeat(x_data, np.arange(20), axis=0)
        np.testing.assert_equal(res, expected)

        t = repeat(x, arange(20, chunk_size=5), axis=0)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.repeat(x_data, np.arange(20), axis=0)
        np.testing.assert_equal(res, expected)

        x_data = sps.random(20, 30, density=.1)
        x = tensor(x_data, chunk_size=(3, 4))

        t = repeat(x, 2, axis=1)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.repeat(x_data.toarray(), 2, axis=1)
        np.testing.assert_equal(res.toarray(), expected)

    def testTileExecution(self):
        a_data = np.array([0, 1, 2])
        a = tensor(a_data, chunk_size=2)

        t = tile(a, 2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(a_data, 2)
        np.testing.assert_equal(res, expected)

        t = tile(a, (2, 2))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(a_data, (2, 2))
        np.testing.assert_equal(res, expected)

        t = tile(a, (2, 1, 2))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(a_data, (2, 1, 2))
        np.testing.assert_equal(res, expected)

        b_data = np.array([[1, 2], [3, 4]])
        b = tensor(b_data, chunk_size=1)

        t = tile(b, 2)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(b_data, 2)
        np.testing.assert_equal(res, expected)

        t = tile(b, (2, 1))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(b_data, (2, 1))
        np.testing.assert_equal(res, expected)

        c_data = np.array([1, 2, 3, 4])
        c = tensor(c_data, chunk_size=3)

        t = tile(c, (4, 1))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.tile(c_data, (4, 1))
        np.testing.assert_equal(res, expected)

    def testIsInExecution(self):
        element = 2 * arange(4, chunk_size=1).reshape((2, 2))
        test_elements = [1, 2, 4, 8]

        mask = isin(element, test_elements)

        res = self.executor.execute_tensor(mask, concat=True)[0]
        expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_elements)
        np.testing.assert_equal(res, expected)

        res = self.executor.execute_tensor(element[mask], concat=True)[0]
        expected = np.array([2, 4])
        np.testing.assert_equal(res, expected)

        mask = isin(element, test_elements, invert=True)

        res = self.executor.execute_tensor(mask, concat=True)[0]
        expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_elements, invert=True)
        np.testing.assert_equal(res, expected)

        res = self.executor.execute_tensor(element[mask], concat=True)[0]
        expected = np.array([0, 6])
        np.testing.assert_equal(res, expected)

        test_set = {1, 2, 4, 8}
        mask = isin(element, test_set)

        res = self.executor.execute_tensor(mask, concat=True)[0]
        expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_set)
        np.testing.assert_equal(res, expected)

    def testRavelExecution(self):
        arr = ones((10, 5), chunk_size=2)
        flat_arr = mt.ravel(arr)

        res = self.executor.execute_tensor(flat_arr, concat=True)[0]
        self.assertEqual(len(res), 50)
        np.testing.assert_equal(res, np.ones(50))
