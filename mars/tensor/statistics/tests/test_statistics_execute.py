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

import numpy as np
import scipy.sparse as sps

from mars.tiles import TilesError
from mars.context import LocalContext
from mars.utils import ignore_warning
from mars.tensor.datasource import arange, tensor
from mars.tensor.statistics import average, cov, corrcoef, ptp, \
    digitize, histogram_bin_edges, histogram
from mars.tensor.base import sort
from mars.tensor.merge import stack
from mars.tensor.reduction import all as tall
from mars.tests.core import ExecutorForTest


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

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
        t = average(data, axis=1, weights=tensor([1. / 4, 3. / 4], chunk_size=2))

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.average(np.arange(6).reshape(3, 2), axis=1, weights=(1. / 4, 3. / 4))
        np.testing.assert_equal(res, expected)

        with self.assertRaises(TypeError):
            average(data, weights=tensor([1. / 4, 3. / 4], chunk_size=2))

    def testCovExecution(self):
        data = np.array([[0, 2], [1, 1], [2, 0]]).T
        x = tensor(data, chunk_size=1)

        t = cov(x)

        res = self.executor.execute_tensor(t, concat=True)[0]
        expected = np.cov(data)
        np.testing.assert_equal(res, expected)

        data_x = [-2.1, -1, 4.3]
        data_y = [3, 1.1, 0.12]
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

    @ignore_warning
    def testHistogramBinEdgesExecution(self):
        rs = np.random.RandomState(0)

        raw = rs.randint(10, size=(20,))
        a = tensor(raw, chunk_size=3)

        # range provided
        for range_ in [(0, 10), (3, 11), (3, 7)]:
            bin_edges = histogram_bin_edges(a, range=range_)
            result = self.executor.execute_tensor(bin_edges)[0]
            expected = np.histogram_bin_edges(raw, range=range_)
            np.testing.assert_array_equal(result, expected)

        this = self

        class MockSession:
            def __init__(self):
                self.executor = this.executor

        ctx = LocalContext(MockSession())
        executor = ExecutorForTest('numpy', storage=ctx)
        with ctx:
            raw2 = rs.randint(10, size=(1,))
            b = tensor(raw2)
            raw3 = rs.randint(10, size=(0,))
            c = tensor(raw3)
            for t, r in [(a, raw), (b, raw2), (c, raw3), (sort(a), raw)]:
                # TODO: test 'auto' and 'fd' after `mt.percentile` implemented
                test_bins = [10, 'stone', 'doane', 'rice', 'scott', 'sqrt', 'sturges']
                for bins in test_bins:
                    bin_edges = histogram_bin_edges(t, bins=bins)

                    if r.size > 0:
                        with self.assertRaises(TilesError):
                            executor.execute_tensor(bin_edges)

                    result = executor.execute_tensors([bin_edges])[0]
                    expected = np.histogram_bin_edges(r, bins=bins)
                    np.testing.assert_array_equal(result, expected)

                test_bins = [[0, 4, 8], tensor([0, 4, 8], chunk_size=2)]
                for bins in test_bins:
                    bin_edges = histogram_bin_edges(t, bins=bins)
                    result = executor.execute_tensors([bin_edges])[0]
                    expected = np.histogram_bin_edges(r, bins=[0, 4, 8])
                    np.testing.assert_array_equal(result, expected)

            raw = np.arange(5)
            a = tensor(raw, chunk_size=3)
            bin_edges = histogram_bin_edges(a)
            result = executor.execute_tensors([bin_edges])[0]
            expected = np.histogram_bin_edges(raw)
            self.assertEqual(bin_edges.shape, expected.shape)
            np.testing.assert_array_equal(result, expected)

    @ignore_warning
    def testHistogramExecution(self):
        rs = np.random.RandomState(0)

        raw = rs.randint(10, size=(20,))
        a = tensor(raw, chunk_size=3)
        raw_weights = rs.random(20)
        weights = tensor(raw_weights, chunk_size=4)

        # range provided
        for range_ in [(0, 10), (3, 11), (3, 7)]:
            bin_edges = histogram(a, range=range_)[0]
            result = self.executor.execute_tensor(bin_edges)[0]
            expected = np.histogram(raw, range=range_)[0]
            np.testing.assert_array_equal(result, expected)

        for wt in (raw_weights, weights):
            for density in (True, False):
                bins = [1, 4, 6, 9]
                bin_edges = histogram(a, bins=bins, weights=wt, density=density)[0]
                result = self.executor.execute_tensor(bin_edges)[0]
                expected = np.histogram(
                    raw, bins=bins, weights=raw_weights, density=density)[0]
                np.testing.assert_almost_equal(result, expected)

        this = self

        class MockSession:
            def __init__(self):
                self.executor = this.executor

        ctx = LocalContext(MockSession())
        executor = ExecutorForTest('numpy', storage=ctx)
        with ctx:
            raw2 = rs.randint(10, size=(1,))
            b = tensor(raw2)
            raw3 = rs.randint(10, size=(0,))
            c = tensor(raw3)
            for t, r in [(a, raw), (b, raw2), (c, raw3), (sort(a), raw)]:
                for density in (True, False):
                    # TODO: test 'auto' and 'fd' after `mt.percentile` implemented
                    test_bins = [10, 'stone', 'doane', 'rice', 'scott', 'sqrt', 'sturges']
                    for bins in test_bins:
                        hist = histogram(t, bins=bins, density=density)[0]

                        if r.size > 0:
                            with self.assertRaises(TilesError):
                                executor.execute_tensor(hist)

                        result = executor.execute_tensors([hist])[0]
                        expected = np.histogram(r, bins=bins, density=density)[0]
                        np.testing.assert_array_equal(result, expected)

                    test_bins = [[0, 4, 8], tensor([0, 4, 8], chunk_size=2)]
                    for bins in test_bins:
                        hist = histogram(t, bins=bins, density=density)[0]
                        result = executor.execute_tensors([hist])[0]
                        expected = np.histogram(r, bins=[0, 4, 8], density=density)[0]
                        np.testing.assert_array_equal(result, expected)
