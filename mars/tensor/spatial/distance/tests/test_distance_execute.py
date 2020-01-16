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

from mars.tensor.datasource import tensor
from mars.tensor.spatial import distance
from mars.tests.core import ExecutorForTest


class Test(unittest.TestCase):
    def setUp(self):
        self._executor = ExecutorForTest('numpy')

    @unittest.skipIf(distance.pdist is None, 'scipy not installed')
    def testPdistExecution(self):
        from scipy.spatial.distance import pdist as sp_pdist

        raw = np.random.rand(100, 10)

        # test 1 chunk
        x = tensor(raw, chunk_size=100)

        dist = distance.pdist(x)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw)
        np.testing.assert_array_equal(result, expected)

        dist = distance.pdist(x, metric='hamming')
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw, metric='hamming')
        np.testing.assert_array_equal(result, expected)

        f = lambda u, v: np.sqrt(((u-v)**2).sum())
        dist = distance.pdist(x, metric=f)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw, metric=f)
        np.testing.assert_array_equal(result, expected)

        # test more than 1 chunk
        x = tensor(raw, chunk_size=12)

        dist = distance.pdist(x)
        tdist = dist.tiles()
        self.assertEqual(len(tdist.chunks), 1)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw)
        np.testing.assert_array_equal(result, expected)

        dist = distance.pdist(x, aggregate_size=3)
        tdist = dist.tiles()
        self.assertEqual(len(tdist.chunks), 3)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw)
        np.testing.assert_array_equal(result, expected)

        dist = distance.pdist(x, metric='hamming', aggregate_size=2)
        tdist = dist.tiles()
        self.assertEqual(len(tdist.chunks), 2)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw, metric='hamming')
        np.testing.assert_array_equal(result, expected)

        f = lambda u, v: np.sqrt(((u-v)**2).sum())
        dist = distance.pdist(x, metric=f, aggregate_size=2)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_pdist(raw, metric=f)
        np.testing.assert_array_equal(result, expected)

        for x in [tensor(raw), tensor(raw, chunk_size=12)]:
            # test w
            weight = np.random.rand(10)
            w = tensor(weight, chunk_size=7)
            dist = distance.pdist(x, metric='wminkowski', p=3, w=w)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_pdist(raw, metric='wminkowski', p=3, w=weight)
            np.testing.assert_array_equal(result, expected)

            # test V
            v = np.random.rand(10)
            V = tensor(v, chunk_size=7)
            dist = distance.pdist(x, metric='seuclidean', V=V)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_pdist(raw, metric='seuclidean', V=v)
            np.testing.assert_array_equal(result, expected)

            # test VI
            vi = np.random.rand(10, 10)
            VI = tensor(vi, chunk_size=8)
            dist = distance.pdist(x, metric='mahalanobis', VI=VI)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_pdist(raw, metric='mahalanobis', VI=vi)
            np.testing.assert_array_equal(result, expected)

    @unittest.skipIf(distance.cdist is None, 'scipy not installed')
    def testCdistExecution(self):
        from scipy.spatial.distance import cdist as sp_cdist

        raw_a = np.random.rand(100, 10)
        raw_b = np.random.rand(89, 10)

        # test 1 chunk
        xa = tensor(raw_a, chunk_size=100)
        xb = tensor(raw_b, chunk_size=100)

        dist = distance.cdist(xa, xb)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b)
        np.testing.assert_array_equal(result, expected)

        dist = distance.cdist(xa, xb, metric='hamming')
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b, metric='hamming')
        np.testing.assert_array_equal(result, expected)

        f = lambda u, v: np.sqrt(((u-v)**2).sum())
        dist = distance.cdist(xa, xb, metric=f)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b, metric=f)
        np.testing.assert_array_equal(result, expected)

        # test more than 1 chunk
        xa = tensor(raw_a, chunk_size=12)
        xb = tensor(raw_b, chunk_size=13)

        dist = distance.cdist(xa, xb)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b)
        np.testing.assert_array_equal(result, expected)

        dist = distance.cdist(xa, xb, metric='hamming')
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b, metric='hamming')
        np.testing.assert_array_equal(result, expected)

        f = lambda u, v: np.sqrt(((u-v)**2).sum())
        dist = distance.cdist(xa, xb, metric=f)
        result = self._executor.execute_tensor(dist, concat=True)[0]
        expected = sp_cdist(raw_a, raw_b, metric=f)
        np.testing.assert_array_equal(result, expected)

        for xa, xb in [(tensor(raw_a), tensor(raw_b)),
                       (tensor(raw_a, chunk_size=12), tensor(raw_b, chunk_size=13))]:
            # test w
            weight = np.random.rand(10)
            w = tensor(weight, chunk_size=7)
            dist = distance.cdist(xa, xb, metric='wminkowski', p=3, w=w)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_cdist(raw_a, raw_b, metric='wminkowski', p=3, w=weight)
            np.testing.assert_array_equal(result, expected)

            # test V
            v = np.random.rand(10)
            V = tensor(v, chunk_size=7)
            dist = distance.cdist(xa, xb, metric='seuclidean', V=V)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_cdist(raw_a, raw_b, metric='seuclidean', V=v)
            np.testing.assert_array_equal(result, expected)

            # test VI
            vi = np.random.rand(10, 10)
            VI = tensor(vi, chunk_size=8)
            dist = distance.cdist(xa, xb, metric='mahalanobis', VI=VI)
            result = self._executor.execute_tensor(dist, concat=True)[0]
            expected = sp_cdist(raw_a, raw_b, metric='mahalanobis', VI=vi)
            np.testing.assert_array_equal(result, expected)

    @unittest.skipIf(distance.cdist is None, 'scipy not installed')
    def testSqureFormExecution(self):
        from scipy.spatial.distance import pdist as sp_pdist, \
            squareform as sp_squareform

        raw_a = np.random.rand(80, 10)
        raw_pdsit = sp_pdist(raw_a)
        raw_square = sp_squareform(raw_pdsit)

        # tomatrix, test 1 chunk
        vec = tensor(raw_pdsit, chunk_size=raw_pdsit.shape[0])
        mat = distance.squareform(vec, chunk_size=100)
        result = self._executor.execute_tensor(mat, concat=True)[0]
        np.testing.assert_array_equal(result, raw_square)

        # tomatrix, test more than 1 chunk
        vec = tensor(raw_pdsit, chunk_size=33)
        self.assertGreater(len(vec.tiles().chunks), 1)
        mat = distance.squareform(vec, chunk_size=34)
        result = self._executor.execute_tensor(mat, concat=True)[0]
        np.testing.assert_array_equal(result, raw_square)

        # tovec, test 1 chunk
        mat = tensor(raw_square)
        vec = distance.squareform(mat, chunk_size=raw_pdsit.shape[0])
        self.assertEqual(len(mat.tiles().chunks), 1)
        self.assertEqual(len(vec.tiles().chunks), 1)
        result = self._executor.execute_tensor(vec, concat=True)[0]
        np.testing.assert_array_equal(result, raw_pdsit)

        # tovec, test more than 1 chunk
        mat = tensor(raw_square, chunk_size=31)
        vec = distance.squareform(mat, chunk_size=40)
        self.assertGreater(len(vec.tiles().chunks), 1)
        result = self._executor.execute_tensor(vec, concat=True)[0]
        np.testing.assert_array_equal(result, raw_pdsit)

        # test checks
        # generate non-symmetric matrix
        non_sym_arr = np.random.RandomState(0).rand(10, 10)

        # 1 chunk
        mat = tensor(non_sym_arr)
        vec = distance.squareform(mat, checks=True, chunk_size=100)
        with self.assertRaises(ValueError):
            _ = self._executor.execute_tensor(vec, concat=True)[0]
        # force checks=False
        vec = distance.squareform(mat, checks=False, chunk_size=100)
        _ = self._executor.execute_tensor(vec, concat=True)[0]

        # more than 1 chunk
        mat = tensor(non_sym_arr, chunk_size=6)
        vec = distance.squareform(mat, checks=True, chunk_size=8)
        self.assertGreater(len(vec.tiles().chunks), 1)
        with self.assertRaises(ValueError):
            _ = self._executor.execute_tensor(vec, concat=True)[0]
        # force checks=False
        vec = distance.squareform(mat, checks=False, chunk_size=100)
        _ = self._executor.execute_tensor(vec, concat=True)[0]
