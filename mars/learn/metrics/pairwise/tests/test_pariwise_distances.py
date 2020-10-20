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

from mars.session import new_session

try:
    import sklearn

    from sklearn.metrics import pairwise_distances as sk_pairwise_distances
    from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
    from sklearn.exceptions import DataConversionWarning
    from sklearn.utils._testing import assert_warns
except ImportError:
    sklearn = None

from mars import tensor as mt
from mars.learn.metrics import pairwise_distances, pairwise_distances_topk
from mars.tests.core import ExecutorForTest


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testPairwiseDistancesExecution(self):
        raw_x = np.random.rand(20, 5)
        raw_y = np.random.rand(21, 5)

        x = mt.tensor(raw_x, chunk_size=11)
        y = mt.tensor(raw_y, chunk_size=12)

        d = pairwise_distances(x, y)
        result = self.executor.execute_tensor(d, concat=True)[0]
        expected = sk_pairwise_distances(raw_x, raw_y)
        np.testing.assert_almost_equal(result, expected)

        # test precomputed
        d2 = d.copy()
        d2[0, 0] = -1
        d2 = pairwise_distances(d2, y, metric='precomputed')
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tensor(d2, concat=True)[0]

        # test cdist
        weight = np.random.rand(5)
        d = pairwise_distances(x, y, metric='wminkowski', p=3,
                               w=weight)
        result = self.executor.execute_tensor(d, concat=True)[0]
        expected = sk_pairwise_distances(raw_x, raw_y, metric='wminkowski',
                                         p=3, w=weight)
        np.testing.assert_almost_equal(result, expected)

        # test pdist
        d = pairwise_distances(x, metric='hamming')
        result = self.executor.execute_tensor(d, concat=True)[0]
        expected = sk_pairwise_distances(raw_x, metric='hamming')
        np.testing.assert_almost_equal(result, expected)

        # test function metric
        m = lambda u, v: np.sqrt(((u-v)**2).sum())
        d = pairwise_distances(x, y, metric=m)
        result = self.executor.execute_tensor(d, concat=True)[0]
        expected = sk_pairwise_distances(raw_x, raw_y, metric=m)
        np.testing.assert_almost_equal(result, expected)

        assert_warns(DataConversionWarning,
                     pairwise_distances, x, y, metric='jaccard')

        with self.assertRaises(ValueError):
            _ = pairwise_distances(x, y, metric='unknown')

    def testPairwiseDistancesTopkExecution(self):
        rs = np.random.RandomState(0)
        raw_x = rs.rand(20, 5)
        raw_y = rs.rand(21, 5)

        x = mt.tensor(raw_x, chunk_size=11)
        y = mt.tensor(raw_y, chunk_size=12)

        d, i = pairwise_distances_topk(x, y, 3, metric='euclidean', return_index=True)
        result = self.executor.execute_tensors([d, i])
        nn = SkNearestNeighbors(n_neighbors=3, algorithm='brute',
                                metric='euclidean')
        nn.fit(raw_y)
        expected = nn.kneighbors(raw_x, return_distance=True)
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])

        x = mt.tensor(raw_x, chunk_size=(11, 3))

        d = pairwise_distances_topk(x, k=4, metric='euclidean', return_index=False)
        result = self.executor.execute_tensor(d, concat=True)[0]
        nn = SkNearestNeighbors(n_neighbors=3, algorithm='brute',
                                metric='euclidean')
        nn.fit(raw_x)
        expected = nn.kneighbors(return_distance=True)[0]
        np.testing.assert_almost_equal(result[:, 1:], expected)

        y = mt.tensor(raw_y, chunk_size=21)

        d, i = pairwise_distances_topk(x, y, 3, metric='cosine',
                                       return_index=True, working_memory='168')
        result = self.executor.execute_tensors([d, i])
        nn = SkNearestNeighbors(n_neighbors=3, algorithm='brute',
                                metric='cosine')
        nn.fit(raw_y)
        expected = nn.kneighbors(raw_x, return_distance=True)
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])

        d = pairwise_distances_topk(x, y, 3, metric='cosine',
                                    axis=0, return_index=False)
        result = self.executor.execute_tensor(d, concat=True)[0]
        nn = SkNearestNeighbors(n_neighbors=3, algorithm='brute',
                                metric='cosine')
        nn.fit(raw_x)
        expected = nn.kneighbors(raw_y, return_distance=True)[0]
        np.testing.assert_almost_equal(result, expected)
