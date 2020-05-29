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

from mars.session import new_session

try:
    import sklearn

    from sklearn.metrics.pairwise import manhattan_distances as sk_manhattan_distances
except ImportError:  # pragma: no cover
    sklearn = None

from mars import tensor as mt
from mars.tests.core import ExecutorForTest
from mars.learn.metrics.pairwise import manhattan_distances


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testManhattanDistances(self):
        x = mt.random.randint(10, size=(10, 3), density=0.4)
        y = mt.random.randint(10, size=(11, 3), density=0.5)

        with self.assertRaises(TypeError):
            manhattan_distances(x, y, sum_over_features=False)

        x = x.todense()
        y = y.todense()

        d = manhattan_distances(x, y, sum_over_features=True)
        self.assertEqual(d.shape, (10, 11))
        d = manhattan_distances(x, y, sum_over_features=False)
        self.assertEqual(d.shape, (110, 3))

    def testManhattanDistancesExecution(self):
        raw_x = np.random.rand(20, 5)
        raw_y = np.random.rand(21, 5)

        x1 = mt.tensor(raw_x, chunk_size=30)
        y1 = mt.tensor(raw_y, chunk_size=30)

        x2 = mt.tensor(raw_x, chunk_size=11)
        y2 = mt.tensor(raw_y, chunk_size=12)

        raw_sparse_x = sps.random(20, 5, density=0.4, format='csr', random_state=0)
        raw_sparse_y = sps.random(21, 5, density=0.3, format='csr', random_state=0)

        x3 = mt.tensor(raw_sparse_x, chunk_size=30)
        y3 = mt.tensor(raw_sparse_y, chunk_size=30)

        x4 = mt.tensor(raw_sparse_x, chunk_size=11)
        y4 = mt.tensor(raw_sparse_y, chunk_size=12)

        for x, y, is_sparse in [(x1, y1, False),
                                (x2, y2, False),
                                (x3, y3, True),
                                (x4, y4, True)]:
            if is_sparse:
                rx, ry = raw_sparse_x, raw_sparse_y
            else:
                rx, ry = raw_x, raw_y

            sv = [True, False] if not is_sparse else [True]

            for sum_over_features in sv:
                d = manhattan_distances(x, y, sum_over_features)

                result = self.executor.execute_tensor(d, concat=True)[0]
                expected = sk_manhattan_distances(rx, ry, sum_over_features)

                np.testing.assert_almost_equal(result, expected)

                d = manhattan_distances(x, sum_over_features=sum_over_features)

                result = self.executor.execute_tensor(d, concat=True)[0]
                expected = sk_manhattan_distances(rx, sum_over_features=sum_over_features)

                np.testing.assert_almost_equal(result, expected)
