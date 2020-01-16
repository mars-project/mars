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
try:
    import sklearn

    from sklearn.metrics.pairwise import cosine_distances as sk_cosine_distances
except ImportError:
    sklearn = None

from mars.tests.core import ExecutorForTest
from mars import tensor as mt
from mars.learn.metrics.pairwise import cosine_distances


@unittest.skipIf(sklearn is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ExecutorForTest('numpy')

    def testCosineDistancesExecution(self):
        raw_dense_x = np.random.rand(25, 10)
        raw_dense_y = np.random.rand(17, 10)

        raw_sparse_x = sps.random(25, 10, density=0.5, format='csr', random_state=0)
        raw_sparse_y = sps.random(17, 10, density=0.4, format='csr', random_state=1)

        for raw_x, raw_y in [
            (raw_dense_x, raw_dense_y),
            (raw_sparse_x, raw_sparse_y)
        ]:
            for chunk_size in (25, 6):
                x = mt.tensor(raw_x, chunk_size=chunk_size)
                y = mt.tensor(raw_y, chunk_size=chunk_size)

                d = cosine_distances(x, y)

                result = self.executor.execute_tensor(d, concat=True)[0]
                expected = sk_cosine_distances(raw_x, raw_y)

                np.testing.assert_almost_equal(np.asarray(result), expected)

                d = cosine_distances(x)

                result = self.executor.execute_tensor(d, concat=True)[0]
                expected = sk_cosine_distances(raw_x)

                np.testing.assert_almost_equal(np.asarray(result), expected)
