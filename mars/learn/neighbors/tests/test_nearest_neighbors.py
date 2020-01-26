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
try:
    from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
except ImportError:  # pragma: no cover
    SkNearestNeighbors = None

import mars.tensor as mt
from mars.learn.neighbors import NearestNeighbors


@unittest.skipIf(SkNearestNeighbors is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def testNearestNeighborsExecution(self):
        rs = np.random.RandomState(0)
        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        X = mt.tensor(raw_X, chunk_size=7)
        Y = mt.tensor(raw_Y, chunk_size=5)

        for algo in ['brute', 'ball_tree', 'kd_tree', 'auto']:
            for metric in ['minkowski', 'manhattan']:
                nn = NearestNeighbors(n_neighbors=3,
                                      algorithm=algo,
                                      metric=metric)
                nn.fit(X)

                ret = nn.kneighbors(Y)

                snn = SkNearestNeighbors(n_neighbors=3,
                                         algorithm=algo,
                                         metric=metric)
                snn.fit(raw_X)
                expected = snn.kneighbors(raw_Y)

                result = [r.fetch() for r in ret]
                np.testing.assert_almost_equal(result[0], expected[0])
                np.testing.assert_almost_equal(result[1], expected[1])
