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
    import faiss
except ImportError:  # pragma: no cover
    faiss = None
try:
    import cupy
except ImportError:
    cupy = None
try:
    from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
    from sklearn.neighbors import BallTree as SkBallTree
    from sklearn.neighbors import KDTree as SkKDTree
    from sklearn.utils.testing import assert_warns
except ImportError:  # pragma: no cover
    SkNearestNeighbors = None

import mars.tensor as mt
from mars.lib.sparse import SparseNDArray
from mars.tiles import get_tiled
from mars.learn.neighbors import NearestNeighbors


@unittest.skipIf(SkNearestNeighbors is None, 'scikit-learn not installed')
class Test(unittest.TestCase):
    def testNearestNeighbors(self):
        rs = np.random.RandomState(0)
        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        X = mt.tensor(raw_X)
        Y = mt.tensor(raw_Y)

        raw_sparse_x = sps.random(10, 5, density=0.5, format='csr', random_state=rs)
        raw_sparse_y = sps.random(8, 5, density=0.4, format='csr', random_state=rs)

        X_sparse = mt.tensor(raw_sparse_x)
        Y_sparse = mt.tensor(raw_sparse_y)

        metric_func = lambda u, v: np.sqrt(((u-v)**2).sum())

        _ = NearestNeighbors(algorithm='auto', metric='precomputed', metric_params={})

        with self.assertRaises(ValueError):
            _ = NearestNeighbors(algorithm='unknown')

        with self.assertRaises(ValueError):
            _ = NearestNeighbors(algorithm='kd_tree', metric=metric_func)

        with self.assertRaises(ValueError):
            _ = NearestNeighbors(algorithm='auto', metric='unknown')

        assert_warns(SyntaxWarning, NearestNeighbors, metric_params={'p': 1})

        with self.assertRaises(ValueError):
            _ = NearestNeighbors(metric='wminkowski', p=0)

        with self.assertRaises(ValueError):
            _ = NearestNeighbors(algorithm='auto', metric='minkowski', p=0)

        nn = NearestNeighbors(algorithm='auto', metric='minkowski', p=1)
        nn.fit(X)
        self.assertEqual(nn.effective_metric_, 'manhattan')

        nn = NearestNeighbors(algorithm='auto', metric='minkowski', p=2)
        nn.fit(X)
        self.assertEqual(nn.effective_metric_, 'euclidean')

        nn = NearestNeighbors(algorithm='auto', metric='minkowski', p=np.inf)
        nn.fit(X)
        self.assertEqual(nn.effective_metric_, 'chebyshev')

        nn2 = NearestNeighbors(algorithm='auto', metric='minkowski')
        nn2.fit(nn)
        self.assertEqual(nn2._fit_method, nn._fit_method)

        nn = NearestNeighbors(algorithm='auto', metric='minkowski')
        ball_tree = SkBallTree(raw_X)
        nn.fit(ball_tree)
        self.assertEqual(nn._fit_method, 'ball_tree')

        nn = NearestNeighbors(algorithm='auto', metric='minkowski')
        kd_tree = SkKDTree(raw_X)
        nn.fit(kd_tree)
        self.assertEqual(nn._fit_method, 'kd_tree')

        with self.assertRaises(ValueError):
            nn = NearestNeighbors()
            nn.fit(np.random.rand(0, 10))

        nn = NearestNeighbors(algorithm='ball_tree')
        assert_warns(UserWarning, nn.fit, X_sparse)

        nn = NearestNeighbors(metric='haversine')
        with self.assertRaises(ValueError):
            nn.fit(X_sparse)

        nn = NearestNeighbors(metric=metric_func, n_neighbors=1)
        nn.fit(X)
        self.assertEqual(nn._fit_method, 'ball_tree')

        nn = NearestNeighbors(metric='sqeuclidean', n_neighbors=1)
        nn.fit(X)
        self.assertEqual(nn._fit_method, 'brute')

        with self.assertRaises(ValueError):
            nn = NearestNeighbors(n_neighbors=-1)
            nn.fit(X)

        with self.assertRaises(TypeError):
            nn = NearestNeighbors(n_neighbors=1.3)
            nn.fit(X)

        nn = NearestNeighbors()
        nn.fit(X)
        with self.assertRaises(ValueError):
            nn.kneighbors(Y, n_neighbors=-1)
        with self.assertRaises(TypeError):
            nn.kneighbors(Y, n_neighbors=1.3)
        with self.assertRaises(ValueError):
            nn.kneighbors(Y, n_neighbors=11)

        nn = NearestNeighbors(algorithm='ball_tree')
        nn.fit(X)
        with self.assertRaises(ValueError):
            nn.kneighbors(Y_sparse)

    def testNearestNeighborsExecution(self):
        rs = np.random.RandomState(0)
        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        X = mt.tensor(raw_X, chunk_size=7)
        Y = mt.tensor(raw_Y, chunk_size=(5, 3))

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

                if nn._tree is not None:
                    self.assertIsInstance(nn._tree.fetch(), type(snn._tree))

                # test return_distance=False
                ret = nn.kneighbors(Y, return_distance=False)

                result = ret.fetch()
                np.testing.assert_almost_equal(result, expected[1])

                # test y is x
                ret = nn.kneighbors()

                expected = snn.kneighbors()

                result = [r.fetch() for r in ret]
                np.testing.assert_almost_equal(result[0], expected[0])
                np.testing.assert_almost_equal(result[1], expected[1])

                # test y is x, and return_distance=False
                ret = nn.kneighbors(return_distance=False)

                result = ret.fetch()
                np.testing.assert_almost_equal(result, expected[1])

        # test callable metric
        metric = lambda u, v: np.sqrt(((u-v)**2).sum())
        for algo in ['brute', 'ball_tree']:
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

        # test sparse
        raw_sparse_x = sps.random(10, 5, density=0.5, format='csr', random_state=rs)
        raw_sparse_y = sps.random(8, 5, density=0.4, format='csr', random_state=rs)

        X = mt.tensor(raw_sparse_x, chunk_size=7)
        Y = mt.tensor(raw_sparse_y, chunk_size=5)

        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(X)

        ret = nn.kneighbors(Y)

        snn = SkNearestNeighbors(n_neighbors=3)
        snn.fit(raw_sparse_x)
        expected = snn.kneighbors(raw_sparse_y)

        result = [r.fetch() for r in ret]
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

        # test input with unknown shape
        X = mt.tensor(raw_X, chunk_size=7)
        X = X[X[:, 0] > 0.1]
        Y = mt.tensor(raw_Y, chunk_size=(5, 3))
        Y = Y[Y[:, 0] > 0.1]

        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(X)

        ret = nn.kneighbors(Y)

        x2 = raw_X[raw_X[:, 0] > 0.1]
        y2 = raw_Y[raw_Y[:, 0] > 0.1]
        snn = SkNearestNeighbors(n_neighbors=3)
        snn.fit(x2)
        expected = snn.kneighbors(y2)

        result = ret.fetch()
        self.assertEqual(nn._fit_method, snn._fit_method)
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

        # test serialization
        graph = ret[0].build_graph()
        self.assertEqual(len(graph.from_pb(graph.to_pb())), len(graph))
        self.assertEqual(len(graph.from_json(graph.to_json())), len(graph))

        # test fit a sklearn tree
        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(snn._tree)

        ret = nn.kneighbors(Y)
        result = ret.fetch()
        self.assertEqual(nn._fit_method, snn._fit_method)
        np.testing.assert_almost_equal(result[0], expected[0])
        np.testing.assert_almost_equal(result[1], expected[1])

        # test serialization
        graph = ret[0].build_graph()
        self.assertEqual(len(graph.from_pb(graph.to_pb())), len(graph))
        self.assertEqual(len(graph.from_json(graph.to_json())), len(graph))

    def testKNeighborsGraphExecution(self):
        rs = np.random.RandomState(0)
        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        X = mt.tensor(raw_X, chunk_size=7)
        Y = mt.tensor(raw_Y, chunk_size=(5, 3))

        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(X)
        sklearn_neigh = SkNearestNeighbors(n_neighbors=3)
        sklearn_neigh.fit(raw_X)

        for mode in ['connectivity', 'distance']:
            graph = neigh.kneighbors_graph(Y, mode=mode)
            result = graph.fetch()

            self.assertIsInstance(result, SparseNDArray)
            self.assertGreater(len(get_tiled(graph).chunks), 1)

            expected = sklearn_neigh.kneighbors_graph(raw_Y, mode=mode)

            np.testing.assert_array_equal(result.toarray(),
                                          expected.toarray())

            graph2 = neigh.kneighbors_graph(mode=mode)
            result2 = graph2.fetch()

            self.assertIsInstance(result2, SparseNDArray)
            self.assertGreater(len(get_tiled(graph2).chunks), 1)

            expected2 = sklearn_neigh.kneighbors_graph(mode=mode)

            np.testing.assert_array_equal(result2.toarray(),
                                          expected2.toarray())

        X = [[0], [3], [1]]

        neigh = NearestNeighbors(n_neighbors=2)
        sklearn_neigh = SkNearestNeighbors(n_neighbors=2)
        neigh.fit(X)
        sklearn_neigh.fit(X)

        A = neigh.kneighbors_graph(X).fetch()
        expected_A = sklearn_neigh.kneighbors_graph(X)
        np.testing.assert_array_equal(A.toarray(), expected_A.toarray())

        # test wrong mode
        with self.assertRaises(ValueError):
            _ = neigh.kneighbors_graph(mode='unknown')

    @unittest.skipIf(faiss is None, 'faiss not installed')
    def testFaissNearestNeighborsExecution(self):
        rs = np.random.RandomState(0)
        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        # test faiss execution
        X = mt.tensor(raw_X, chunk_size=7)
        Y = mt.tensor(raw_Y, chunk_size=(5, 3))

        nn = NearestNeighbors(n_neighbors=3, algorithm='faiss', metric='l2')
        nn.fit(X)

        ret = nn.kneighbors(Y)

        snn = SkNearestNeighbors(n_neighbors=3, algorithm='auto', metric='l2')
        snn.fit(raw_X)
        expected = snn.kneighbors(raw_Y)

        result = [r.fetch() for r in ret]
        np.testing.assert_almost_equal(result[0], expected[0], decimal=6)
        np.testing.assert_almost_equal(result[1], expected[1])

        # test return_distance=False
        ret = nn.kneighbors(Y, return_distance=False)

        result = ret.fetch()
        np.testing.assert_almost_equal(result, expected[1])

        # test y is x
        ret = nn.kneighbors()

        expected = snn.kneighbors()

        result = [r.fetch() for r in ret]
        np.testing.assert_almost_equal(result[0], expected[0], decimal=5)
        np.testing.assert_almost_equal(result[1], expected[1])

    @unittest.skipIf(cupy is None or faiss is None, 'either cupy or faiss not installed')
    def testGPUFaissNearestNeighborsExecution(self):
        rs = np.random.RandomState(0)

        raw_X = rs.rand(10, 5)
        raw_Y = rs.rand(8, 5)

        # test faiss execution
        X = mt.tensor(raw_X, chunk_size=7).to_gpu()
        Y = mt.tensor(raw_Y, chunk_size=8).to_gpu()

        nn = NearestNeighbors(n_neighbors=3, algorithm='faiss', metric='l2')
        nn.fit(X)

        ret = nn.kneighbors(Y)

        snn = SkNearestNeighbors(n_neighbors=3, algorithm='auto', metric='l2')
        snn.fit(raw_X)
        expected = snn.kneighbors(raw_Y)

        result = [r.fetch() for r in ret]
        np.testing.assert_almost_equal(result[0].get(), expected[0], decimal=6)
        np.testing.assert_almost_equal(result[1].get(), expected[1])
