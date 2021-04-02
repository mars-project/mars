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
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from mars import tensor as mt
from mars.core import get_tiled
from mars.learn.neighbors._faiss import build_faiss_index, _load_index, \
    faiss_query, _gen_index_string_and_sample_count
from mars.learn.neighbors import NearestNeighbors
from mars.session import new_session
from mars.tests.core import ExecutorForTest


@unittest.skipIf(faiss is None, 'faiss not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testManualBuildFaissIndex(self):
        d = 8
        n = 50
        n_test = 10
        x = np.random.RandomState(0).rand(n, d).astype(np.float32)
        y = np.random.RandomState(0).rand(n_test, d).astype(np.float32)

        nn = NearestNeighbors(algorithm='kd_tree')
        nn.fit(x)
        _, expected_indices = nn.kneighbors(y, 5)

        for index_type in ['object', 'filename', 'bytes']:
            # test brute-force search
            X = mt.tensor(x, chunk_size=10)
            index = build_faiss_index(X, 'Flat', None, random_state=0,
                                      same_distribution=True, return_index_type=index_type)
            faiss_index = self.executor.execute_tileable(index)

            index_shards = faiss.IndexShards(d)
            for ind in faiss_index:
                shard = _load_index(None, index.op, ind, -1)
                index_shards.add_shard(shard)
            faiss_index = index_shards

            faiss_index.nprob = 10
            _, indices = faiss_index.search(y, k=5)

            np.testing.assert_array_equal(indices, expected_indices.fetch())

        # test one chunk, brute force
        X = mt.tensor(x, chunk_size=50)
        index = build_faiss_index(X, 'Flat', None, random_state=0,
                                  same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        faiss_index.nprob = 10
        _, indices = faiss_index.search(y, k=5)

        np.testing.assert_array_equal(indices, expected_indices.fetch())

        # test train, same distribution
        X = mt.tensor(x, chunk_size=10)
        index = build_faiss_index(X, 'IVF30,Flat', 30, random_state=0,
                                  same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        self.assertIsInstance(faiss_index, faiss.IndexIVFFlat)
        self.assertEqual(faiss_index.ntotal, n)
        self.assertEqual(len(get_tiled(index).chunks), 1)

        # test train, distributions are variant
        X = mt.tensor(x, chunk_size=10)
        index = build_faiss_index(X, 'IVF10,Flat', None, random_state=0,
                                  same_distribution=False, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)

        self.assertEqual(len(faiss_index), 5)
        for ind in faiss_index:
            self.assertIsInstance(ind, faiss.IndexIVFFlat)
            self.assertEqual(ind.ntotal, 10)

        # test more index type
        index = build_faiss_index(X, 'PCAR6,IVF8_HNSW32,SQ8', 10, random_state=0,
                                  return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)

        self.assertEqual(len(faiss_index), 5)
        for ind in faiss_index:
            self.assertIsInstance(ind, faiss.IndexPreTransform)
            self.assertEqual(ind.ntotal, 10)

        # test one chunk, train
        X = mt.tensor(x, chunk_size=50)
        index = build_faiss_index(X, 'IVF30,Flat', 30, random_state=0,
                                  same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        self.assertIsInstance(faiss_index, faiss.IndexIVFFlat)
        self.assertEqual(faiss_index.ntotal, n)

        # test wrong index
        with self.assertRaises(ValueError):
            build_faiss_index(X, 'unknown_index', None)

        # test unknown metric
        with self.assertRaises(ValueError):
            build_faiss_index(X, 'Flat', None, metric='unknown_metric')

    def testFaissQuery(self):
        d = 8
        n = 50
        n_test = 10
        x = np.random.RandomState(0).rand(n, d).astype(np.float32)
        y = np.random.RandomState(1).rand(n_test, d).astype(np.float32)

        test_tensors = [
            # multi chunks
            (mt.tensor(x, chunk_size=(20, 5)), mt.tensor(y, chunk_size=5)),
            # one chunk
            (mt.tensor(x, chunk_size=50), mt.tensor(y, chunk_size=10))
        ]

        for X, Y in test_tensors:
            for metric in ['l2', 'cosine']:
                faiss_index = build_faiss_index(X, 'Flat', None, metric=metric,
                                                random_state=0, return_index_type='object')
                d, i = faiss_query(faiss_index, Y, 5, nprobe=10)
                distance, indices = self.executor.execute_tensors([d, i])

                nn = NearestNeighbors(metric=metric)
                nn.fit(x)
                expected_distance, expected_indices = nn.kneighbors(y, 5)

                np.testing.assert_array_equal(indices, expected_indices.fetch())
                np.testing.assert_almost_equal(
                    distance, expected_distance.fetch(), decimal=4)

                # test other index
                X2 = X.astype(np.float64)
                Y2 = y.astype(np.float64)
                faiss_index = build_faiss_index(X2, 'PCAR6,IVF8_HNSW32,SQ8', 10,
                                                random_state=0, return_index_type='object')
                d, i = faiss_query(faiss_index, Y2, 5, nprobe=10)
                # test execute only
                self.executor.execute_tensors([d, i])

    def testGenIndexStringAndSampleCount(self):
        d = 32

        # accuracy=True, could be Flat only
        ret = _gen_index_string_and_sample_count((10 ** 9, d), None, True, 'minimum')
        self.assertEqual(ret, ('Flat', None))

        # no memory concern
        ret = _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'maximum')
        self.assertEqual(ret, ('HNSW32', None))
        index = faiss.index_factory(d, ret[0])
        self.assertTrue(index.is_trained)

        # memory concern not much
        ret = _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'high')
        self.assertEqual(ret, ('IVF1580,Flat', 47400))
        index = faiss.index_factory(d, ret[0])
        self.assertFalse(index.is_trained)

        # memory quite important
        ret = _gen_index_string_and_sample_count((5 * 10 ** 6, d), None, False, 'low')
        self.assertEqual(ret, ('PCAR16,IVF65536_HNSW32,SQ8', 32 * 65536))
        index = faiss.index_factory(d, ret[0])
        self.assertFalse(index.is_trained)

        # memory very important
        ret = _gen_index_string_and_sample_count((10 ** 8, d), None, False, 'minimum')
        self.assertEqual(ret, ('OPQ16_32,IVF1048576_HNSW32,PQ16', 64 * 65536))
        index = faiss.index_factory(d, ret[0])
        self.assertFalse(index.is_trained)

        ret = _gen_index_string_and_sample_count((10 ** 10, d), None, False, 'low')
        self.assertEqual(ret, ('PCAR16,IVF1048576_HNSW32,SQ8', 64 * 65536))
        index = faiss.index_factory(d, ret[0])
        self.assertFalse(index.is_trained)

        with self.assertRaises(ValueError):
            # M > 64 raise error
            _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'maximum', M=128)

        with self.assertRaises(ValueError):
            # M > 64
            _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'minimum', M=128)

        with self.assertRaises(ValueError):
            # dim should be multiple of M
            _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'minimum', M=16, dim=17)

        with self.assertRaises(ValueError):
            _gen_index_string_and_sample_count((10 ** 5, d), None, False, 'low', k=5)

    def testAutoIndex(self):
        d = 8
        n = 50
        n_test = 10
        x = np.random.RandomState(0).rand(n, d).astype(np.float32)
        y = np.random.RandomState(1).rand(n_test, d).astype(np.float32)

        for chunk_size in (50, 20):
            X = mt.tensor(x, chunk_size=chunk_size)

            faiss_index = build_faiss_index(X, random_state=0, return_index_type='object')
            d, i = faiss_query(faiss_index, y, 5, nprobe=10)
            indices = self.executor.execute_tensor(i, concat=True)[0]

            nn = NearestNeighbors()
            nn.fit(x)
            expected_indices = nn.kneighbors(y, 5, return_distance=False)

            np.testing.assert_array_equal(indices, expected_indices)
