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
from mars.learn.neighbors._faiss import _build_faiss_index, _load_index
from mars.learn.neighbors import NearestNeighbors
from mars.tiles import get_tiled
from mars.tests.core import ExecutorForTest


@unittest.skipIf(faiss is None, 'faiss not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ExecutorForTest('numpy')

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
            index = _build_faiss_index(X, 'Flat', None, random_state=0,
                                       same_distribution=True, return_index_type=index_type)
            faiss_index = self.executor.execute_tileable(index)

            index_shards = faiss.IndexShards()
            for ind in faiss_index:
                shard = _load_index(None, index.op, ind, -1)
                index_shards.add_shard(shard)
            faiss_index = index_shards

            faiss_index.nprob = 10
            _, indices = faiss_index.search(y, k=5)

            np.testing.assert_array_equal(indices, expected_indices.fetch())

        # test one chunk, brute force
        X = mt.tensor(x, chunk_size=50)
        index = _build_faiss_index(X, 'Flat', None, random_state=0,
                                   same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        faiss_index.nprob = 10
        _, indices = faiss_index.search(y, k=5)

        np.testing.assert_array_equal(indices, expected_indices.fetch())

        # test train, same distribution
        X = mt.tensor(x, chunk_size=10)
        index = _build_faiss_index(X, 'IVF30,Flat', 30, random_state=0,
                                   same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        self.assertIsInstance(faiss_index, faiss.IndexIVFFlat)
        self.assertEqual(faiss_index.ntotal, n)
        self.assertEqual(len(get_tiled(index).chunks), 1)

        # test train, distributions are variant
        X = mt.tensor(x, chunk_size=10)
        index = _build_faiss_index(X, 'IVF10,Flat', None, random_state=0,
                                   same_distribution=False, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)

        self.assertEqual(len(faiss_index), 5)
        for ind in faiss_index:
            self.assertIsInstance(ind, faiss.IndexIVFFlat)
            self.assertEqual(ind.ntotal, 10)

        # test one chunk, train
        X = mt.tensor(x, chunk_size=50)
        index = _build_faiss_index(X, 'IVF30,Flat', 30, random_state=0,
                                   same_distribution=True, return_index_type='object')
        faiss_index = self.executor.execute_tileable(index)[0]

        self.assertIsInstance(faiss_index, faiss.IndexIVFFlat)
        self.assertEqual(faiss_index.ntotal, n)

        # test wrong index
        with self.assertRaises(ValueError):
            _build_faiss_index(X, 'unknown_index', None)
