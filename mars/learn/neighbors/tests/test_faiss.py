# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import numpy as np
import pytest

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from .... import tensor as mt
from ....core import tile
from ....session import execute, fetch
from .. import NearestNeighbors
from .._faiss import (
    build_faiss_index,
    _load_index,
    faiss_query,
    _gen_index_string_and_sample_count,
)


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_manual_build_faiss_index(setup):
    d = 8
    n = 50
    n_test = 10
    x = np.random.RandomState(0).rand(n, d).astype(np.float32)
    y = np.random.RandomState(0).rand(n_test, d).astype(np.float32)

    nn = NearestNeighbors(algorithm="kd_tree")
    nn.fit(x)
    _, expected_indices = nn.kneighbors(y, 5)

    # test brute-force search
    X = mt.tensor(x, chunk_size=10)
    index = build_faiss_index(X, "Flat", None, random_state=0, same_distribution=True)
    faiss_index = index.execute().fetch()

    index_shards = faiss.IndexShards(d)
    for ind in faiss_index:
        shard = _load_index(ind, -1)
        index_shards.add_shard(shard)
    faiss_index = index_shards

    faiss_index.nprob = 10
    _, indices = faiss_index.search(y, k=5)

    np.testing.assert_array_equal(indices, expected_indices.fetch())

    # test one chunk, brute force
    X = mt.tensor(x, chunk_size=50)
    index = build_faiss_index(X, "Flat", None, random_state=0, same_distribution=True)
    faiss_index = _load_index(index.execute().fetch(), -1)

    faiss_index.nprob = 10
    _, indices = faiss_index.search(y, k=5)

    np.testing.assert_array_equal(indices, expected_indices.fetch())

    # test train, same distribution
    X = mt.tensor(x, chunk_size=10)
    index = build_faiss_index(
        X, "IVF30,Flat", 30, random_state=0, same_distribution=True
    )
    faiss_index = _load_index(index.execute().fetch(), -1)

    assert isinstance(faiss_index, faiss.IndexIVFFlat)
    assert faiss_index.ntotal == n
    assert len(tile(index).chunks) == 1

    # test train, distributions are variant
    X = mt.tensor(x, chunk_size=10)
    index = build_faiss_index(
        X, "IVF10,Flat", None, random_state=0, same_distribution=False
    )
    faiss_index = index.execute().fetch()

    assert len(faiss_index) == 5
    for ind in faiss_index:
        ind = _load_index(ind, -1)
        assert isinstance(ind, faiss.IndexIVFFlat)
        assert ind.ntotal == 10

    # test more index type
    index = build_faiss_index(X, "PCAR6,IVF8_HNSW32,SQ8", 10, random_state=0)
    faiss_index = index.execute().fetch()

    assert len(faiss_index) == 5
    for ind in faiss_index:
        ind = _load_index(ind, -1)
        assert isinstance(ind, faiss.IndexPreTransform)
        assert ind.ntotal == 10

    # test one chunk, train
    X = mt.tensor(x, chunk_size=50)
    index = build_faiss_index(
        X, "IVF30,Flat", 30, random_state=0, same_distribution=True
    )
    faiss_index = _load_index(index.execute().fetch(), -1)

    assert isinstance(faiss_index, faiss.IndexIVFFlat)
    assert faiss_index.ntotal == n

    # test wrong index
    with pytest.raises(ValueError):
        build_faiss_index(X, "unknown_index", None)

    # test unknown metric
    with pytest.raises(ValueError):
        build_faiss_index(X, "Flat", None, metric="unknown_metric")


d = 8
n = 50
n_test = 10
x = np.random.RandomState(0).rand(n, d).astype(np.float32)
y = np.random.RandomState(1).rand(n_test, d).astype(np.float32)


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.parametrize(
    "X, Y",
    [
        # multi chunks
        (mt.tensor(x, chunk_size=(20, 5)), mt.tensor(y, chunk_size=5)),
        # one chunk
        (mt.tensor(x, chunk_size=50), mt.tensor(y, chunk_size=10)),
    ],
)
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_faiss_query(setup, X, Y, metric):
    faiss_index = build_faiss_index(X, "Flat", None, metric=metric, random_state=0)
    d, i = faiss_query(faiss_index, Y, 5, nprobe=10)
    distance, indices = fetch(*execute(d, i))

    nn = NearestNeighbors(metric=metric)
    nn.fit(x)
    expected_distance, expected_indices = nn.kneighbors(y, 5)

    np.testing.assert_array_equal(indices, expected_indices.fetch())
    np.testing.assert_almost_equal(distance, expected_distance.fetch(), decimal=4)

    # test other index
    X2 = X.astype(np.float64)
    Y2 = y.astype(np.float64)
    faiss_index = build_faiss_index(
        X2, "PCAR6,IVF8_HNSW32,SQ8", 10, random_state=0, return_index_type="object"
    )
    d, i = faiss_query(faiss_index, Y2, 5, nprobe=10)
    # test execute only
    execute(d, i)


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_gen_index_string_and_sample_count(setup):
    d = 32

    # accuracy=True, could be Flat only
    ret = _gen_index_string_and_sample_count((10**9, d), None, True, "minimum")
    assert ret == ("Flat", None)

    # no memory concern
    ret = _gen_index_string_and_sample_count((10**5, d), None, False, "maximum")
    assert ret == ("HNSW32", None)
    index = faiss.index_factory(d, ret[0])
    assert index.is_trained is True

    # memory concern not much
    ret = _gen_index_string_and_sample_count((10**5, d), None, False, "high")
    assert ret == ("IVF1580,Flat", 47400)
    index = faiss.index_factory(d, ret[0])
    assert index.is_trained is False

    # memory quite important
    ret = _gen_index_string_and_sample_count((5 * 10**6, d), None, False, "low")
    assert ret == ("PCAR16,IVF65536_HNSW32,SQ8", 32 * 65536)
    index = faiss.index_factory(d, ret[0])
    assert index.is_trained is False

    # memory very important
    ret = _gen_index_string_and_sample_count((10**8, d), None, False, "minimum")
    assert ret == ("OPQ16_32,IVF1048576_HNSW32,PQ16", 64 * 65536)
    index = faiss.index_factory(d, ret[0])
    assert index.is_trained is False

    ret = _gen_index_string_and_sample_count((10**10, d), None, False, "low")
    assert ret == ("PCAR16,IVF1048576_HNSW32,SQ8", 64 * 65536)
    index = faiss.index_factory(d, ret[0])
    assert index.is_trained is False

    with pytest.raises(ValueError):
        # M > 64 raise error
        _gen_index_string_and_sample_count((10**5, d), None, False, "maximum", M=128)

    with pytest.raises(ValueError):
        # M > 64
        _gen_index_string_and_sample_count((10**5, d), None, False, "minimum", M=128)

    with pytest.raises(ValueError):
        # dim should be multiple of M
        _gen_index_string_and_sample_count(
            (10**5, d), None, False, "minimum", M=16, dim=17
        )

    with pytest.raises(ValueError):
        _gen_index_string_and_sample_count((10**5, d), None, False, "low", k=5)


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_auto_index(setup):
    d = 8
    n = 50
    n_test = 10
    x = np.random.RandomState(0).rand(n, d).astype(np.float32)
    y = np.random.RandomState(1).rand(n_test, d).astype(np.float32)

    for chunk_size in (50, 20):
        X = mt.tensor(x, chunk_size=chunk_size)

        faiss_index = build_faiss_index(X, random_state=0, return_index_type="object")
        d, i = faiss_query(faiss_index, y, 5, nprobe=10)
        indices = i.execute().fetch()

        nn = NearestNeighbors()
        nn.fit(x)
        expected_indices = nn.kneighbors(y, 5, return_distance=False)

        np.testing.assert_array_equal(indices, expected_indices)
