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
from sklearn.metrics import pairwise_distances as sk_pairwise_distances
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
from sklearn.exceptions import DataConversionWarning


from ..... import tensor as mt
from .....session import execute, fetch
from ... import pairwise_distances, pairwise_distances_topk


def test_pairwise_distances_execution(setup):
    raw_x = np.random.rand(20, 5)
    raw_y = np.random.rand(21, 5)

    x = mt.tensor(raw_x, chunk_size=11)
    y = mt.tensor(raw_y, chunk_size=12)

    d = pairwise_distances(x, y)
    result = d.execute().fetch()
    expected = sk_pairwise_distances(raw_x, raw_y)
    np.testing.assert_almost_equal(result, expected)

    # test precomputed
    d2 = d.copy()
    d2[0, 0] = -1
    d2 = pairwise_distances(d2, y, metric="precomputed")
    with pytest.raises(ValueError):
        _ = d2.execute().fetch()

    # test cdist
    weight = np.random.rand(5)
    d = pairwise_distances(x, y, metric="wminkowski", p=3, w=weight)
    result = d.execute().fetch()
    expected = sk_pairwise_distances(raw_x, raw_y, metric="minkowski", p=3, w=weight)
    np.testing.assert_almost_equal(result, expected)

    # test pdist
    d = pairwise_distances(x, metric="hamming")
    result = d.execute().fetch()
    expected = sk_pairwise_distances(raw_x, metric="hamming")
    np.testing.assert_almost_equal(result, expected)

    # test function metric
    m = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    d = pairwise_distances(x, y, metric=m)
    result = d.execute().fetch()
    expected = sk_pairwise_distances(raw_x, raw_y, metric=m)
    np.testing.assert_almost_equal(result, expected)

    with pytest.warns(DataConversionWarning):
        pairwise_distances(x, y, metric="jaccard")

    with pytest.raises(ValueError):
        _ = pairwise_distances(x, y, metric="unknown")


def test_pairwise_distances_topk_execution(setup):
    rs = np.random.RandomState(0)
    raw_x = rs.rand(20, 5)
    raw_y = rs.rand(21, 5)

    x = mt.tensor(raw_x, chunk_size=11)
    y = mt.tensor(raw_y, chunk_size=12)

    d, i = pairwise_distances_topk(x, y, 3, metric="euclidean", return_index=True)
    result = fetch(*execute(d, i))
    nn = SkNearestNeighbors(n_neighbors=3, algorithm="brute", metric="euclidean")
    nn.fit(raw_y)
    expected = nn.kneighbors(raw_x, return_distance=True)
    np.testing.assert_almost_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])

    x = mt.tensor(raw_x, chunk_size=(11, 3))

    d = pairwise_distances_topk(x, k=4, metric="euclidean", return_index=False)
    result = d.execute().fetch()
    nn = SkNearestNeighbors(n_neighbors=3, algorithm="brute", metric="euclidean")
    nn.fit(raw_x)
    expected = nn.kneighbors(return_distance=True)[0]
    np.testing.assert_almost_equal(result[:, 1:], expected)

    y = mt.tensor(raw_y, chunk_size=21)

    d, i = pairwise_distances_topk(
        x, y, 3, metric="cosine", return_index=True, working_memory="168"
    )
    result = fetch(*execute(d, i))
    nn = SkNearestNeighbors(n_neighbors=3, algorithm="brute", metric="cosine")
    nn.fit(raw_y)
    expected = nn.kneighbors(raw_x, return_distance=True)
    np.testing.assert_almost_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])

    d = pairwise_distances_topk(x, y, 3, metric="cosine", axis=0, return_index=False)
    result = d.execute().fetch()
    nn = SkNearestNeighbors(n_neighbors=3, algorithm="brute", metric="cosine")
    nn.fit(raw_x)
    expected = nn.kneighbors(raw_y, return_distance=True)[0]
    np.testing.assert_almost_equal(result, expected)
