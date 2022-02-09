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

from .....core import tile
from ....datasource import tensor
from ... import distance


@pytest.mark.skipif(distance.pdist is None, reason="scipy not installed")
def test_pdist_execution(setup):
    from scipy.spatial.distance import pdist as sp_pdist

    raw = np.random.rand(100, 10)

    # test 1 chunk
    x = tensor(raw, chunk_size=100)

    dist = distance.pdist(x)
    result = dist.execute().fetch()
    expected = sp_pdist(raw)
    np.testing.assert_array_equal(result, expected)

    dist = distance.pdist(x, metric="hamming")
    result = dist.execute().fetch()
    expected = sp_pdist(raw, metric="hamming")
    np.testing.assert_array_equal(result, expected)

    f = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    dist = distance.pdist(x, metric=f)
    result = dist.execute().fetch()
    expected = sp_pdist(raw, metric=f)
    np.testing.assert_array_equal(result, expected)

    # test more than 1 chunk
    x = tensor(raw, chunk_size=12)

    dist = distance.pdist(x)
    tdist = tile(dist)
    assert len(tdist.chunks) == 1
    result = dist.execute().fetch()
    expected = sp_pdist(raw)
    np.testing.assert_array_equal(result, expected)

    dist = distance.pdist(x, aggregate_size=3)
    tdist = tile(dist)
    assert len(tdist.chunks) == 3
    result = dist.execute().fetch()
    expected = sp_pdist(raw)
    np.testing.assert_array_equal(result, expected)

    dist = distance.pdist(x, metric="hamming", aggregate_size=2)
    tdist = tile(dist)
    assert len(tdist.chunks) == 2
    result = dist.execute().fetch()
    expected = sp_pdist(raw, metric="hamming")
    np.testing.assert_array_equal(result, expected)

    f = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    dist = distance.pdist(x, metric=f, aggregate_size=2)
    result = dist.execute().fetch()
    expected = sp_pdist(raw, metric=f)
    np.testing.assert_array_equal(result, expected)

    for x in [tensor(raw), tensor(raw, chunk_size=12)]:
        # test w
        weight = np.random.rand(10)
        w = tensor(weight, chunk_size=7)
        dist = distance.pdist(x, metric="wminkowski", p=3, w=w)
        result = dist.execute().fetch()
        expected = sp_pdist(raw, metric="minkowski", p=3, w=weight)
        np.testing.assert_array_equal(result, expected)

        # test V
        v = np.random.rand(10)
        V = tensor(v, chunk_size=7)
        dist = distance.pdist(x, metric="seuclidean", V=V)
        result = dist.execute().fetch()
        expected = sp_pdist(raw, metric="seuclidean", V=v)
        np.testing.assert_array_equal(result, expected)

        # test VI
        vi = np.random.rand(10, 10)
        VI = tensor(vi, chunk_size=8)
        dist = distance.pdist(x, metric="mahalanobis", VI=VI)
        result = dist.execute().fetch()
        expected = sp_pdist(raw, metric="mahalanobis", VI=vi)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(distance.cdist is None, reason="scipy not installed")
def test_cdist_execution(setup):
    from scipy.spatial.distance import cdist as sp_cdist

    raw_a = np.random.rand(100, 10)
    raw_b = np.random.rand(89, 10)

    # test 1 chunk
    xa = tensor(raw_a, chunk_size=100)
    xb = tensor(raw_b, chunk_size=100)

    dist = distance.cdist(xa, xb)
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b)
    np.testing.assert_array_equal(result, expected)

    dist = distance.cdist(xa, xb, metric="hamming")
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b, metric="hamming")
    np.testing.assert_array_equal(result, expected)

    f = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    dist = distance.cdist(xa, xb, metric=f)
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b, metric=f)
    np.testing.assert_array_equal(result, expected)

    # test more than 1 chunk
    xa = tensor(raw_a, chunk_size=12)
    xb = tensor(raw_b, chunk_size=13)

    dist = distance.cdist(xa, xb)
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b)
    np.testing.assert_array_equal(result, expected)

    dist = distance.cdist(xa, xb, metric="hamming")
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b, metric="hamming")
    np.testing.assert_array_equal(result, expected)

    f = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    dist = distance.cdist(xa, xb, metric=f)
    result = dist.execute().fetch()
    expected = sp_cdist(raw_a, raw_b, metric=f)
    np.testing.assert_array_equal(result, expected)

    for xa, xb in [
        (tensor(raw_a), tensor(raw_b)),
        (tensor(raw_a, chunk_size=12), tensor(raw_b, chunk_size=13)),
    ]:
        # test w
        weight = np.random.rand(10)
        w = tensor(weight, chunk_size=7)
        dist = distance.cdist(xa, xb, metric="wminkowski", p=3, w=w)
        result = dist.execute().fetch()
        expected = sp_cdist(raw_a, raw_b, metric="minkowski", p=3, w=weight)
        np.testing.assert_array_equal(result, expected)

        # test V
        v = np.random.rand(10)
        V = tensor(v, chunk_size=7)
        dist = distance.cdist(xa, xb, metric="seuclidean", V=V)
        result = dist.execute().fetch()
        expected = sp_cdist(raw_a, raw_b, metric="seuclidean", V=v)
        np.testing.assert_array_equal(result, expected)

        # test VI
        vi = np.random.rand(10, 10)
        VI = tensor(vi, chunk_size=8)
        dist = distance.cdist(xa, xb, metric="mahalanobis", VI=VI)
        result = dist.execute().fetch()
        expected = sp_cdist(raw_a, raw_b, metric="mahalanobis", VI=vi)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(distance.cdist is None, reason="scipy not installed")
def test_squareform_execution(setup):
    from scipy.spatial.distance import pdist as sp_pdist, squareform as sp_squareform

    raw_a = np.random.rand(80, 10)
    raw_pdsit = sp_pdist(raw_a)
    raw_square = sp_squareform(raw_pdsit)

    # tomatrix, test 1 chunk
    vec = tensor(raw_pdsit, chunk_size=raw_pdsit.shape[0])
    mat = distance.squareform(vec, chunk_size=100)
    result = mat.execute().fetch()
    np.testing.assert_array_equal(result, raw_square)

    # tomatrix, test more than 1 chunk
    vec = tensor(raw_pdsit, chunk_size=33)
    assert len(tile(vec).chunks) > 1
    mat = distance.squareform(vec, chunk_size=34)
    result = mat.execute().fetch()
    np.testing.assert_array_equal(result, raw_square)

    # tovec, test 1 chunk
    mat = tensor(raw_square)
    vec = distance.squareform(mat, chunk_size=raw_pdsit.shape[0])
    assert len(tile(mat).chunks) == 1
    assert len(tile(vec).chunks) == 1
    result = vec.execute().fetch()
    np.testing.assert_array_equal(result, raw_pdsit)

    # tovec, test more than 1 chunk
    mat = tensor(raw_square, chunk_size=31)
    vec = distance.squareform(mat, chunk_size=40)
    assert len(tile(vec).chunks) > 1
    result = vec.execute().fetch()
    np.testing.assert_array_equal(result, raw_pdsit)

    # test checks
    # generate non-symmetric matrix
    non_sym_arr = np.random.RandomState(0).rand(10, 10)

    # 1 chunk
    mat = tensor(non_sym_arr)
    vec = distance.squareform(mat, checks=True, chunk_size=100)
    with pytest.raises(ValueError):
        _ = vec.execute().fetch()
    # force checks=False
    vec = distance.squareform(mat, checks=False, chunk_size=100)
    _ = vec.execute().fetch()

    # more than 1 chunk
    mat = tensor(non_sym_arr, chunk_size=6)
    vec = distance.squareform(mat, checks=True, chunk_size=8)
    assert len(tile(vec).chunks) > 1
    with pytest.raises(ValueError):
        _ = vec.execute().fetch()
    # force checks=False
    vec = distance.squareform(mat, checks=False, chunk_size=100)
    _ = vec.execute().fetch()
