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
import scipy.sparse as sps

from .....core import tile
from ....datasource import tensor
from ... import distance


def test_pdist():
    raw = np.random.rand(100, 10)

    # test 1 chunk
    a = tensor(raw, chunk_size=100)
    dist = distance.pdist(a)
    assert dist.shape == (100 * 99 // 2,)

    dist = tile(dist)
    assert len(dist.chunks) == 1
    for c in dist.chunks:
        assert c.shape == (dist.shape[0],)

    # test multiple chunks
    a = tensor(raw, chunk_size=15)
    dist = distance.pdist(a, aggregate_size=2)
    assert dist.shape == (100 * 99 // 2,)

    dist = tile(dist)
    assert len(dist.chunks) == 2
    for c in dist.chunks:
        assert c.shape == (dist.shape[0] // 2,)

    # X cannot be sparse
    raw = sps.csr_matrix(np.zeros((4, 3)))
    a = tensor(raw)
    with pytest.raises(ValueError):
        distance.pdist(a)

    # X can only be 2-d
    with pytest.raises(ValueError):
        distance.pdist(np.random.rand(3, 3, 3))

    # out type wrong
    with pytest.raises(TypeError):
        distance.pdist(np.random.rand(3, 3), out=2)

    # out shape wrong
    with pytest.raises(ValueError):
        distance.pdist(np.random.rand(3, 3), out=tensor(np.random.rand(2)))

    # out dtype wrong
    with pytest.raises(ValueError):
        distance.pdist(
            np.random.rand(3, 3), out=tensor(np.random.randint(2, size=(3,)))
        )

    # test extra param
    with pytest.raises(TypeError):
        distance.pdist(np.random.rand(3, 3), unknown_kw="unknown_kw")


def test_cdist():
    raw_a = np.random.rand(100, 10)
    raw_b = np.random.rand(90, 10)

    # test 1 chunk
    a = tensor(raw_a, chunk_size=100)
    b = tensor(raw_b, chunk_size=100)
    dist = distance.cdist(a, b)
    assert dist.shape == (100, 90)

    dist = tile(dist)
    assert len(dist.chunks) == 1
    for c in dist.chunks:
        assert c.shape == dist.shape

    # test multiple chunks
    a = tensor(raw_a, chunk_size=15)
    b = tensor(raw_b, chunk_size=16)
    dist = distance.cdist(a, b)
    assert dist.shape == (100, 90)

    ta, tb, dist = tile(a, b, dist)
    assert len(dist.chunks) == (100 // 15 + 1) * (90 // 16 + 1)
    assert dist.nsplits == (ta.nsplits[0], tb.nsplits[0])
    for c in dist.chunks:
        assert c.shape == (
            ta.cix[c.index[0], 0].shape[0],
            tb.cix[c.index[1], 0].shape[0],
        )

    # XA can only be 2-d
    with pytest.raises(ValueError):
        distance.cdist(np.random.rand(3, 3, 3), np.random.rand(3, 3))

    # XB can only be 2-d
    with pytest.raises(ValueError):
        distance.cdist(np.random.rand(3, 3), np.random.rand(3, 3, 3))

    # XA cannot be sparse
    raw = sps.csr_matrix(np.zeros((4, 3)))
    a = tensor(raw)
    with pytest.raises(ValueError):
        distance.cdist(a, np.random.rand(10, 3))

    # XB cannot be sparse
    raw = sps.csr_matrix(np.zeros((4, 3)))
    b = tensor(raw)
    with pytest.raises(ValueError):
        distance.cdist(np.random.rand(10, 3), b)

    # out type wrong
    with pytest.raises(TypeError):
        distance.cdist(raw_a, raw_b, out=2)

    # out shape wrong
    with pytest.raises(ValueError):
        distance.cdist(raw_a, raw_b, out=tensor(np.random.rand(100, 91)))

    # out dtype wrong
    with pytest.raises(ValueError):
        distance.cdist(raw_a, raw_b, out=tensor(np.random.randint(2, size=(100, 90))))

    # test extra param
    with pytest.raises(TypeError):
        distance.cdist(raw_a, raw_b, unknown_kw="unknown_kw")


def test_squareform():
    assert distance.squareform(np.array([], dtype=float)).shape == (1, 1)
    assert distance.squareform(np.atleast_2d(np.random.rand())).shape == (0,)

    with pytest.raises(ValueError):
        distance.squareform(np.random.rand(3, 3), force="tomatrix")

    with pytest.raises(ValueError):
        distance.squareform(np.random.rand(3), force="tovector")

    with pytest.raises(ValueError):
        distance.squareform(np.random.rand(3, 3, 3))

    with pytest.raises(ValueError):
        distance.squareform(np.random.rand(2, 4))

    with pytest.raises(ValueError):
        distance.squareform(np.random.rand(7))
