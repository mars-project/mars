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

from ....core import tile
from ...datasource import tensor, array
from .. import digitize, histogram_bin_edges, quantile, percentile
from ..quantile import INTERPOLATION_TYPES


def test_digitize():
    x = tensor(np.array([0.2, 6.4, 3.0, 1.6]), chunk_size=2)
    bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    inds = digitize(x, bins)

    assert inds.shape == (4,)
    assert inds.dtype is not None

    inds = tile(inds)

    assert len(inds.chunks) == 2


def test_histogram_bin_edges():
    a = array([0, 0, 0, 1, 2, 3, 3, 4, 5], chunk_size=3)

    with pytest.raises(ValueError):
        histogram_bin_edges(a, bins="unknown")

    with pytest.raises(TypeError):
        # bins is str, weights cannot be provided
        histogram_bin_edges(a, bins="scott", weights=a)

    with pytest.raises(ValueError):
        histogram_bin_edges(a, bins=-1)

    with pytest.raises(ValueError):
        # not asc
        histogram_bin_edges(a, bins=[3, 2, 1])

    with pytest.raises(ValueError):
        # bins cannot be 2d
        histogram_bin_edges(a, bins=np.random.rand(2, 3))

    with pytest.raises(ValueError):
        histogram_bin_edges(a, range=(5, 0))

    with pytest.raises(ValueError):
        histogram_bin_edges(a, range=(np.nan, np.nan))

    bins = histogram_bin_edges(a, bins=3, range=(0, 5))
    # if range specified, no error will occur
    tile(bins)


def test_quantile():
    raw = np.random.rand(100)
    q = np.random.rand(10)

    for dtype in [np.float32, np.int64, np.complex128]:
        raw2 = raw.astype(dtype)
        a = tensor(raw2, chunk_size=100)

        b = quantile(a, q, overwrite_input=True)
        assert b.shape == (10,)
        assert b.dtype == np.quantile(raw2, q).dtype

        b = tile(b)
        assert len(b.chunks) == 1

    raw = np.random.rand(20, 10)
    q = np.random.rand(10)

    for dtype in [np.float32, np.int64, np.complex128]:
        for axis in (None, 0, 1, [0, 1]):
            for interpolation in INTERPOLATION_TYPES:
                for keepdims in [True, False]:
                    raw2 = raw.astype(dtype)
                    a = tensor(raw2, chunk_size=(8, 6))

                    b = quantile(
                        a, q, axis=axis, interpolation=interpolation, keepdims=keepdims
                    )
                    expected = np.quantile(
                        raw2,
                        q,
                        axis=axis,
                        interpolation=interpolation,
                        keepdims=keepdims,
                    )
                    assert b.shape == expected.shape
                    assert b.dtype == expected.dtype

    a = tensor(raw, chunk_size=10)
    b = quantile(a, q)

    b = tile(b)
    assert b.shape == (10,)

    b = quantile(a, 0.3)
    assert b.ndim == 0

    raw2 = np.random.rand(3, 4, 5)
    a2 = tensor(raw2, chunk_size=3)
    b2 = quantile(a2, q, axis=(0, 2))
    expected = np.quantile(raw2, q, axis=(0, 2))
    assert b2.shape == expected.shape

    b2 = tile(b2)
    assert b2.shape == expected.shape

    # q has to be 1-d
    with pytest.raises(ValueError):
        quantile(a, q.reshape(5, 2))

    # wrong out type
    with pytest.raises(TypeError):
        quantile(a, q, out=2)

    # wrong q
    with pytest.raises(ValueError):
        q2 = q.copy()
        q2[0] = 1.1
        quantile(a, q2)

    # wrong q, with size < 10
    with pytest.raises(ValueError):
        q2 = np.random.rand(5)
        q2[0] = 1.1
        quantile(a, q2)

    # wrong interpolation
    with pytest.raises(ValueError):
        quantile(a, q, interpolation="unknown")


def test_percentile():
    raw = np.random.rand(100)
    q = [101]

    a = tensor(raw, chunk_size=100)

    with pytest.raises(ValueError) as cm:
        percentile(a, q)
    the_exception = cm.value.args[0]
    assert "Percentiles" in the_exception
