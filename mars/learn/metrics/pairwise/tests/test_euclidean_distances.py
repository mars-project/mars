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
import scipy.sparse as sps
import pytest
from sklearn.metrics import euclidean_distances as sk_euclidean_distances

from ..... import tensor as mt
from .....config import option_context
from .....lib.sparse import SparseNDArray
from ....utils import check_array
from ... import euclidean_distances


def test_euclidean_distances_op():
    x = mt.random.rand(10, 3)
    xx = mt.random.rand(1, 10)
    y = mt.random.rand(11, 3)

    d = euclidean_distances(x, X_norm_squared=xx)
    assert d.op.x_norm_squared.key == check_array(xx).T.key

    d = euclidean_distances(
        x,
        y,
        X_norm_squared=mt.random.rand(10, 1, dtype=mt.float32),
        Y_norm_squared=mt.random.rand(1, 11, dtype=mt.float32),
    )
    assert d.op.x_norm_squared is None
    assert d.op.y_norm_squared is None

    # XX shape incompatible
    with pytest.raises(ValueError):
        euclidean_distances(x, X_norm_squared=mt.random.rand(10))

    # XX shape incompatible
    with pytest.raises(ValueError):
        euclidean_distances(x, X_norm_squared=mt.random.rand(11, 1))

    # YY shape incompatible
    with pytest.raises(ValueError):
        euclidean_distances(x, y, Y_norm_squared=mt.random.rand(10))


def test_euclidean_distances_execution(setup):
    dense_raw_x = np.random.rand(30, 10)
    dense_raw_y = np.random.rand(40, 10)
    sparse_raw_x = SparseNDArray(sps.random(30, 10, density=0.5, format="csr"))
    sparse_raw_y = SparseNDArray(sps.random(40, 10, density=0.5, format="csr"))

    for raw_x, raw_y in [(dense_raw_x, dense_raw_y), (sparse_raw_x, sparse_raw_y)]:
        x = mt.tensor(raw_x, chunk_size=9)
        y = mt.tensor(raw_y, chunk_size=7)

        distance = euclidean_distances(x, y)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(raw_x, Y=raw_y)
        np.testing.assert_almost_equal(result, expected)

        x_norm = x.sum(axis=1)[..., np.newaxis]
        y_norm = y.sum(axis=1)[np.newaxis, ...]
        distance = euclidean_distances(
            x, y, X_norm_squared=x_norm, Y_norm_squared=y_norm
        )
        x_raw_norm = raw_x.sum(axis=1)[..., np.newaxis]
        y_raw_norm = raw_y.sum(axis=1)[np.newaxis, ...]

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(
            raw_x, raw_y, X_norm_squared=x_raw_norm, Y_norm_squared=y_raw_norm
        )
        np.testing.assert_almost_equal(result, expected)

        x_sq = (x**2).astype(np.float32)
        y_sq = (y**2).astype(np.float32)

        distance = euclidean_distances(x_sq, y_sq, squared=True)

        x_raw_sq = (raw_x**2).astype(np.float32)
        y_raw_sq = (raw_y**2).astype(np.float32)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(x_raw_sq, y_raw_sq, squared=True)
        np.testing.assert_almost_equal(result, expected, decimal=6)

        # test x is y
        distance = euclidean_distances(x)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(raw_x)

        np.testing.assert_almost_equal(result, expected)

    # test size adjust
    raw1 = np.random.rand(12, 4)
    raw2 = np.random.rand(18, 4)

    t1 = mt.tensor(raw1, chunk_size=4)
    t2 = mt.tensor(raw2, chunk_size=6)
    with option_context({"chunk_store_limit": 80}):
        distance = euclidean_distances(t1, t2)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(raw1, raw2)
        np.testing.assert_almost_equal(result, expected)

        distance = euclidean_distances(t2, t1)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(raw2, raw1)
        np.testing.assert_almost_equal(result, expected)

    with option_context({"chunk_store_limit": 20}):
        distance = euclidean_distances(t1, t2)

        result = distance.execute().fetch()
        expected = sk_euclidean_distances(raw1, raw2)
        np.testing.assert_almost_equal(result, expected)
