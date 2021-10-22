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
from sklearn.metrics.pairwise import manhattan_distances as sk_manhattan_distances

from ..... import tensor as mt
from .. import manhattan_distances


def test_manhattan_distances():
    x = mt.random.randint(10, size=(10, 3), density=0.4)
    y = mt.random.randint(10, size=(11, 3), density=0.5)

    with pytest.raises(TypeError):
        manhattan_distances(x, y, sum_over_features=False)

    x = x.todense()
    y = y.todense()

    d = manhattan_distances(x, y, sum_over_features=True)
    assert d.shape == (10, 11)
    d = manhattan_distances(x, y, sum_over_features=False)
    assert d.shape == (110, 3)


raw_x = np.random.rand(20, 5)
raw_y = np.random.rand(21, 5)

x1 = mt.tensor(raw_x, chunk_size=30)
y1 = mt.tensor(raw_y, chunk_size=30)

x2 = mt.tensor(raw_x, chunk_size=11)
y2 = mt.tensor(raw_y, chunk_size=12)

raw_sparse_x = sps.random(20, 5, density=0.4, format="csr", random_state=0)
raw_sparse_y = sps.random(21, 5, density=0.3, format="csr", random_state=0)

x3 = mt.tensor(raw_sparse_x, chunk_size=30)
y3 = mt.tensor(raw_sparse_y, chunk_size=30)

x4 = mt.tensor(raw_sparse_x, chunk_size=11)
y4 = mt.tensor(raw_sparse_y, chunk_size=12)


@pytest.mark.parametrize(
    "x, y, is_sparse",
    [(x1, y1, False), (x2, y2, False), (x3, y3, True), (x4, y4, True)],
)
def test_manhattan_distances_execution(setup, x, y, is_sparse):
    if is_sparse:
        rx, ry = raw_sparse_x, raw_sparse_y
    else:
        rx, ry = raw_x, raw_y

    sv = [True, False] if not is_sparse else [True]

    for sum_over_features in sv:
        d = manhattan_distances(x, y, sum_over_features)

        result = d.execute().fetch()
        expected = sk_manhattan_distances(rx, ry, sum_over_features=sum_over_features)

        np.testing.assert_almost_equal(result, expected)

        d = manhattan_distances(x, sum_over_features=sum_over_features)

        result = d.execute().fetch()
        expected = sk_manhattan_distances(rx, sum_over_features=sum_over_features)

        np.testing.assert_almost_equal(result, expected)
