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
from sklearn.metrics.pairwise import cosine_distances as sk_cosine_distances

from ..... import tensor as mt
from .. import cosine_distances


raw_dense_x = np.random.rand(25, 10)
raw_dense_y = np.random.rand(17, 10)

raw_sparse_x = sps.random(25, 10, density=0.5, format="csr", random_state=0)
raw_sparse_y = sps.random(17, 10, density=0.4, format="csr", random_state=1)

raw_x_ys = [(raw_dense_x, raw_dense_y), (raw_sparse_x, raw_sparse_y)]


@pytest.mark.parametrize("raw_x, raw_y", raw_x_ys)
@pytest.mark.parametrize("chunk_size", [25, 6])
def test_cosine_distances_execution(setup, raw_x, raw_y, chunk_size):
    x = mt.tensor(raw_x, chunk_size=chunk_size)
    y = mt.tensor(raw_y, chunk_size=chunk_size)

    d = cosine_distances(x, y)

    result = d.execute().fetch()
    expected = sk_cosine_distances(raw_x, raw_y)

    np.testing.assert_almost_equal(np.asarray(result), expected)

    d = cosine_distances(x)

    result = d.execute().fetch()
    expected = sk_cosine_distances(raw_x)

    np.testing.assert_almost_equal(np.asarray(result), expected)
