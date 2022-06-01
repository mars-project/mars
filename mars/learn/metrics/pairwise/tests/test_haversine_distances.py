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
from sklearn.metrics.pairwise import haversine_distances as sk_haversine_distances

from ..... import tensor as mt
from .. import haversine_distances


def test_haversine_distances_op():
    # shape[1] != 2
    with pytest.raises(ValueError):
        haversine_distances(mt.random.rand(10, 3))

    # shape[1] != 2
    with pytest.raises(ValueError):
        haversine_distances(mt.random.rand(10, 2), mt.random.rand(11, 3))

    # cannot support sparse
    with pytest.raises(TypeError):
        haversine_distances(mt.random.randint(10, size=(10, 2), density=0.5))


raw_x = np.random.rand(30, 2)
raw_y = np.random.rand(21, 2)

# one chunk
x1 = mt.tensor(raw_x, chunk_size=30)
y1 = mt.tensor(raw_y, chunk_size=30)

# multiple chunks
x2 = mt.tensor(raw_x, chunk_size=(11, 1))
y2 = mt.tensor(raw_y, chunk_size=(17, 1))


@pytest.mark.parametrize("x, y", [(x1, y1), (x2, y2)])
@pytest.mark.parametrize("use_sklearn", [True, False])
def test_haversine_distances_execution(setup, x, y, use_sklearn):
    distance = haversine_distances(x, y)
    distance.op._use_sklearn = use_sklearn

    result = distance.execute().fetch()
    expected = sk_haversine_distances(raw_x, raw_y)
    np.testing.assert_almost_equal(result, expected)

    # test x is y
    distance = haversine_distances(x)
    distance.op._use_sklearn = use_sklearn

    result = distance.execute().fetch()
    expected = sk_haversine_distances(raw_x, raw_x)
    np.testing.assert_almost_equal(result, expected)
