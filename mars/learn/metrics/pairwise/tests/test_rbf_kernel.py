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
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

from .. import rbf_kernel


def test_rbf_kernel(setup):
    rs = np.random.RandomState(0)
    raw_X = rs.rand(10, 4)
    raw_Y = rs.rand(11, 4)

    r = rbf_kernel(raw_X, raw_Y)
    result = r.to_numpy()
    expected = sklearn_rbf_kernel(raw_X, raw_Y)

    np.testing.assert_almost_equal(result, expected)
