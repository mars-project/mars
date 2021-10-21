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

import pytest
from numpy.testing import assert_array_almost_equal

from ..extmath import softmax


@pytest.mark.parametrize("copy", [True, False])
def test_softmax(setup, copy):
    x = [[1, 2, 3], [2, 3, 4]]
    ref = [[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]
    x_ = softmax(x, copy=copy)
    assert_array_almost_equal(ref, x_)
