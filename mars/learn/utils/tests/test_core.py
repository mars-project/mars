# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from .... import tensor as mt
from ..core import sort_by


def test_sort_by(setup):
    rs = np.random.RandomState(0)
    raw1 = rs.rand(10)
    raw2 = rs.rand(10)
    raw3 = rs.rand(10)

    a1 = mt.tensor(raw1, chunk_size=4)
    a2 = mt.tensor(raw2, chunk_size=4)
    a3 = mt.tensor(raw3, chunk_size=4)

    s1, s2 = sort_by([a1, a2], by=a3)
    ind = np.argsort(raw3)
    e1, e2 = raw1[ind], raw2[ind]
    np.testing.assert_array_equal(s1, e1)
    np.testing.assert_array_equal(s2, e2)

    s1, s2 = sort_by([a1, a2], by=a2, ascending=False)
    ind = np.argsort(raw2)[::-1]
    e1, e2 = raw1[ind], raw2[ind]
    np.testing.assert_array_equal(s1, e1)
    np.testing.assert_array_equal(s2, e2)
