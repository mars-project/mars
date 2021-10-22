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

from ... import tensor as mt
from ...core import tile


def test_params():
    raw = np.random.rand(10, 10)
    a = mt.tensor(raw)
    a = a[a[0] < 0.5]
    a = tile(a)
    c = a.chunks[0]

    assert any(np.isnan(s) for s in c.params["shape"])
    c.params = c.get_params_from_data(raw[raw[0] < 0.5])
    assert not any(np.isnan(s) for s in c.params["shape"])

    params = c.params.copy()
    params.pop("index", None)
    a.params = params
    assert np.prod(a.shape) > 0
    a.refresh_params()
