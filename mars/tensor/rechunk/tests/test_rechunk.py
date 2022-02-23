#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .... import tensor as mt

# dense
raw = np.random.RandomState(0).rand(12, 9)
raw2 = raw.copy()
raw2.ravel()[::2] = 0
# dense, F-order
raw3 = np.asfortranarray(raw)
# sparse
raw_s = sps.csr_matrix(raw2)


@pytest.mark.parametrize("data", [raw, raw3, raw_s])
@pytest.mark.parametrize("chunk_size", [3, (12, 9), (4, 8)])
def test_rechunk_execute(setup, data, chunk_size):
    tensor = mt.tensor(data, chunk_size=4)
    new_tensor = tensor.rechunk(chunk_size)
    result = new_tensor.execute().fetch()
    if hasattr(result, "toarray"):
        # sparse
        result = result.toarray()
        data = data.toarray()
    assert result.flags["C_CONTIGUOUS"] == data.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(result, data)
