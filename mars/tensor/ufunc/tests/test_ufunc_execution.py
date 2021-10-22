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

from .... import tensor as mt
from ...core import Tensor


@pytest.mark.parametrize("ufunc_name", ["negative"])
def test_unary_ufunc(setup, ufunc_name):
    raw_data = np.random.rand(100, 100)
    t = mt.tensor(raw_data.copy(), chunk_size=20)

    ufunc_obj = getattr(np, ufunc_name)

    res = ufunc_obj(t)
    expected = ufunc_obj(raw_data)
    assert isinstance(res, Tensor)
    np.testing.assert_array_equal(res.execute().fetch(), expected)

    ufunc_obj.at(t, 3)
    ufunc_obj.at(raw_data, 3)
    np.testing.assert_array_equal(t.execute().fetch(), raw_data)


@pytest.mark.parametrize("ufunc_name", ["add", "multiply", "logaddexp", "logaddexp2"])
def test_binary_ufunc(setup, ufunc_name):
    raw_data1 = np.random.rand(100, 100)
    t1 = mt.tensor(raw_data1.copy(), chunk_size=50)
    raw_data2 = np.random.rand(100, 100)
    t2 = mt.tensor(raw_data2.copy(), chunk_size=50)

    ufunc_obj = getattr(np, ufunc_name)

    res = ufunc_obj(t1, t2)
    expected = ufunc_obj(raw_data1, raw_data2)
    assert isinstance(res, Tensor)
    np.testing.assert_array_equal(res.execute().fetch(), expected)

    ufunc_obj.at(t1, (3, 4), 2)
    ufunc_obj.at(raw_data1, (3, 4), 2)
    np.testing.assert_array_equal(t1.execute().fetch(), raw_data1)

    res = ufunc_obj.reduce(t1, axis=1)
    expected = ufunc_obj.reduce(raw_data1, axis=1)
    assert isinstance(res, Tensor)
    np.testing.assert_almost_equal(res.execute().fetch(), expected)

    res = t1.copy()
    ufunc_obj.reduce(t1, axis=1, out=res)
    expected = ufunc_obj.reduce(raw_data1, axis=1)
    assert isinstance(res, Tensor)
    np.testing.assert_almost_equal(res.execute().fetch(), expected)

    res = ufunc_obj.accumulate(t1, axis=1)
    expected = ufunc_obj.accumulate(raw_data1, axis=1)
    assert isinstance(res, Tensor)
    np.testing.assert_almost_equal(res.execute().fetch(), expected)

    res = t1.copy()
    ufunc_obj.accumulate(t1, axis=1, out=res)
    expected = ufunc_obj.accumulate(raw_data1, axis=1)
    assert isinstance(res, Tensor)
    np.testing.assert_almost_equal(res.execute().fetch(), expected)

    with pytest.raises(TypeError):
        ufunc_obj.reduceat(t1, [(3, 4)])
