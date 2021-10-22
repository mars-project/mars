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

from ... import einsum
from ...datasource import tensor


def test_einsum_execution(setup):
    data1 = np.random.rand(3, 4, 5)
    data2 = np.random.rand(4, 3, 2)

    t1 = tensor(data1, chunk_size=2)
    t2 = tensor(data2, chunk_size=3)
    t = einsum("ijk, jil -> kl", t1, t2)
    res = t.execute().fetch()
    expected = np.einsum("ijk, jil -> kl", data1, data2)
    np.testing.assert_almost_equal(res, expected)

    # dot
    t = einsum("ijk, jil", t1, t2, optimize=True)
    res = t.execute().fetch()
    expected = np.einsum("ijk, jil", data1, data2, optimize=True)
    np.testing.assert_almost_equal(res, expected)

    # multiply(data1, data2)
    data1 = np.random.rand(6, 6)
    data2 = np.random.rand(6, 6)
    t1 = tensor(data1, chunk_size=3)
    t2 = tensor(data2, chunk_size=3)
    t = einsum("..., ...", t1, t2, order="C")
    res = t.execute().fetch()
    expected = np.einsum("..., ...", data1, data2, order="C")
    np.testing.assert_almost_equal(res, expected)

    # sum(data, axis=-1)
    data = np.random.rand(10)
    t1 = tensor(data, chunk_size=3)
    t = einsum("i->", t1, order="F")
    res = t.execute().fetch()
    expected = np.einsum("i->", data, order="F")
    np.testing.assert_almost_equal(res, expected)

    # sum(data, axis=0)
    t1 = tensor(data)
    t = einsum("...i->...", t1)
    res = t.execute().fetch()
    expected = np.einsum("...i->...", data)
    np.testing.assert_almost_equal(res, expected)

    # test broadcast
    data1 = np.random.rand(1, 10, 9)
    data2 = np.random.rand(9, 6)
    data3 = np.random.rand(10, 6)
    data4 = np.random.rand(8)

    t1 = tensor(data1, chunk_size=(1, (5, 5), (3, 3, 3)))
    t2 = tensor(data2, chunk_size=((3, 3, 3), (3, 3)))
    t3 = tensor(data3, chunk_size=((6, 4), (4, 2)))
    t4 = tensor(data4, chunk_size=4)
    t = einsum("ajk,kl,jl,a->a", t1, t2, t3, t4, optimize="optimal")
    res = t.execute().fetch()
    expected = np.einsum(
        "ajk,kl,jl,a->a", data1, data2, data3, data4, optimize="optimal"
    )
    np.testing.assert_almost_equal(res, expected)

    t = einsum("ajk,kl,jl,a->a", t1, t2, t3, t4, optimize="greedy")
    res = t.execute().fetch()
    expected = np.einsum(
        "ajk,kl,jl,a->a", data1, data2, data3, data4, optimize="greedy"
    )
    np.testing.assert_almost_equal(res, expected)
