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

from .... import tensor as mt
from ....core import tile
from ...lib import nd_grid


def test_index_tricks():
    mgrid = nd_grid()
    g = mgrid[0:5, 0:5]
    tile(g)  # tileable means no loop exists

    ogrid = nd_grid(sparse=True)
    o = ogrid[0:5, 0:5]
    tile(*o)  # tilesable means no loop exists


def test_r_(setup):
    r = mt.r_[mt.array([1, 2, 3]), 0, 0, mt.array([4, 5, 6])]

    result = r.execute().fetch()
    expected = np.r_[np.array([1, 2, 3]), 0, 0, np.array([4, 5, 6])]

    np.testing.assert_array_equal(result, expected)

    r = mt.r_[-1:1:6j, [0] * 3, 5, 6]

    result = r.execute().fetch()
    expected = np.r_[-1:1:6j, [0] * 3, 5, 6]

    np.testing.assert_array_equal(result, expected)

    r = mt.r_[-1:1:6j]

    result = r.execute().fetch()
    expected = np.r_[-1:1:6j]

    np.testing.assert_array_equal(result, expected)

    raw = [[0, 1, 2], [3, 4, 5]]
    a = mt.array(raw, chunk_size=2)
    r = mt.r_["-1", a, a]

    result = r.execute().fetch()
    expected = np.r_["-1", raw, raw]

    np.testing.assert_array_equal(result, expected)

    r = mt.r_["0,2", [1, 2, 3], [4, 5, 6]]

    result = r.execute().fetch()
    expected = np.r_["0,2", [1, 2, 3], [4, 5, 6]]

    np.testing.assert_array_equal(result, expected)

    r = mt.r_["0,2,0", [1, 2, 3], [4, 5, 6]]

    result = r.execute().fetch()
    expected = np.r_["0,2,0", [1, 2, 3], [4, 5, 6]]
    np.testing.assert_array_equal(result, expected)

    r = mt.r_["1,2,0", [1, 2, 3], [4, 5, 6]]

    result = r.execute().fetch()
    expected = np.r_["1,2,0", [1, 2, 3], [4, 5, 6]]
    np.testing.assert_array_equal(result, expected)

    assert len(mt.r_) == 0

    with pytest.raises(ValueError):
        _ = mt.r_[:3, "wrong"]


def test_c_(setup):
    r = mt.c_[mt.array([1, 2, 3]), mt.array([4, 5, 6])]

    result = r.execute().fetch()
    expected = np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
    np.testing.assert_array_equal(result, expected)

    r = mt.c_[mt.array([[1, 2, 3]]), 0, 0, mt.array([[4, 5, 6]])]

    result = r.execute().fetch()
    expected = np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
    np.testing.assert_array_equal(result, expected)

    r = mt.c_[:3, 1:4]
    result = r.execute().fetch()
    expected = np.c_[:3, 1:4]
    np.testing.assert_array_equal(result, expected)
