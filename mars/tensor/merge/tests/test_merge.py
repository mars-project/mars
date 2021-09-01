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

from ....core import tile
from ...datasource import ones, empty
from .. import concatenate, stack


def test_concatenate():
    a = ones((10, 20, 30), chunk_size=10)
    b = ones((20, 20, 30), chunk_size=20)

    c = concatenate([a, b])
    assert c.shape == (30, 20, 30)

    a = ones((10, 20, 30), chunk_size=10)
    b = ones((10, 20, 40), chunk_size=20)

    c = concatenate([a, b], axis=-1)
    assert c.shape == (10, 20, 70)

    with pytest.raises(ValueError):
        a = ones((10, 20, 30), chunk_size=10)
        b = ones((20, 30, 30), chunk_size=20)

        concatenate([a, b])

    with pytest.raises(ValueError):
        a = ones((10, 20, 30), chunk_size=10)
        b = ones((20, 20), chunk_size=20)

        concatenate([a, b])

    a = ones((10, 20, 30), chunk_size=5)
    b = ones((20, 20, 30), chunk_size=10)

    a, c = tile(a, concatenate([a, b]))
    assert c.chunk_shape[0] == 4
    assert c.chunk_shape[1] == 4
    assert c.chunk_shape[2] == 6
    assert c.nsplits == ((5, 5, 10, 10), (5,) * 4, (5,) * 6)
    assert c.cix[0, 0, 0].key == a.cix[0, 0, 0].key
    assert c.cix[1, 0, 0].key == a.cix[1, 0, 0].key


def test_stack():
    raw_arrs = [ones((3, 4), chunk_size=2) for _ in range(10)]
    arr2 = stack(raw_arrs, axis=0)

    assert arr2.shape == (10, 3, 4)

    arr2 = tile(arr2)
    assert arr2.nsplits == ((1,) * 10, (2, 1), (2, 2))

    arr3 = stack(raw_arrs, axis=1)

    assert arr3.shape == (3, 10, 4)

    arr3 = tile(arr3)
    assert arr3.nsplits == ((2, 1), (1,) * 10, (2, 2))

    arr4 = stack(raw_arrs, axis=2)

    assert arr4.shape == (3, 4, 10)

    arr4 = tile(arr4)
    assert arr4.nsplits == ((2, 1), (2, 2), (1,) * 10)

    with pytest.raises(ValueError):
        raw_arrs2 = [ones((3, 4), chunk_size=2), ones((4, 3), chunk_size=2)]
        stack(raw_arrs2)

    with pytest.raises(np.AxisError):
        stack(raw_arrs, axis=3)

    arr5 = tile(stack(raw_arrs, -1))
    assert arr5.nsplits == ((2, 1), (2, 2), (1,) * 10)

    arr6 = tile(stack(raw_arrs, -3))
    assert arr6.nsplits == ((1,) * 10, (2, 1), (2, 2))

    with pytest.raises(np.AxisError):
        stack(raw_arrs, axis=-4)

    with pytest.raises(TypeError):
        stack(raw_arrs, out=1)

    with pytest.raises(ValueError):
        stack(raw_arrs, empty((1, 10, 3, 4)))
