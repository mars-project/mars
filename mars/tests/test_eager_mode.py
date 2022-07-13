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
import pandas as pd
import pytest

from .. import tensor as mt
from .. import dataframe as md
from ..config import option_context
from ..dataframe.datasource.dataframe import from_pandas


def test_base_execute(setup):
    with option_context({"eager_mode": True}):
        a_data = np.random.rand(10, 10)
        a = mt.tensor(a_data, chunk_size=6)
        np.testing.assert_array_equal(a.fetch(), a_data)

        r1 = a + 1
        np.testing.assert_array_equal(r1.fetch(), a_data + 1)

        r2 = 2 * r1
        np.testing.assert_array_equal(r2.fetch(), (a_data + 1) * 2)

        # test add with out
        b = mt.ones((10, 10), chunk_size=6)
        np.testing.assert_array_equal(b.fetch(), np.ones((10, 10)))

        mt.add(a, b, out=b)
        np.testing.assert_array_equal(b.fetch(), a_data + 1)

        # test tensor dot
        c_data1 = np.random.rand(10, 10)
        c_data2 = np.random.rand(10, 10)
        c1 = mt.tensor(c_data1, chunk_size=6)
        c2 = mt.tensor(c_data2, chunk_size=6)
        r3 = c1.dot(c2)
        np.testing.assert_array_almost_equal(r3.fetch(), c_data1.dot(c_data2))


def test_multiple_output_execute(setup):
    with option_context({"eager_mode": True}):
        data = np.random.random((5, 9))

        arr1 = mt.tensor(data.copy(), chunk_size=3)
        result = mt.modf(arr1)
        expected = np.modf(data)

        np.testing.assert_array_equal(result[0].fetch(), expected[0])
        np.testing.assert_array_equal(result[1].fetch(), expected[1])

        arr3 = mt.tensor(data.copy(), chunk_size=3)
        result1, result2, result3 = mt.split(arr3, 3, axis=1)
        expected = np.split(data, 3, axis=1)

        np.testing.assert_array_equal(result1.fetch(), expected[0])
        np.testing.assert_array_equal(result2.fetch(), expected[1])
        np.testing.assert_array_equal(result3.fetch(), expected[2])


def test_mixed_config(setup):
    a = mt.ones((10, 10), chunk_size=6)
    with pytest.raises(ValueError):
        a.fetch()

    with option_context({"eager_mode": True}):
        b = mt.ones((10, 10), chunk_size=(6, 8))
        np.testing.assert_array_equal(b.fetch(), np.ones((10, 10)))

        r = b + 1
        np.testing.assert_array_equal(r.fetch(), np.ones((10, 10)) * 2)

        r2 = b.dot(b)
        np.testing.assert_array_equal(r2.fetch(), np.ones((10, 10)) * 10)

    c = mt.ones((10, 10), chunk_size=6)
    with pytest.raises(ValueError):
        c.fetch()
    np.testing.assert_array_equal(c.execute(), np.ones((10, 10)))

    r = c.dot(c)
    with pytest.raises(ValueError):
        r.fetch()
    np.testing.assert_array_equal(r.execute(), np.ones((10, 10)) * 10)


@pytest.mark.ray_dag
def test_index(setup):
    with option_context({"eager_mode": True}):
        a = mt.random.rand(10, 5, chunk_size=5)
        idx = slice(0, 5), slice(0, 5)
        a[idx] = 1
        np.testing.assert_array_equal(a.fetch()[idx], np.ones((5, 5)))

        split1, split2 = mt.split(a, 2)
        np.testing.assert_array_equal(split1.fetch(), np.ones((5, 5)))

        # test bool indexing
        a = mt.random.rand(8, 8, chunk_size=4)
        set_value = mt.ones((2, 2)) * 2
        a[4:6, 4:6] = set_value
        b = a[a > 1]
        assert b.shape == (4,)
        np.testing.assert_array_equal(b.fetch(), np.ones((4,)) * 2)

        c = b.reshape((2, 2))
        assert c.shape == (2, 2)
        np.testing.assert_array_equal(c.fetch(), np.ones((2, 2)) * 2)


def test_repr_tensor(setup):
    a = mt.ones((10, 10), chunk_size=3)
    assert a.key in repr(a)

    assert repr(np.ones((10, 10))) not in repr(a)
    assert str(np.ones((10, 10))) not in str(a)

    with option_context({"eager_mode": True}):
        a = mt.ones((10, 10))
        assert repr(np.ones((10, 10))) == repr(a)
        assert str(np.ones((10, 10))) == str(a)


def test_repr_dataframe(setup):
    x = pd.DataFrame(np.ones((10, 10)))

    with option_context({"eager_mode": True}):
        a = md.DataFrame(np.ones((10, 10)), chunk_size=3)
        assert repr(x) in repr(a)
        assert str(x) in str(a)

    a = md.DataFrame(np.ones((10, 10)), chunk_size=3)
    assert repr(x) not in repr(a)
    assert str(x) not in str(a)


def test_view(setup):
    with option_context({"eager_mode": True}):
        data = np.random.rand(10, 20)
        a = mt.tensor(data, chunk_size=5)
        b = a[0][1:4]
        b[1] = 10

        npa = data.copy()
        npb = npa[0][1:4]
        npb[1] = 10

        np.testing.assert_array_equal(a.fetch(), npa)
        np.testing.assert_array_equal(b.fetch(), npb)


def test_dataframe(setup):
    with option_context({"eager_mode": True}):
        from ..dataframe.arithmetic import add

        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        pd.testing.assert_frame_equal(df1.fetch(), data1)

        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)
        pd.testing.assert_frame_equal(df2.fetch(), data2)

        df3 = add(df1, df2)
        pd.testing.assert_frame_equal(df3.fetch(), data1 + data2)
