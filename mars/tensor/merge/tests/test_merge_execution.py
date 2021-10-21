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
import scipy.sparse as sps
import pytest

from ...datasource import tensor, empty, eye, ones, zeros
from ... import (
    concatenate,
    stack,
    hstack,
    vstack,
    dstack,
    column_stack,
    union1d,
    array,
    block,
    append,
)


def test_concatenate_execution(setup):
    a_data = np.random.rand(10, 20, 30)
    b_data = np.random.rand(10, 20, 40)
    c_data = np.random.rand(10, 20, 50)

    a = tensor(a_data, chunk_size=8)
    b = tensor(b_data, chunk_size=10)
    c = tensor(c_data, chunk_size=15)

    d = concatenate([a, b, c], axis=-1)
    res = d.execute().fetch()
    expected = np.concatenate([a_data, b_data, c_data], axis=-1)
    np.testing.assert_array_equal(res, expected)

    a_data = sps.random(10, 30)
    b_data = sps.rand(10, 40)
    c_data = sps.rand(10, 50)

    a = tensor(a_data, chunk_size=8)
    b = tensor(b_data, chunk_size=10)
    c = tensor(c_data, chunk_size=15)

    d = concatenate([a, b, c], axis=-1)
    res = d.execute().fetch()
    expected = np.concatenate([a_data.A, b_data.A, c_data.A], axis=-1)
    np.testing.assert_array_equal(res.toarray(), expected)


def test_stack_execution(setup):
    raw = [np.random.randn(3, 4) for _ in range(10)]
    arrs = [tensor(a, chunk_size=3) for a in raw]

    arr2 = stack(arrs)
    res = arr2.execute().fetch()
    assert np.array_equal(res, np.stack(raw)) is True

    arr3 = stack(arrs, axis=1)
    res = arr3.execute().fetch()
    assert np.array_equal(res, np.stack(raw, axis=1)) is True

    arr4 = stack(arrs, axis=2)
    res = arr4.execute().fetch()
    assert np.array_equal(res, np.stack(raw, axis=2)) is True

    raw2 = [np.asfortranarray(np.random.randn(3, 4)) for _ in range(10)]
    arr5 = [tensor(a, chunk_size=3) for a in raw2]

    arr6 = stack(arr5)
    res = arr6.execute().fetch()
    expected = np.stack(raw2).copy("A")
    np.testing.assert_array_equal(res, expected)

    arr7 = stack(arr5, out=empty((10, 3, 4), order="F"))
    res = arr7.execute().fetch()
    expected = np.stack(raw2, out=np.empty((10, 3, 4), order="F")).copy("A")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    # test stack with unknown shapes
    t = tensor(raw[0], chunk_size=3)
    t2 = t[t[:, 0] > 0.0]
    t3 = t2 + 1

    arr8 = stack([t2, t3])
    result = arr8.execute().fetch()
    e = raw[0]
    e2 = e[e[:, 0] > 0.0]
    e3 = e2 + 1
    np.testing.assert_array_equal(result, np.stack([e2, e3]))


def test_h_stack_execution(setup):
    a_data = np.random.rand(10)
    b_data = np.random.rand(20)

    a = tensor(a_data, chunk_size=8)
    b = tensor(b_data, chunk_size=8)

    c = hstack([a, b])
    res = c.execute().fetch()
    expected = np.hstack([a_data, b_data])
    assert np.array_equal(res, expected) is True

    a_data = np.random.rand(10, 20)
    b_data = np.random.rand(10, 5)

    a = tensor(a_data, chunk_size=6)
    b = tensor(b_data, chunk_size=8)

    c = hstack([a, b])
    res = c.execute().fetch()
    expected = np.hstack([a_data, b_data])
    assert np.array_equal(res, expected) is True


def test_v_stack_execution(setup):
    a_data = np.random.rand(10)
    b_data = np.random.rand(10)

    a = tensor(a_data, chunk_size=8)
    b = tensor(b_data, chunk_size=8)

    c = vstack([a, b])
    res = c.execute().fetch()
    expected = np.vstack([a_data, b_data])
    assert np.array_equal(res, expected) is True

    a_data = np.random.rand(10, 20)
    b_data = np.random.rand(5, 20)

    a = tensor(a_data, chunk_size=6)
    b = tensor(b_data, chunk_size=8)

    c = vstack([a, b])
    res = c.execute().fetch()
    expected = np.vstack([a_data, b_data])
    assert np.array_equal(res, expected) is True


def test_d_stack_execution(setup):
    a_data = np.random.rand(10)
    b_data = np.random.rand(10)

    a = tensor(a_data, chunk_size=8)
    b = tensor(b_data, chunk_size=8)

    c = dstack([a, b])
    res = c.execute().fetch()
    expected = np.dstack([a_data, b_data])
    assert np.array_equal(res, expected) is True

    a_data = np.random.rand(10, 20)
    b_data = np.random.rand(10, 20)

    a = tensor(a_data, chunk_size=6)
    b = tensor(b_data, chunk_size=8)

    c = dstack([a, b])
    res = c.execute().fetch()
    expected = np.dstack([a_data, b_data])
    assert np.array_equal(res, expected) is True


def test_column_stack_execution(setup):
    a_data = np.array((1, 2, 3))
    b_data = np.array((2, 3, 4))
    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=2)

    c = column_stack((a, b))
    res = c.execute().fetch()
    expected = np.column_stack((a_data, b_data))
    np.testing.assert_equal(res, expected)

    a_data = np.random.rand(4, 2, 3)
    b_data = np.random.rand(4, 2, 3)
    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=2)

    c = column_stack((a, b))
    res = c.execute().fetch()
    expected = np.column_stack((a_data, b_data))
    np.testing.assert_equal(res, expected)


def test_union1d_execution(setup):
    rs = np.random.RandomState(0)
    raw1 = rs.random(10)
    raw2 = rs.random(9)

    t1 = tensor(raw1, chunk_size=3)
    t2 = tensor(raw2, chunk_size=4)

    t = union1d(t1, t2, aggregate_size=1)
    res = t.execute().fetch()
    expected = np.union1d(raw1, raw2)
    np.testing.assert_array_equal(res, expected)

    t = union1d(t1, t2)
    res = t.execute().fetch()
    expected = np.union1d(raw1, raw2)
    np.testing.assert_array_equal(res, expected)


def test_block_execution(setup):
    # arrays is a tuple.
    with pytest.raises(TypeError):
        block((1, 2, 3))

    # List depths are mismatched.
    with pytest.raises(ValueError):
        block([[1, 2], [[3, 4]]])

    # List at arrays cannot be empty.
    with pytest.raises(ValueError):
        block([])

    # List at arrays[1] cannot be empty.
    with pytest.raises(ValueError):
        block([[1, 2], []])

    # Mismatched array shapes.
    with pytest.raises(ValueError):
        block([eye(512), eye(512), ones((511, 1))])

    # Test large block.
    block([eye(512), eye(512), ones((512, 1))])

    # Test block inputs a single array.
    c = block(array([1, 2, 3]))
    r = c.execute().fetch()
    np.testing.assert_array_equal(r, array([1, 2, 3]))

    a = eye(2) * 2
    b = eye(3) * 3
    c = block([[a, zeros((2, 3))], [ones((3, 2)), b]])
    r = c.execute().fetch()
    expected = array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 3.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 3.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(r, expected)

    # eye with different chunk sizes
    a = eye(5, chunk_size=2) * 2
    b = eye(4, chunk_size=3) * 3
    c = block([[a, zeros((5, 4), chunk_size=4)], [ones((4, 5), chunk_size=5), b]])
    r = c.execute().fetch()
    expected = array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 3.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(r, expected)

    # hstack([1, 2, 3])
    c = block([1, 2, 3])
    r = c.execute().fetch()
    expected = array([1, 2, 3])
    np.testing.assert_array_equal(r, expected)

    # hstack([a, b, 10])
    a = array([1, 2, 3])
    b = array([2, 3, 4])
    c = block([a, b, 10])
    r = c.execute().fetch()
    expected = array([1, 2, 3, 2, 3, 4, 10])
    np.testing.assert_array_equal(r, expected)

    # hstack([a, b, 10]) with different chunk sizes
    a = array([1, 2, 3, 4, 5, 6, 7], chunk_size=3)
    b = array([2, 3, 4, 5], chunk_size=4)
    c = block([a, b, 10])
    r = c.execute().fetch()
    expected = array([1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 10])
    np.testing.assert_array_equal(r, expected)

    # hstack([A, B])
    A = ones((2, 2), int)
    B = 2 * A
    c = block([A, B])
    r = c.execute().fetch()
    expected = array([[1, 1, 2, 2], [1, 1, 2, 2]])
    np.testing.assert_array_equal(r, expected)

    # vstack([a, b])
    a = array([1, 2, 3])
    b = array([2, 3, 4])
    c = block([[a], [b]])
    r = c.execute().fetch()
    expected = array([[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(r, expected)

    # vstack([a, b]) with different chunk sizes
    a = array([1, 2, 3, 4, 5, 6, 7], chunk_size=5)
    b = array([2, 3, 4, 5, 6, 7, 8], chunk_size=6)
    c = block([[a], [b]])
    r = c.execute().fetch()
    expected = array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]])
    np.testing.assert_array_equal(r, expected)

    # vstack([A, B])
    A = ones((2, 2), int)
    B = 2 * A
    c = block([[A], [B]])
    r = c.execute().fetch()
    expected = array([[1, 1], [1, 1], [2, 2], [2, 2]])
    np.testing.assert_array_equal(r, expected)

    a = array(0)
    b = array([1])
    # atleast_1d(a)
    c = block([a])
    r = c.execute().fetch()
    expected = array([0])
    np.testing.assert_array_equal(r, expected)
    # atleast_1d(b)
    c = block([b])
    r = c.execute().fetch()
    expected = array([1])
    np.testing.assert_array_equal(r, expected)
    # atleast_2d(a)
    c = block([[a]])
    r = c.execute().fetch()
    expected = array([[0]])
    np.testing.assert_array_equal(r, expected)
    # atleast_2d(b)
    c = block([[b]])
    r = c.execute().fetch()
    expected = array([[1]])
    np.testing.assert_array_equal(r, expected)


@pytest.mark.parametrize("axis", [0, None])
def test_append_execution(setup, axis):
    raw1 = np.random.rand(10, 3)
    raw2 = np.random.rand(6, 3)

    a1 = tensor(raw1, chunk_size=3)
    a2 = tensor(raw2, chunk_size=4)
    r = append(a1, a2, axis=axis)
    result = r.execute().fetch()
    expected = np.append(raw1, raw2, axis=axis)
    np.testing.assert_array_equal(result, expected)
