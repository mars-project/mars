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
import pandas as pd
import pytest

from .... import dataframe as md
from .... import tensor as mt
from .... import execute, fetch
from ....tests.core import require_cupy
from ...datasource import tensor, ones, zeros, arange
from .. import (
    copyto,
    transpose,
    moveaxis,
    broadcast_to,
    broadcast_arrays,
    where,
    expand_dims,
    rollaxis,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    argwhere,
    array_split,
    split,
    hsplit,
    vsplit,
    dsplit,
    roll,
    squeeze,
    diff,
    ediff1d,
    flip,
    flipud,
    fliplr,
    repeat,
    tile,
    isin,
    searchsorted,
    unique,
    sort,
    argsort,
    partition,
    argpartition,
    topk,
    argtopk,
    trapz,
    shape,
    to_gpu,
    to_cpu,
    swapaxes,
)


def test_rechunk_execution(setup):
    raw = np.random.RandomState(0).random((11, 8))
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.rechunk(4)

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, raw)


def test_copyto_execution(setup):
    a = ones((2, 3), chunk_size=1)
    b = tensor([3, -1, 3], chunk_size=2)

    copyto(a, b, where=b > 1)

    res = a.execute().fetch()
    expected = np.array([[3, 1, 3], [3, 1, 3]])

    np.testing.assert_equal(res, expected)

    a = ones((2, 3), chunk_size=1)
    b = tensor(np.asfortranarray(np.random.rand(2, 3)), chunk_size=2)

    copyto(b, a)

    res = b.execute().fetch()
    expected = np.asfortranarray(np.ones((2, 3)))

    np.testing.assert_array_equal(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False


@pytest.mark.ray_dag
def test_astype_execution(setup):
    raw = np.random.random((10, 5))
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.astype("i8")

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, raw.astype("i8"))

    raw = sps.random(10, 5, density=0.2)
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.astype("i8")

    res = arr2.execute().fetch()
    assert np.array_equal(res.toarray(), raw.astype("i8").toarray()) is True

    raw = np.asfortranarray(np.random.random((10, 5)))
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.astype("i8", order="C")

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, raw.astype("i8"))
    assert res.flags["C_CONTIGUOUS"] is True
    assert res.flags["F_CONTIGUOUS"] is False


def test_transpose_execution(setup):
    raw = np.random.random((11, 8, 5))
    arr = tensor(raw, chunk_size=3)
    arr2 = transpose(arr)

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, raw.T)

    arr3 = transpose(arr, axes=(-2, -1, -3))

    res = arr3.execute().fetch()
    np.testing.assert_array_equal(res, raw.transpose(1, 2, 0))

    raw = sps.random(11, 8)
    arr = tensor(raw, chunk_size=3)
    arr2 = transpose(arr)

    assert arr2.issparse() is True

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res.toarray(), raw.T.toarray())

    # test order
    raw = np.asfortranarray(np.random.random((11, 8, 5)))

    arr = tensor(raw, chunk_size=3)
    arr2 = transpose(arr)

    res = arr2.execute().fetch()
    expected = np.transpose(raw).copy(order="A")

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    arr = tensor(raw, chunk_size=3)
    arr2 = transpose(arr, (1, 2, 0))

    res = arr2.execute().fetch()
    expected = np.transpose(raw, (1, 2, 0)).copy(order="A")

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    df = md.DataFrame(mt.random.rand(10, 5, chunk_size=5))
    df = df[df[0] < 1]
    # generate tensor with unknown shape
    t = df.to_tensor()
    t2 = transpose(t)

    res = t2.execute().fetch()
    assert res.shape == (5, 10)


def test_swapaxes_execution(setup):
    raw = np.random.random((11, 8, 5))
    arr = swapaxes(raw, 2, 0)

    res = arr.execute().fetch()
    np.testing.assert_array_equal(res, raw.swapaxes(2, 0))

    raw = np.random.random((11, 8, 5))
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.swapaxes(2, 0)

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, raw.swapaxes(2, 0))

    raw = sps.random(11, 8, density=0.2)
    arr = tensor(raw, chunk_size=3)
    arr2 = arr.swapaxes(1, 0)

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res.toarray(), raw.toarray().swapaxes(1, 0))

    # test order
    raw = np.asfortranarray(np.random.rand(11, 8, 5))

    arr = tensor(raw, chunk_size=3)
    arr2 = arr.swapaxes(2, 0)

    res = arr2.execute().fetch()
    expected = raw.swapaxes(2, 0).copy(order="A")

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    arr = tensor(raw, chunk_size=3)
    arr2 = arr.swapaxes(0, 2)

    res = arr2.execute().fetch()
    expected = raw.swapaxes(0, 2).copy(order="A")

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    arr = tensor(raw, chunk_size=3)
    arr2 = arr.swapaxes(1, 0)

    res = arr2.execute().fetch()
    expected = raw.swapaxes(1, 0).copy(order="A")

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_moveaxis_execution(setup):
    x = zeros((3, 4, 5), chunk_size=2)

    t = moveaxis(x, 0, -1)

    res = t.execute().fetch()
    assert res.shape == (4, 5, 3)

    t = moveaxis(x, -1, 0)

    res = t.execute().fetch()
    assert res.shape == (5, 3, 4)

    t = moveaxis(x, [0, 1], [-1, -2])

    res = t.execute().fetch()
    assert res.shape == (5, 4, 3)

    t = moveaxis(x, [0, 1, 2], [-1, -2, -3])

    res = t.execute().fetch()
    assert res.shape == (5, 4, 3)


def test_broadcast_to_execution(setup):
    raw = np.random.random((10, 5, 1))
    arr = tensor(raw, chunk_size=2)
    arr2 = broadcast_to(arr, (5, 10, 5, 6))

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, np.broadcast_to(raw, (5, 10, 5, 6)))

    # test chunk with unknown shape
    arr1 = mt.random.rand(3, 4, chunk_size=2)
    arr2 = mt.random.permutation(arr1)
    arr3 = broadcast_to(arr2, (2, 3, 4))

    res = arr3.execute().fetch()
    assert res.shape == (2, 3, 4)


def test_broadcast_arrays_executions(setup):
    x_data = [[1, 2, 3]]
    x = tensor(x_data, chunk_size=1)
    y_data = [[1], [2], [3]]
    y = tensor(y_data, chunk_size=2)

    a = broadcast_arrays(x, y)

    res = [arr.execute().fetch() for arr in a]
    expected = np.broadcast_arrays(x_data, y_data)

    for r, e in zip(res, expected):
        np.testing.assert_equal(r, e)


def test_where_execution(setup):
    raw_cond = np.random.randint(0, 2, size=(4, 4), dtype="?")
    raw_x = np.random.rand(4, 1)
    raw_y = np.random.rand(4, 4)

    cond, x, y = (
        tensor(raw_cond, chunk_size=2),
        tensor(raw_x, chunk_size=2),
        tensor(raw_y, chunk_size=2),
    )

    arr = where(cond, x, y)
    res = arr.execute().fetch()
    assert np.array_equal(res, np.where(raw_cond, raw_x, raw_y)) is True

    raw_cond = sps.csr_matrix(np.random.randint(0, 2, size=(4, 4), dtype="?"))
    raw_x = sps.random(4, 1, density=0.1)
    raw_y = sps.random(4, 4, density=0.1)

    cond, x, y = (
        tensor(raw_cond, chunk_size=2),
        tensor(raw_x, chunk_size=2),
        tensor(raw_y, chunk_size=2),
    )

    arr = where(cond, x, y)
    res = arr.execute().fetch()
    assert (
        np.array_equal(
            res.toarray(),
            np.where(raw_cond.toarray(), raw_x.toarray(), raw_y.toarray()),
        )
        is True
    )

    # GH 2009
    raw_x = np.arange(9.0).reshape(3, 3)
    x = arange(9.0).reshape(3, 3)
    arr = where(x < 5, 2, -1)
    res = arr.execute().fetch()
    np.testing.assert_array_equal(res, np.where(raw_x < 5, 2, -1))


@pytest.mark.ray_dag
def test_reshape_execution(setup):
    raw_data = np.random.rand(5, 10, 30)
    x = tensor(raw_data, chunk_size=8)

    y = x.reshape(-1, 30)

    res = y.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(-1, 30))

    y2 = x.reshape(10, -1)

    res = y2.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(10, -1))

    y3 = x.reshape(-1)

    res = y3.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(-1))

    y4 = x.ravel()

    res = y4.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.ravel())

    raw_data = np.random.rand(6, 20, 4)
    x = tensor(raw_data, chunk_size=5)

    y = x.reshape(-1, 4, 5, 2, 2)

    res = y.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(-1, 4, 5, 2, 2))

    y2 = x.reshape(120, 2, 2)

    res = y2.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(120, 2, 2))

    y3 = x.reshape(12, 5, 8)

    res = y3.execute().fetch()
    np.testing.assert_array_equal(res, raw_data.reshape(12, 5, 8))

    y4 = x.reshape(12, 5, 8)
    y4.op.extra_params["_reshape_with_shuffle"] = True

    # size_res = self.executor.execute_tensor(y4, mock=True)
    res = y4.execute().fetch()
    # assert res[0].nbytes == sum(v[0] for v in size_res)
    assert np.array_equal(res, raw_data.reshape(12, 5, 8)) is True

    y5 = x.ravel(order="F")

    res = y5.execute().fetch()
    expected = raw_data.ravel(order="F")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_expand_dims_execution(setup):
    raw_data = np.random.rand(10, 20, 30)
    x = tensor(raw_data, chunk_size=6)

    y = expand_dims(x, 1)

    res = y.execute().fetch()
    assert np.array_equal(res, np.expand_dims(raw_data, 1)) is True

    y = expand_dims(x, 0)

    res = y.execute().fetch()
    assert np.array_equal(res, np.expand_dims(raw_data, 0)) is True

    y = expand_dims(x, 3)

    res = y.execute().fetch()
    assert np.array_equal(res, np.expand_dims(raw_data, 3)) is True

    y = expand_dims(x, -1)

    res = y.execute().fetch()
    assert np.array_equal(res, np.expand_dims(raw_data, -1)) is True

    y = expand_dims(x, -4)

    res = y.execute().fetch()
    assert np.array_equal(res, np.expand_dims(raw_data, -4)) is True

    with pytest.raises(np.AxisError):
        expand_dims(x, -5)

    with pytest.raises(np.AxisError):
        expand_dims(x, 4)


def test_rollaxis_execution(setup):
    x = ones((3, 4, 5, 6), chunk_size=1)
    y = rollaxis(x, 3, 1)

    res = y.execute().fetch()
    np.testing.assert_array_equal(res, np.rollaxis(np.ones((3, 4, 5, 6)), 3, 1))


def test_atleast1d_execution(setup):
    x = 1
    y = ones(3, chunk_size=2)
    z = ones((3, 4), chunk_size=2)

    t = atleast_1d(x, y, z)

    res = [i.execute().fetch() for i in t]

    np.testing.assert_array_equal(res[0], np.array([1]))
    np.testing.assert_array_equal(res[1], np.ones(3))
    np.testing.assert_array_equal(res[2], np.ones((3, 4)))


def test_atleast2d_execution(setup):
    x = 1
    y = ones(3, chunk_size=2)
    z = ones((3, 4), chunk_size=2)

    t = atleast_2d(x, y, z)

    res = [i.execute().fetch() for i in t]

    np.testing.assert_array_equal(res[0], np.array([[1]]))
    np.testing.assert_array_equal(res[1], np.atleast_2d(np.ones(3)))
    assert np.array_equal(res[2], np.ones((3, 4))) is True


def test_atleast3d_execution(setup):
    x = 1
    y = ones(3, chunk_size=2)
    z = ones((3, 4), chunk_size=2)

    t = atleast_3d(x, y, z)

    res = [i.execute().fetch() for i in t]

    np.testing.assert_array_equal(res[0], np.atleast_3d(x))
    np.testing.assert_array_equal(res[1], np.atleast_3d(np.ones(3)))
    np.testing.assert_array_equal(res[2], np.atleast_3d(np.ones((3, 4))))


def test_argwhere_execution(setup):
    x = arange(6, chunk_size=2).reshape(2, 3)
    t = argwhere(x > 1)

    res = t.execute().fetch()
    expected = np.argwhere(np.arange(6).reshape(2, 3) > 1)

    np.testing.assert_array_equal(res, expected)

    data = np.asfortranarray(np.random.rand(10, 20))
    x = tensor(data, chunk_size=10)

    t = argwhere(x > 0.5)

    res = t.execute().fetch()
    expected = np.argwhere(data > 0.5)

    np.testing.assert_array_equal(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False


def test_array_split_execution(setup):
    x = arange(48, chunk_size=3).reshape(2, 3, 8)
    ss = array_split(x, 3, axis=2)

    res = [i.execute().fetch() for i in ss]
    expected = np.array_split(np.arange(48).reshape(2, 3, 8), 3, axis=2)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    ss = array_split(x, [3, 5, 6, 10], axis=2)

    res = [i.execute().fetch() for i in ss]
    expected = np.array_split(np.arange(48).reshape(2, 3, 8), [3, 5, 6, 10], axis=2)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]


def test_split_execution(setup):
    for a in ((1, 1, 1, 2, 2, 3), [1, 1, 1, 2, 2, 3]):
        splits = split(a, (3, 5))
        assert len(splits) == 3
        splits0 = splits[0].execute().fetch()
        np.testing.assert_array_equal(splits0, (1, 1, 1))
        splits1 = splits[1].execute().fetch()
        np.testing.assert_array_equal(splits1, (2, 2))
        splits2 = splits[2].execute().fetch()
        np.testing.assert_array_equal(splits2, (3,))

    x = arange(48, chunk_size=3).reshape(2, 3, 8)
    ss = split(x, 4, axis=2)

    res = [i.execute().fetch() for i in ss]
    expected = np.split(np.arange(48).reshape(2, 3, 8), 4, axis=2)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    ss = split(x, [3, 5, 6, 10], axis=2)

    res = [i.execute().fetch() for i in ss]
    expected = np.split(np.arange(48).reshape(2, 3, 8), [3, 5, 6, 10], axis=2)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    # hsplit
    x = arange(120, chunk_size=3).reshape(2, 12, 5)
    ss = hsplit(x, 4)

    res = [i.execute().fetch() for i in ss]
    expected = np.hsplit(np.arange(120).reshape(2, 12, 5), 4)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    # vsplit
    x = arange(48, chunk_size=3).reshape(8, 3, 2)
    ss = vsplit(x, 4)

    res = [i.execute().fetch() for i in ss]
    expected = np.vsplit(np.arange(48).reshape(8, 3, 2), 4)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    # dsplit
    x = arange(48, chunk_size=3).reshape(2, 3, 8)
    ss = dsplit(x, 4)

    res = [i.execute().fetch() for i in ss]
    expected = np.dsplit(np.arange(48).reshape(2, 3, 8), 4)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]

    x_data = sps.random(12, 8, density=0.1)
    x = tensor(x_data, chunk_size=3)
    ss = split(x, 4, axis=0)

    res = [i.execute().fetch() for i in ss]
    expected = np.split(x_data.toarray(), 4, axis=0)
    assert len(res) == len(expected)
    [np.testing.assert_equal(r.toarray(), e) for r, e in zip(res, expected)]


def test_roll_execution(setup):
    x = arange(10, chunk_size=2)

    t = roll(x, 2)

    res = t.execute().fetch()
    expected = np.roll(np.arange(10), 2)
    np.testing.assert_equal(res, expected)

    x2 = x.reshape(2, 5)

    t = roll(x2, 1)

    res = t.execute().fetch()
    expected = np.roll(np.arange(10).reshape(2, 5), 1)
    np.testing.assert_equal(res, expected)

    t = roll(x2, 1, axis=0)

    res = t.execute().fetch()
    expected = np.roll(np.arange(10).reshape(2, 5), 1, axis=0)
    np.testing.assert_equal(res, expected)

    t = roll(x2, 1, axis=1)

    res = t.execute().fetch()
    expected = np.roll(np.arange(10).reshape(2, 5), 1, axis=1)
    np.testing.assert_equal(res, expected)


def test_squeeze_execution(setup):
    data = np.array([[[0], [1], [2]]])
    x = tensor(data, chunk_size=1)

    t = squeeze(x)

    res = t.execute().fetch()
    expected = np.squeeze(data)
    np.testing.assert_equal(res, expected)

    t = squeeze(x, axis=2)

    res = t.execute().fetch()
    expected = np.squeeze(data, axis=2)
    np.testing.assert_equal(res, expected)


def test_diff_execution(setup):
    data = np.array([1, 2, 4, 7, 0])
    x = tensor(data, chunk_size=2)

    t = diff(x)

    res = t.execute().fetch()
    expected = np.diff(data)
    np.testing.assert_equal(res, expected)

    t = diff(x, n=2)

    res = t.execute().fetch()
    expected = np.diff(data, n=2)
    np.testing.assert_equal(res, expected)

    data = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    x = tensor(data, chunk_size=2)

    t = diff(x)

    res = t.execute().fetch()
    expected = np.diff(data)
    np.testing.assert_equal(res, expected)

    t = diff(x, axis=0)

    res = t.execute().fetch()
    expected = np.diff(data, axis=0)
    np.testing.assert_equal(res, expected)

    x = mt.arange("1066-10-13", "1066-10-16", dtype=mt.datetime64)
    t = diff(x)

    res = t.execute().fetch()
    expected = np.diff(np.arange("1066-10-13", "1066-10-16", dtype=np.datetime64))
    np.testing.assert_equal(res, expected)


def test_ediff1d(setup):
    data = np.array([1, 2, 4, 7, 0])
    x = tensor(data, chunk_size=2)

    t = ediff1d(x)

    res = t.execute().fetch()
    expected = np.ediff1d(data)
    np.testing.assert_equal(res, expected)

    to_begin = tensor(-99, chunk_size=2)
    to_end = tensor([88, 99], chunk_size=2)
    t = ediff1d(x, to_begin=to_begin, to_end=to_end)

    res = t.execute().fetch()
    expected = np.ediff1d(data, to_begin=-99, to_end=np.array([88, 99]))
    np.testing.assert_equal(res, expected)

    data = [[1, 2, 4], [1, 6, 24]]

    t = ediff1d(tensor(data, chunk_size=2))

    res = t.execute().fetch()
    expected = np.ediff1d(data)
    np.testing.assert_equal(res, expected)


def test_flip_execution(setup):
    a = arange(8, chunk_size=2).reshape((2, 2, 2))

    t = flip(a, 0)

    res = t.execute().fetch()
    expected = np.flip(np.arange(8).reshape(2, 2, 2), 0)
    np.testing.assert_equal(res, expected)

    t = flip(a, 1)

    res = t.execute().fetch()
    expected = np.flip(np.arange(8).reshape(2, 2, 2), 1)
    np.testing.assert_equal(res, expected)

    t = flipud(a)

    res = t.execute().fetch()
    expected = np.flipud(np.arange(8).reshape(2, 2, 2))
    np.testing.assert_equal(res, expected)

    t = fliplr(a)

    res = t.execute().fetch()
    expected = np.fliplr(np.arange(8).reshape(2, 2, 2))
    np.testing.assert_equal(res, expected)


def test_repeat_execution(setup):
    a = repeat(3, 4)

    res = a.execute().fetch()
    expected = np.repeat(3, 4)
    np.testing.assert_equal(res, expected)

    x_data = np.random.randn(20, 30)
    x = tensor(x_data, chunk_size=(12, 16))

    t = repeat(x, 2)

    res = t.execute().fetch()
    expected = np.repeat(x_data, 2)
    np.testing.assert_equal(res, expected)

    t = repeat(x, 3, axis=1)

    res = t.execute().fetch()
    expected = np.repeat(x_data, 3, axis=1)
    np.testing.assert_equal(res, expected)

    t = repeat(x, np.arange(20), axis=0)

    res = t.execute().fetch()
    expected = np.repeat(x_data, np.arange(20), axis=0)
    np.testing.assert_equal(res, expected)

    t = repeat(x, arange(20, chunk_size=10), axis=0)

    res = t.execute().fetch()
    expected = np.repeat(x_data, np.arange(20), axis=0)
    np.testing.assert_equal(res, expected)

    x_data = sps.random(20, 30, density=0.1)
    x = tensor(x_data, chunk_size=(12, 16))

    t = repeat(x, 2, axis=1)

    res = t.execute().fetch()
    expected = np.repeat(x_data.toarray(), 2, axis=1)
    np.testing.assert_equal(res.toarray(), expected)


def test_tile_execution(setup):
    a_data = np.array([0, 1, 2])
    a = tensor(a_data, chunk_size=2)

    t = tile(a, 2)

    res = t.execute().fetch()
    expected = np.tile(a_data, 2)
    np.testing.assert_equal(res, expected)

    t = tile(a, (2, 2))

    res = t.execute().fetch()
    expected = np.tile(a_data, (2, 2))
    np.testing.assert_equal(res, expected)

    t = tile(a, (2, 1, 2))

    res = t.execute().fetch()
    expected = np.tile(a_data, (2, 1, 2))
    np.testing.assert_equal(res, expected)

    b_data = np.array([[1, 2], [3, 4]])
    b = tensor(b_data, chunk_size=1)

    t = tile(b, 2)

    res = t.execute().fetch()
    expected = np.tile(b_data, 2)
    np.testing.assert_equal(res, expected)

    t = tile(b, (2, 1))

    res = t.execute().fetch()
    expected = np.tile(b_data, (2, 1))
    np.testing.assert_equal(res, expected)

    c_data = np.array([1, 2, 3, 4])
    c = tensor(c_data, chunk_size=3)

    t = tile(c, (4, 1))

    res = t.execute().fetch()
    expected = np.tile(c_data, (4, 1))
    np.testing.assert_equal(res, expected)


@pytest.mark.ray_dag
def test_isin_execution(setup):
    element = 2 * arange(4, chunk_size=1).reshape((2, 2))
    test_elements = [1, 2, 4, 8]

    mask = isin(element, test_elements)

    res = mask.execute().fetch()
    expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_elements)
    np.testing.assert_equal(res, expected)

    res = element[mask].execute().fetch()
    expected = np.array([2, 4])
    np.testing.assert_equal(res, expected)

    mask = isin(element, test_elements, invert=True)

    res = mask.execute().fetch()
    expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_elements, invert=True)
    np.testing.assert_equal(res, expected)

    res = element[mask].execute().fetch()
    expected = np.array([0, 6])
    np.testing.assert_equal(res, expected)

    test_set = {1, 2, 4, 8}
    mask = isin(element, test_set)

    res = mask.execute().fetch()
    expected = np.isin(2 * np.arange(4).reshape((2, 2)), test_set)
    np.testing.assert_equal(res, expected)


def test_ravel_execution(setup):
    arr = ones((10, 5), chunk_size=2)
    flat_arr = mt.ravel(arr)

    res = flat_arr.execute().fetch()
    assert len(res) == 50
    np.testing.assert_equal(res, np.ones(50))


def test_searchsorted_execution(setup):
    raw = np.sort(np.random.randint(100, size=(16,)))

    # test different chunk_size, 3 will have combine, 6 will skip combine
    for chunk_size in (3, 8):
        arr = tensor(raw, chunk_size=chunk_size)

        # test scalar, with value in the middle
        t1 = searchsorted(arr, 20)

        res = t1.execute().fetch()
        expected = np.searchsorted(raw, 20)
        np.testing.assert_array_equal(res, expected)

        # test scalar, with value larger than 100
        t2 = searchsorted(arr, 200)

        res = t2.execute().fetch()
        expected = np.searchsorted(raw, 200)
        np.testing.assert_array_equal(res, expected)

        # test scalar, side left, with value exact in the middle of the array
        t3 = searchsorted(arr, raw[10], side="left")

        res = t3.execute().fetch()
        expected = np.searchsorted(raw, raw[10], side="left")
        np.testing.assert_array_equal(res, expected)

        # test scalar, side right, with value exact in the middle of the array
        t4 = searchsorted(arr, raw[10], side="right")

        res = t4.execute().fetch()
        expected = np.searchsorted(raw, raw[10], side="right")
        np.testing.assert_array_equal(res, expected)

        # test scalar, side left, with value exact in the end of the array
        t5 = searchsorted(arr, raw[15], side="left")

        res = t5.execute().fetch()
        expected = np.searchsorted(raw, raw[15], side="left")
        np.testing.assert_array_equal(res, expected)

        # test scalar, side right, with value exact in the end of the array
        t6 = searchsorted(arr, raw[15], side="right")

        res = t6.execute().fetch()
        expected = np.searchsorted(raw, raw[15], side="right")
        np.testing.assert_array_equal(res, expected)

        # test scalar, side left, with value exact in the start of the array
        t7 = searchsorted(arr, raw[0], side="left")

        res = t7.execute().fetch()
        expected = np.searchsorted(raw, raw[0], side="left")
        np.testing.assert_array_equal(res, expected)

        # test scalar, side right, with value exact in the start of the array
        t8 = searchsorted(arr, raw[0], side="right")

        res = t8.execute().fetch()
        expected = np.searchsorted(raw, raw[0], side="right")
        np.testing.assert_array_equal(res, expected)

        raw2 = np.random.randint(100, size=(3, 4))

        # test tensor, side left
        t9 = searchsorted(arr, tensor(raw2, chunk_size=2), side="left")

        res = t9.execute().fetch()
        expected = np.searchsorted(raw, raw2, side="left")
        np.testing.assert_array_equal(res, expected)

        # test tensor, side right
        t10 = searchsorted(arr, tensor(raw2, chunk_size=2), side="right")

        res = t10.execute().fetch()
        expected = np.searchsorted(raw, raw2, side="right")
        np.testing.assert_array_equal(res, expected)

    # test one chunk
    arr = tensor(raw, chunk_size=16)

    # test scalar, tensor to search has 1 chunk
    t11 = searchsorted(arr, 20)
    res = t11.execute().fetch()
    expected = np.searchsorted(raw, 20)
    np.testing.assert_array_equal(res, expected)

    # test tensor with 1 chunk, tensor to search has 1 chunk
    t12 = searchsorted(arr, tensor(raw2, chunk_size=4))

    res = t12.execute().fetch()
    expected = np.searchsorted(raw, raw2)
    np.testing.assert_array_equal(res, expected)

    # test tensor with more than 1 chunk, tensor to search has 1 chunk
    t13 = searchsorted(arr, tensor(raw2, chunk_size=2))

    res = t13.execute().fetch()
    expected = np.searchsorted(raw, raw2)
    np.testing.assert_array_equal(res, expected)

    # test sorter
    raw3 = np.random.randint(100, size=(16,))
    arr = tensor(raw3, chunk_size=3)
    order = np.argsort(raw3)
    order_arr = tensor(order, chunk_size=4)

    t14 = searchsorted(arr, 20, sorter=order_arr)

    res = t14.execute().fetch()
    expected = np.searchsorted(raw3, 20, sorter=order)
    np.testing.assert_array_equal(res, expected)

    # all data same
    raw4 = np.ones(8)
    arr = tensor(raw4, chunk_size=2)

    for val in (0, 1, 2):
        for side in ("left", "right"):
            t15 = searchsorted(arr, val, side=side)

            res = t15.execute().fetch()
            expected = np.searchsorted(raw4, val, side=side)
            np.testing.assert_array_equal(res, expected)


@pytest.mark.ray_dag
def test_unique_execution(setup):
    rs = np.random.RandomState(0)
    raw = rs.randint(10, size=(10,))

    for chunk_size in (10, 3):
        x = tensor(raw, chunk_size=chunk_size)

        y = unique(x)

        res = y.execute().fetch()
        expected = np.unique(raw)
        np.testing.assert_array_equal(res, expected)

        y, indices = unique(x, return_index=True)

        res = fetch(execute(y, indices))
        expected = np.unique(raw, return_index=True)
        assert len(res) == 2
        assert len(expected) == 2
        np.testing.assert_array_equal(res[0], expected[0])
        np.testing.assert_array_equal(res[1], expected[1])

        y, inverse = unique(x, return_inverse=True)

        res = fetch(*execute(y, inverse))
        expected = np.unique(raw, return_inverse=True)
        assert len(res) == 2
        assert len(expected) == 2
        np.testing.assert_array_equal(res[0], expected[0])
        np.testing.assert_array_equal(res[1], expected[1])

        y, counts = unique(x, return_counts=True)

        res = fetch(*execute(y, counts))
        expected = np.unique(raw, return_counts=True)
        assert len(res) == 2
        assert len(expected) == 2
        np.testing.assert_array_equal(res[0], expected[0])
        np.testing.assert_array_equal(res[1], expected[1])

        y, indices, inverse, counts = unique(
            x, return_index=True, return_inverse=True, return_counts=True
        )

        res = fetch(*execute(y, indices, inverse, counts))
        expected = np.unique(
            raw, return_index=True, return_inverse=True, return_counts=True
        )
        assert len(res) == 4
        assert len(expected) == 4
        np.testing.assert_array_equal(res[0], expected[0])
        np.testing.assert_array_equal(res[1], expected[1])
        np.testing.assert_array_equal(res[2], expected[2])
        np.testing.assert_array_equal(res[3], expected[3])

        y, indices, counts = unique(x, return_index=True, return_counts=True)

        res = fetch(*execute(y, indices, counts))
        expected = np.unique(raw, return_index=True, return_counts=True)
        assert len(res) == 3
        assert len(expected) == 3
        np.testing.assert_array_equal(res[0], expected[0])
        np.testing.assert_array_equal(res[1], expected[1])
        np.testing.assert_array_equal(res[2], expected[2])

        raw2 = rs.randint(10, size=(4, 5, 6))
        x2 = tensor(raw2, chunk_size=chunk_size)

        y2 = unique(x2)

        res = y2.execute().fetch()
        expected = np.unique(raw2)
        np.testing.assert_array_equal(res, expected)

        y2 = unique(x2, axis=1)

        res = y2.execute().fetch()
        expected = np.unique(raw2, axis=1)
        np.testing.assert_array_equal(res, expected)

        y2 = unique(x2, axis=2)

        res = y2.execute().fetch()
        expected = np.unique(raw2, axis=2)
        np.testing.assert_array_equal(res, expected)

    raw = rs.randint(10, size=(10, 20))
    raw[:, 0] = raw[:, 11] = rs.randint(10, size=(10,))
    x = tensor(raw, chunk_size=2)
    y, ind, inv, counts = unique(
        x,
        aggregate_size=3,
        axis=1,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    res_unique, res_ind, res_inv, res_counts = fetch(*execute(y, ind, inv, counts))
    exp_unique, exp_ind, exp_counts = np.unique(
        raw, axis=1, return_index=True, return_counts=True
    )
    raw_res_unique = res_unique
    res_unique_df = pd.DataFrame(res_unique)
    res_unique_ind = np.asarray(
        res_unique_df.sort_values(list(range(res_unique.shape[0])), axis=1).columns
    )
    res_unique = res_unique[:, res_unique_ind]
    res_ind = res_ind[res_unique_ind]
    res_counts = res_counts[res_unique_ind]

    np.testing.assert_array_equal(res_unique, exp_unique)
    np.testing.assert_array_equal(res_ind, exp_ind)
    np.testing.assert_array_equal(raw_res_unique[:, res_inv], raw)
    np.testing.assert_array_equal(res_counts, exp_counts)

    x = (mt.random.RandomState(0).rand(1000, chunk_size=20) > 0.5).astype(np.int32)
    y = unique(x)
    res = np.sort(y.execute().fetch())
    np.testing.assert_array_equal(res, np.array([0, 1]))

    # test sparse
    sparse_raw = sps.random(10, 3, density=0.1, format="csr", random_state=rs)
    x = tensor(sparse_raw, chunk_size=2)
    y = unique(x)
    res = np.sort(y.execute().fetch())
    np.testing.assert_array_equal(res, np.unique(sparse_raw.data))

    # test empty
    x = tensor([])
    y = unique(x)
    res = y.execute().fetch()
    np.testing.assert_array_equal(res, np.unique([]))

    x = tensor([[]])
    y = unique(x)
    res = y.execute().fetch()
    np.testing.assert_array_equal(res, np.unique([[]]))


@require_cupy
def test_to_gpu_execution(setup_gpu):
    raw = np.random.rand(10, 10)
    x = tensor(raw, chunk_size=3)

    gx = to_gpu(x)

    res = gx.execute().fetch()
    np.testing.assert_array_equal(res.get(), raw)


@require_cupy
def test_to_cpu_execution(setup_gpu):
    raw = np.random.rand(10, 10)
    x = tensor(raw, chunk_size=3, gpu=True)

    cx = to_cpu(x)

    res = cx.execute().fetch()
    np.testing.assert_array_equal(res, raw)


@pytest.mark.ray_dag
def test_sort_execution(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=20)

    sx = sort(x)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    # 1-d chunk
    raw = np.random.rand(100)
    x = tensor(raw, chunk_size=20)

    sx = sort(x)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    # test force need_align=True
    sx = sort(x)
    sx.op._need_align = True

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    # test psrs_kinds
    sx = sort(x, psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    # structured dtype
    raw = np.empty(100, dtype=[("id", np.int32), ("size", np.int64)])
    raw["id"] = np.random.randint(1000, size=100, dtype=np.int32)
    raw["size"] = np.random.randint(1000, size=100, dtype=np.int64)
    x = tensor(raw, chunk_size=10)

    sx = sort(x, order=["size", "id"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, order=["size", "id"]))

    # test psrs_kinds with structured dtype
    sx = sort(x, order=["size", "id"], psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, order=["size", "id"]))

    # test flatten case
    raw = np.random.rand(10, 10)
    x = tensor(raw, chunk_size=(5, 10))

    sx = sort(x, axis=None)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=None))

    # test multi-dimension
    raw = np.random.rand(10, 100)
    x = tensor(raw, chunk_size=(5, 40))

    sx = sort(x, psrs_kinds=["quicksort"] * 3)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    sx = sort(x, psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    raw = np.random.rand(10, 99)
    x = tensor(raw, chunk_size=(5, 20))

    sx = sort(x)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    # test 3-d
    raw = np.random.rand(20, 25, 28)
    x = tensor(raw, chunk_size=(10, 15, 14))

    sx = sort(x)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    sx = sort(x, psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    sx = sort(x, axis=0)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=0))

    sx = sort(x, axis=0, psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=0))

    sx = sort(x, axis=1)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=1))

    sx = sort(x, axis=1, psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=1))

    # test multi-dimension with structured type
    raw = np.empty((10, 100), dtype=[("id", np.int32), ("size", np.int64)])
    raw["id"] = np.random.randint(1000, size=(10, 100), dtype=np.int32)
    raw["size"] = np.random.randint(1000, size=(10, 100), dtype=np.int64)
    x = tensor(raw, chunk_size=(7, 30))

    sx = sort(x)

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw))

    sx = sort(x, order=["size", "id"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, order=["size", "id"]))

    sx = sort(x, order=["size"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, order=["size"]))

    sx = sort(x, axis=0, order=["size", "id"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=0, order=["size", "id"]))

    sx = sort(x, axis=0, order=["size", "id"], psrs_kinds=[None, None, "quicksort"])

    res = sx.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=0, order=["size", "id"]))

    # test inplace sort
    raw = np.random.rand(10, 12)
    a = tensor(raw, chunk_size=(5, 4))
    a.sort(axis=1)

    res = a.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw, axis=1))

    a.sort(axis=0)

    res = a.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(np.sort(raw, axis=1), axis=0))

    # test with empty chunk
    raw = np.random.rand(20, 10)
    raw[:, :8] = 1
    a = tensor(raw, chunk_size=5)
    filtered = a[a < 1]
    filtered.sort()

    res = filtered.execute().fetch()
    np.testing.assert_array_equal(res, np.sort(raw[raw < 1]))


@pytest.mark.ray_dag
def test_sort_indices_execution(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=20)

    r = sort(x, return_index=True)

    sr, si = r.execute().fetch()
    np.testing.assert_array_equal(sr, np.take_along_axis(raw, si, axis=-1))

    x = tensor(raw, chunk_size=(22, 4))

    r = sort(x, return_index=True)

    sr, si = r.execute().fetch()
    np.testing.assert_array_equal(sr, np.take_along_axis(raw, si, axis=-1))

    raw = np.random.rand(100)

    x = tensor(raw, chunk_size=23)

    r = sort(x, axis=0, return_index=True)

    sr, si = r.execute().fetch()
    np.testing.assert_array_equal(sr, raw[si])


@pytest.mark.ray_dag
def test_argsort(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=10)

    xa = argsort(x)

    r = xa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw), np.take_along_axis(raw, r, axis=-1))

    x = tensor(raw, chunk_size=(22, 4))

    xa = argsort(x)

    r = xa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw), np.take_along_axis(raw, r, axis=-1))

    raw = np.random.rand(100)

    x = tensor(raw, chunk_size=23)

    xa = argsort(x, axis=0)

    r = xa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw, axis=0), raw[r])


@pytest.mark.ray_dag
def test_partition_execution(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=20)

    px = partition(x, [1, 8])

    res = px.execute().fetch()
    np.testing.assert_array_equal(res, np.partition(raw, [1, 8]))

    # 1-d chunk
    raw = np.random.rand(100)
    x = tensor(raw, chunk_size=20)

    kth = np.random.RandomState(0).randint(-100, 100, size=(10,))
    px = partition(x, kth)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[kth], np.partition(raw, kth)[kth])

    # structured dtype
    raw = np.empty(100, dtype=[("id", np.int32), ("size", np.int64)])
    raw["id"] = np.random.randint(1000, size=100, dtype=np.int32)
    raw["size"] = np.random.randint(1000, size=100, dtype=np.int64)
    x = tensor(raw, chunk_size=20)

    px = partition(x, kth, order=["size", "id"])

    res = px.execute().fetch()
    np.testing.assert_array_equal(
        res[kth], np.partition(raw, kth, order=["size", "id"])[kth]
    )

    # test flatten case
    raw = np.random.rand(10, 10)
    x = tensor(raw, chunk_size=5)

    px = partition(x, kth, axis=None)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[kth], np.partition(raw, kth, axis=None)[kth])

    # test multi-dimension
    raw = np.random.rand(10, 100)
    x = tensor(raw, chunk_size=(5, 20))

    kth = np.random.RandomState(0).randint(-10, 10, size=(3,))
    px = partition(x, kth)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[:, kth], np.partition(raw, kth)[:, kth])

    raw = np.random.rand(10, 99)
    x = tensor(raw, chunk_size=(5, 20))

    px = partition(x, kth)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[:, kth], np.partition(raw, kth)[:, kth])

    # test 3-d
    raw = np.random.rand(20, 25, 28)
    x = tensor(raw, chunk_size=(10, 15, 14))

    kth = np.random.RandomState(0).randint(-28, 28, size=(3,))
    px = partition(x, kth)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[:, :, kth], np.partition(raw, kth)[:, :, kth])

    kth = np.random.RandomState(0).randint(-20, 20, size=(3,))
    px = partition(x, kth, axis=0)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[kth], np.partition(raw, kth, axis=0)[kth])

    kth = np.random.RandomState(0).randint(-25, 25, size=(3,))
    px = partition(x, kth, axis=1)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[:, kth], np.partition(raw, kth, axis=1)[:, kth])

    # test multi-dimension with structured type
    raw = np.empty((10, 100), dtype=[("id", np.int32), ("size", np.int64)])
    raw["id"] = np.random.randint(1000, size=(10, 100), dtype=np.int32)
    raw["size"] = np.random.randint(1000, size=(10, 100), dtype=np.int64)
    x = tensor(raw, chunk_size=(7, 30))

    kth = np.random.RandomState(0).randint(-100, 100, size=(10,))
    px = partition(x, kth)

    res = px.execute().fetch()
    np.testing.assert_array_equal(res[:, kth], np.partition(raw, kth)[:, kth])

    px = partition(x, kth, order=["size", "id"])

    res = px.execute().fetch()
    np.testing.assert_array_equal(
        res[:, kth], np.partition(raw, kth, order=["size", "id"])[:, kth]
    )

    px = partition(x, kth, order=["size"])

    res = px.execute().fetch()
    np.testing.assert_array_equal(
        res[:, kth], np.partition(raw, kth, order=["size"])[:, kth]
    )

    kth = np.random.RandomState(0).randint(-10, 10, size=(5,))
    px = partition(x, kth, axis=0, order=["size", "id"])

    res = px.execute().fetch()
    np.testing.assert_array_equal(
        res[kth], np.partition(raw, kth, axis=0, order=["size", "id"])[kth]
    )

    raw = np.random.rand(10, 12)
    a = tensor(raw, chunk_size=(5, 4))
    kth = np.random.RandomState(0).randint(-12, 12, size=(2,))
    a.partition(kth, axis=1)

    res = a.execute().fetch()
    np.testing.assert_array_equal(res[:, kth], np.partition(raw, kth, axis=1)[:, kth])

    kth = np.random.RandomState(0).randint(-10, 10, size=(2,))
    a.partition(kth, axis=0)

    raw_base = res
    res = a.execute().fetch()
    np.testing.assert_array_equal(res[kth], np.partition(raw_base, kth, axis=0)[kth])

    # test kth which is tensor
    raw = np.random.rand(10, 12)
    a = tensor(raw, chunk_size=(3, 5))
    kth = (mt.random.rand(5) * 24 - 12).astype(int)

    px = partition(a, kth)
    sx = sort(a)

    res = px.execute().fetch()
    kth_res = kth.execute().fetch()
    sort_res = sx.execute().fetch()
    np.testing.assert_array_equal(res[:, kth_res], sort_res[:, kth_res])

    a = tensor(raw, chunk_size=(10, 12))
    kth = (mt.random.rand(5) * 24 - 12).astype(int)

    px = partition(a, kth)
    sx = sort(a)

    res = px.execute().fetch()
    kth_res = kth.execute().fetch()
    sort_res = sx.execute().fetch()
    np.testing.assert_array_equal(res[:, kth_res], sort_res[:, kth_res])


@pytest.mark.ray_dag
def test_partition_indices_execution(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=10)

    kth = [2, 5, 9]
    r = partition(x, kth, return_index=True)

    pr, pi = r.execute().fetch()
    np.testing.assert_array_equal(pr, np.take_along_axis(raw, pi, axis=-1))
    np.testing.assert_array_equal(np.sort(raw)[:, kth], pr[:, kth])

    x = tensor(raw, chunk_size=(22, 4))

    r = partition(x, kth, return_index=True)

    pr, pi = r.execute().fetch()
    np.testing.assert_array_equal(pr, np.take_along_axis(raw, pi, axis=-1))
    np.testing.assert_array_equal(np.sort(raw)[:, kth], pr[:, kth])

    raw = np.random.rand(100)

    x = tensor(raw, chunk_size=23)

    r = partition(x, kth, axis=0, return_index=True)

    pr, pi = r.execute().fetch()
    np.testing.assert_array_equal(pr, np.take_along_axis(raw, pi, axis=-1))
    np.testing.assert_array_equal(np.sort(raw)[kth], pr[kth])


@pytest.mark.ray_dag
def test_argpartition_execution(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=10)

    kth = [6, 3, 8]
    pa = argpartition(x, kth)

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, kth], np.take_along_axis(raw, r, axis=-1)[:, kth]
    )

    x = tensor(raw, chunk_size=(22, 4))

    pa = argpartition(x, kth)

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, kth], np.take_along_axis(raw, r, axis=-1)[:, kth]
    )

    raw = np.random.rand(100)

    x = tensor(raw, chunk_size=23)

    pa = argpartition(x, kth, axis=0)

    r = pa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw, axis=0)[kth], raw[r][kth])


def _topk_slow(a, k, axis, largest, order):
    if axis is None:
        a = a.flatten()
        axis = 0
    a = np.sort(a, axis=axis, order=order)
    if largest:
        a = a[(slice(None),) * axis + (slice(None, None, -1),)]
    return a[(slice(None),) * axis + (slice(k),)]


def _handle_result(result, axis, largest, order):
    result = np.sort(result, axis=axis, order=order)
    if largest:
        ax = axis if axis is not None else 0
        result = result[(slice(None),) * ax + (slice(None, None, -1),)]
    return result


@pytest.mark.parametrize("chunk_size", [7, 4])
@pytest.mark.parametrize("axis", [0, 1, 2, None])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("to_sort", [True, False])
@pytest.mark.parametrize("parallel_kind", ["tree", "psrs"])
def test_topk_execution(setup, chunk_size, axis, largest, to_sort, parallel_kind):
    raw1, order1 = np.random.rand(5, 6, 7), None
    raw2 = np.empty((5, 6, 7), dtype=[("a", np.int32), ("b", np.float64)])
    raw2["a"] = np.random.randint(1000, size=(5, 6, 7), dtype=np.int32)
    raw2["b"] = np.random.rand(5, 6, 7)
    order2 = ["b", "a"]

    for raw, order in [(raw1, order1), (raw2, order2)]:
        a = tensor(raw, chunk_size=chunk_size)
        size = raw.shape[axis] if axis is not None else raw.size
        for k in [2, size - 2, size, size + 2]:
            r = topk(
                a,
                k,
                axis=axis,
                largest=largest,
                sorted=to_sort,
                order=order,
                parallel_kind=parallel_kind,
            )

            result = r.execute().fetch()

            if not to_sort:
                result = _handle_result(result, axis, largest, order)
            expected = _topk_slow(raw, k, axis, largest, order)
            np.testing.assert_array_equal(result, expected)

            r = topk(
                a,
                k,
                axis=axis,
                largest=largest,
                sorted=to_sort,
                order=order,
                parallel_kind=parallel_kind,
                return_index=True,
            )

            ta, ti = r.execute().fetch()
            raw2 = raw
            if axis is None:
                raw2 = raw.flatten()
            np.testing.assert_array_equal(ta, np.take_along_axis(raw2, ti, axis))
            if not to_sort:
                ta = _handle_result(ta, axis, largest, order)
            np.testing.assert_array_equal(ta, expected)


def test_argtopk(setup):
    # only 1 chunk when axis = -1
    raw = np.random.rand(100, 10)
    x = tensor(raw, chunk_size=20)

    pa = argtopk(x, 3, parallel_kind="tree")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, -1:-4:-1], np.take_along_axis(raw, r, axis=-1)
    )

    pa = argtopk(x, 3, parallel_kind="psrs")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, -1:-4:-1], np.take_along_axis(raw, r, axis=-1)
    )

    x = tensor(raw, chunk_size=(22, 4))

    pa = argtopk(x, 3, parallel_kind="tree")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, -1:-4:-1], np.take_along_axis(raw, r, axis=-1)
    )

    pa = argtopk(x, 3, parallel_kind="psrs")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(
        np.sort(raw)[:, -1:-4:-1], np.take_along_axis(raw, r, axis=-1)
    )

    raw = np.random.rand(100)

    x = tensor(raw, chunk_size=23)

    pa = argtopk(x, 3, axis=0, parallel_kind="tree")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw, axis=0)[-1:-4:-1], raw[r])

    pa = argtopk(x, 3, axis=0, parallel_kind="psrs")

    r = pa.execute().fetch()
    np.testing.assert_array_equal(np.sort(raw, axis=0)[-1:-4:-1], raw[r])


def test_copy(setup):
    x = tensor([1, 2, 3])
    y = mt.copy(x)
    z = x

    x[0] = 10
    y_res = y.execute().fetch()
    np.testing.assert_array_equal(y_res, np.array([1, 2, 3]))

    z_res = z.execute().fetch()
    np.testing.assert_array_equal(z_res, np.array([10, 2, 3]))


def test_trapz_execution(setup):
    raws = [np.random.rand(10), np.random.rand(10, 3)]

    for raw in raws:
        for chunk_size in (4, 10):
            for dx in (1.0, 2.0):
                t = tensor(raw, chunk_size=chunk_size)
                r = trapz(t, dx=dx)

                result = r.execute().fetch()
                expected = np.trapz(raw, dx=dx)
                np.testing.assert_almost_equal(
                    result,
                    expected,
                    err_msg=f"failed when raw={raw}, "
                    f"chunk_size={chunk_size}, dx={dx}",
                )

    # test x not None
    raw_ys = [np.random.rand(10), np.random.rand(10, 3)]
    raw_xs = [np.random.rand(10), np.random.rand(10, 3)]

    for raw_y, raw_x in zip(raw_ys, raw_xs):
        ys = [tensor(raw_y, chunk_size=5), tensor(raw_y, chunk_size=10)]
        x = tensor(raw_x, chunk_size=4)

        for y in ys:
            r = trapz(y, x=x)

            result = r.execute().fetch()
            expected = np.trapz(raw_y, x=raw_x)
            np.testing.assert_almost_equal(result, expected)


@pytest.mark.ray_dag
def test_shape(setup):
    raw = np.random.RandomState(0).rand(4, 3)
    x = mt.tensor(raw, chunk_size=2)

    s = shape(x)

    result = s.execute().fetch()
    assert result == [4, 3]

    s = shape(x[x > 0.5])

    result = s.execute().fetch()
    expected = np.shape(raw[raw > 0.5])
    assert result == expected

    s = shape(0)

    result = s.execute().fetch()
    expected = np.shape(0)
    assert result == expected


@pytest.mark.ray_dag
def test_rebalance_execution(setup):
    session = setup

    raw = np.random.rand(10, 3)
    x = mt.tensor(raw)

    r = x.rebalance(num_partitions=3)
    result = r.execute().fetch()
    np.testing.assert_array_equal(result, raw)
    assert len(session._session._tileable_to_fetch[r.data].chunks) == 3

    r = x.rebalance(factor=1.5)
    result = r.execute().fetch()
    np.testing.assert_array_equal(result, raw)

    r = x.rebalance()
    result = r.execute().fetch()
    np.testing.assert_array_equal(result, raw)
    assert len(session._session._tileable_to_fetch[r.data].chunks) == 2


def test_map_chunk_execution(setup):
    raw = np.random.rand(20)
    a = tensor(raw, chunk_size=10)

    r = a.map_chunk(lambda x: x * 0.5)
    results = r.execute().fetch()
    np.testing.assert_array_equal(raw * 0.5, results)

    r = a.map_chunk(lambda x: x * 0.5, elementwise=True)
    results = r.execute().fetch()
    np.testing.assert_array_equal(raw * 0.5, results)

    r = a.map_chunk(
        lambda x, chunk_index: x * 0.5 + chunk_index[0], with_chunk_index=True
    )
    results = r.execute().fetch()
    np.testing.assert_array_equal(raw * 0.5 + np.arange(0, 20) // 10, results)


def test_insert_execution(setup):
    raw = np.random.randint(0, 100, size=(20, 10))
    a = tensor(raw, chunk_size=6)

    r1 = mt.insert(a, 1, 5)
    result = r1.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, 1, 5), result)

    r2 = mt.insert(a, [3, 50, 10], 10)
    result = r2.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [3, 50, 10], 10), result)

    r3 = mt.insert(a, [2, 3, 4], [5, 6, 7])
    result = r3.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [2, 3, 4], [5, 6, 7]), result)

    # specify axis
    r4 = mt.insert(a, 5, 4, axis=0)
    result = r4.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, 5, 4, axis=0), result)

    r5 = mt.insert(a, [1, 2, 6], np.arange(20).reshape((20, 1)), axis=1)
    result = r5.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 2, 6], np.arange(20).reshape((20, 1)), axis=1), result
    )

    r6 = mt.insert(a, [1, 16, 10], np.arange(30).reshape((3, 10)), axis=0)
    result = r6.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 16, 10], np.arange(30).reshape((3, 10)), axis=0), result
    )

    # test mt.tensor as values
    r5 = mt.insert(a, [1, 2, 6], mt.arange(20).reshape((20, 1)), axis=1)
    result = r5.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 2, 6], np.arange(20).reshape((20, 1)), axis=1), result
    )

    r6 = mt.insert(a, [1, 16, 10], mt.arange(30).reshape((3, 10)), axis=0)
    result = r6.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 16, 10], np.arange(30).reshape((3, 10)), axis=0), result
    )

    r7 = mt.insert(a, [20, 30, 50], mt.tensor([5, 6, 7]))
    result = r7.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [20, 30, 50], [5, 6, 7]), result)

    # test mt.tensor as index
    r8 = mt.insert(a, mt.tensor([1, 2, 6]), mt.arange(20).reshape((20, 1)), axis=1)
    result = r8.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 2, 6], np.arange(20).reshape((20, 1)), axis=1), result
    )

    r9 = mt.insert(a, mt.tensor([1, 16, 10]), mt.arange(30).reshape((3, 10)), axis=0)
    result = r9.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, [1, 16, 10], np.arange(30).reshape((3, 10)), axis=0), result
    )

    r10 = mt.insert(a, mt.tensor([20, 30, 50]), mt.tensor([5, 6, 7]))
    result = r10.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [20, 30, 50], [5, 6, 7]), result)

    r11 = mt.insert(a, slice(0, 10), mt.arange(10), axis=0)
    result = r11.execute().fetch()
    np.testing.assert_array_equal(
        np.insert(raw, slice(0, 10), np.arange(10), axis=0), result
    )

    r12 = mt.insert(a, 10, 5, axis=1)
    result = r12.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, 10, 5, axis=1), result)

    r13 = mt.insert(a, [2, 10], 5, axis=1)
    result = r13.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [2, 10], 5, axis=1), result)

    r14 = mt.insert(a, mt.tensor([2, 20]), 5, axis=0)
    result = r14.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, [2, 20], 5, axis=0), result)

    r15 = mt.insert(a, 7, mt.arange(20), axis=1)
    result = r15.execute().fetch()
    np.testing.assert_array_equal(np.insert(raw, 7, mt.arange(20), axis=1), result)


def test_delete_execution(setup):
    raw = np.random.randint(0, 100, size=(20, 10))
    a = tensor(raw, chunk_size=6)

    r1 = mt.delete(a, 1)
    result = r1.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, 1), result)

    r2 = mt.delete(a, [3, 50, 10])
    result = r2.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, [3, 50, 10]), result)

    # specify axis
    r4 = mt.delete(a, 5, axis=0)
    result = r4.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, 5, axis=0), result)

    r5 = mt.delete(a, [1, 2, 6], axis=1)
    result = r5.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, [1, 2, 6], axis=1), result)

    r6 = mt.delete(a, mt.tensor([1, 2, 6, 8], chunk_size=3), axis=1)
    result = r6.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, [1, 2, 6, 8], axis=1), result)

    r7 = mt.delete(a, slice(0, 10), axis=0)
    result = r7.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, slice(0, 10), axis=0), result)

    r8 = mt.delete(a, mt.tensor([10, 20, 6, 80]))
    result = r8.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, [10, 20, 6, 80]), result)

    r9 = mt.delete(a, 9, axis=1)
    result = r9.execute().fetch()
    np.testing.assert_array_equal(np.delete(raw, 9, axis=1), result)


@pytest.mark.parametrize("chunk_size", [3, 5])
@pytest.mark.parametrize("invert", [True, False])
def test_in1d_execute(setup, chunk_size, invert):
    rs = np.random.RandomState(0)
    raw1 = rs.randint(10, size=10)
    ar1 = mt.tensor(raw1, chunk_size=5)
    raw2 = np.arange(5)
    ar2 = mt.tensor(raw2, chunk_size=chunk_size)
    ar = mt.in1d(ar1, ar2, invert=invert)
    result = ar.execute().fetch()
    expected = np.in1d(raw1, raw2, invert=invert)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("chunk_size", [3, 5])
def test_setdiff1d_execute(setup, chunk_size):
    rs = np.random.RandomState(0)
    raw1 = rs.randint(10, size=10)
    ar1 = mt.tensor(raw1, chunk_size=5)
    raw2 = np.arange(5)
    ar2 = mt.tensor(raw2, chunk_size=chunk_size)
    ar = mt.setdiff1d(ar1, ar2)
    result = ar.execute().fetch()
    expected = np.setdiff1d(raw1, raw2)
    np.testing.assert_array_equal(result, expected)

    raw3 = rs.shuffle(rs.choice(np.arange(100), 10))
    ar3 = mt.tensor(raw3, chunk_size=5)
    ar = mt.setdiff1d(ar3, ar2, assume_unique=True)
    result = ar.execute().fetch()
    expected = np.setdiff1d(raw3, raw2, assume_unique=True)
    np.testing.assert_array_equal(result, expected)
