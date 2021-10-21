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
from ....core.operand import OperandStage
from ...datasource import ones, tensor, arange
from .. import (
    transpose,
    broadcast_to,
    where,
    argwhere,
    array_split,
    split,
    squeeze,
    result_type,
    repeat,
    copyto,
    isin,
    moveaxis,
    TensorCopyTo,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    ravel,
    searchsorted,
    unique,
    sort,
    partition,
    topk,
    to_gpu,
    to_cpu,
)


def test_dir():
    a = tensor([0, 1, 2], chunk_size=2)
    tensor_dir = dir(a)
    for attr in dir(a.data):
        assert attr in tensor_dir


def test_copyto():
    a = ones((10, 20), chunk_size=3)
    b = ones(10, chunk_size=4)

    with pytest.raises(ValueError):
        copyto(a, b)

    tp = type(a.op)
    b = ones(20, chunk_size=4)
    copyto(a, b)

    assert isinstance(a.op, TensorCopyTo)
    assert a.inputs[0] is b.data
    assert isinstance(a.inputs[1].op, tp)

    a = tile(a)

    assert isinstance(a.chunks[0].op, TensorCopyTo)
    assert len(a.chunks[0].inputs) == 2

    a = ones((10, 20), chunk_size=3, dtype="i4")
    b = ones(20, chunk_size=4, dtype="f8")

    with pytest.raises(TypeError):
        copyto(a, b)

    b = ones(20, chunk_size=4, dtype="i4")
    copyto(a, b, where=b > 0)

    assert a.op.where is not None

    a = tile(a)

    assert isinstance(a.chunks[0].op, TensorCopyTo)
    assert len(a.chunks[0].inputs) == 3

    with pytest.raises(ValueError):
        copyto(a, a, where=np.ones(30, dtype="?"))


def test_astype():
    arr = ones((10, 20, 30), chunk_size=3)

    arr2 = arr.astype(np.int32)
    arr2 = tile(arr2)

    assert arr2.shape == (10, 20, 30)
    assert np.issubdtype(arr2.dtype, np.int32) is True
    assert arr2.op.casting == "unsafe"

    with pytest.raises(TypeError):
        arr.astype(np.int32, casting="safe")

    arr3 = arr.astype(arr.dtype, order="F")
    assert arr3.flags["F_CONTIGUOUS"] is True
    assert arr3.flags["C_CONTIGUOUS"] is False

    arr3 = tile(arr3)

    assert arr3.chunks[0].order.value == "F"


def test_transpose():
    arr = ones((10, 20, 30), chunk_size=[4, 3, 5])

    arr2 = transpose(arr)
    arr2 = tile(arr2)

    assert arr2.shape == (30, 20, 10)
    assert len(arr2.chunks) == 126
    assert arr2.chunks[0].shape == (5, 3, 4)
    assert arr2.chunks[-1].shape == (5, 2, 2)

    with pytest.raises(ValueError):
        transpose(arr, axes=(1, 0))

    arr3 = transpose(arr, (-2, 2, 0))
    arr3 = tile(arr3)

    assert arr3.shape == (20, 30, 10)
    assert len(arr3.chunks) == 126
    assert arr3.chunks[0].shape == (3, 5, 4)
    assert arr3.chunks[-1].shape == (2, 5, 2)

    arr4 = arr.transpose(-2, 2, 0)
    arr4 = tile(arr4)

    assert arr4.shape == (20, 30, 10)
    assert len(arr4.chunks) == 126
    assert arr4.chunks[0].shape == (3, 5, 4)
    assert arr4.chunks[-1].shape == (2, 5, 2)

    arr5 = arr.T
    arr5 = tile(arr5)

    assert arr5.shape == (30, 20, 10)
    assert len(arr5.chunks) == 126
    assert arr5.chunks[0].shape == (5, 3, 4)
    assert arr5.chunks[-1].shape == (5, 2, 2)


def test_swapaxes():
    arr = ones((10, 20, 30), chunk_size=[4, 3, 5])
    arr2 = arr.swapaxes(0, 1)
    arr, arr2 = tile(arr, arr2)

    assert arr2.shape == (20, 10, 30)
    assert len(arr.chunks) == len(arr2.chunks)


def test_broadcast_to():
    arr = ones((10, 5), chunk_size=2)
    arr2 = broadcast_to(arr, (20, 10, 5))
    arr, arr2 = tile(arr, arr2)

    assert arr2.shape == (20, 10, 5)
    assert len(arr2.chunks) == len(arr.chunks)
    assert arr2.chunks[0].shape == (20, 2, 2)

    arr = ones((10, 5, 1), chunk_size=2)
    arr3 = broadcast_to(arr, (5, 10, 5, 6))
    arr, arr3 = tile(arr, arr3)

    assert arr3.shape == (5, 10, 5, 6)
    assert len(arr3.chunks) == len(arr.chunks)
    assert arr3.nsplits == ((5,), (2, 2, 2, 2, 2), (2, 2, 1), (6,))
    assert arr3.chunks[0].shape == (5, 2, 2, 6)

    arr = ones((10, 1), chunk_size=2)
    arr4 = broadcast_to(arr, (20, 10, 5))
    arr, arr4 = tile(arr, arr4)

    assert arr4.shape == (20, 10, 5)
    assert len(arr4.chunks) == len(arr.chunks)
    assert arr4.chunks[0].shape == (20, 2, 5)

    with pytest.raises(ValueError):
        broadcast_to(arr, (10,))

    with pytest.raises(ValueError):
        broadcast_to(arr, (5, 1))

    arr = ones((4, 5), chunk_size=2)
    with pytest.raises((ValueError)):
        broadcast_to(arr[arr < 2], (3, 20))


def test_where():
    cond = tensor([[True, False], [False, True]], chunk_size=1)
    x = tensor([1, 2], chunk_size=1)
    y = tensor([3, 4], chunk_size=1)

    arr = where(cond, x, y)
    arr = tile(arr)

    assert len(arr.chunks) == 4
    np.testing.assert_equal(arr.chunks[0].inputs[0].op.data, [[True]])
    np.testing.assert_equal(arr.chunks[0].inputs[1].op.data, [1])
    np.testing.assert_equal(arr.chunks[0].inputs[2].op.data, [3])
    np.testing.assert_equal(arr.chunks[1].inputs[0].op.data, [[False]])
    np.testing.assert_equal(arr.chunks[1].inputs[1].op.data, [2])
    np.testing.assert_equal(arr.chunks[1].inputs[2].op.data, [4])
    np.testing.assert_equal(arr.chunks[2].inputs[0].op.data, [[False]])
    np.testing.assert_equal(arr.chunks[2].inputs[1].op.data, [1])
    np.testing.assert_equal(arr.chunks[2].inputs[2].op.data, [3])
    np.testing.assert_equal(arr.chunks[3].inputs[0].op.data, [[True]])
    np.testing.assert_equal(arr.chunks[3].inputs[1].op.data, [2])
    np.testing.assert_equal(arr.chunks[3].inputs[2].op.data, [4])

    with pytest.raises(ValueError):
        where(cond, x)

    x = arange(9.0).reshape(3, 3)
    y = where(x < 5, x, -1)

    assert y.dtype == np.float64


def test_argwhere():
    cond = tensor([[True, False], [False, True]], chunk_size=1)
    indices = argwhere(cond)

    assert np.isnan(indices.shape[0])
    assert indices.shape[1] == 2

    indices = tile(indices)

    assert indices.nsplits[1] == (1, 1)


def test_argwhere_order():
    data = np.asfortranarray([[True, False], [False, True]])
    cond = tensor(data, chunk_size=1)
    indices = argwhere(cond)

    assert indices.flags["F_CONTIGUOUS"] is True
    assert indices.flags["C_CONTIGUOUS"] is False

    indices = tile(indices)

    assert indices.chunks[0].order.value == "F"


def test_array_split():
    a = arange(8, chunk_size=2)

    splits = array_split(a, 3)
    assert len(splits) == 3
    assert [s.shape[0] for s in splits] == [3, 3, 2]

    splits = tile(*splits)
    assert splits[0].nsplits == ((2, 1),)
    assert splits[1].nsplits == ((1, 2),)
    assert splits[2].nsplits == ((2,),)

    a = arange(7, chunk_size=2)

    splits = array_split(a, 3)
    assert len(splits) == 3
    assert [s.shape[0] for s in splits] == [3, 2, 2]

    splits = tile(*splits)
    assert splits[0].nsplits == ((2, 1),)
    assert splits[1].nsplits == ((1, 1),)
    assert splits[2].nsplits == ((1, 1),)


def test_split():
    a = arange(9, chunk_size=2)

    splits = split(a, 3)
    assert len(splits) == 3
    assert all(s.shape == (3,) for s in splits) is True

    splits = tile(*splits)
    assert splits[0].nsplits == ((2, 1),)
    assert splits[1].nsplits == ((1, 2),)
    assert splits[2].nsplits == ((2, 1),)

    a = arange(8, chunk_size=2)

    splits = split(a, [3, 5, 6, 10])
    assert len(splits) == 5
    assert splits[0].shape == (3,)
    assert splits[1].shape == (2,)
    assert splits[2].shape == (1,)
    assert splits[3].shape == (2,)
    assert splits[4].shape == (0,)

    splits = tile(*splits)
    assert splits[0].nsplits == ((2, 1),)
    assert splits[1].nsplits == ((1, 1),)
    assert splits[2].nsplits == ((1,),)
    assert splits[3].nsplits == ((2,),)
    assert splits[4].nsplits == ((0,),)

    a = tensor(np.asfortranarray(np.random.rand(9, 10)), chunk_size=4)
    splits = split(a, 3)
    assert splits[0].flags["F_CONTIGUOUS"] is True
    assert splits[0].flags["C_CONTIGUOUS"] is False
    assert splits[1].flags["F_CONTIGUOUS"] is True
    assert splits[0].flags["C_CONTIGUOUS"] is False
    assert splits[2].flags["F_CONTIGUOUS"] is True
    assert splits[0].flags["C_CONTIGUOUS"] is False

    for a in ((1, 1, 1, 2, 2, 3), [1, 1, 1, 2, 2, 3]):
        splits = split(a, (3, 5))
        assert len(splits) == 3


def test_squeeze():
    data = np.array([[[0], [1], [2]]])
    x = tensor(data)

    t = squeeze(x)
    assert t.shape == (3,)
    assert t.dtype is not None

    t = squeeze(x, axis=0)
    assert t.shape == (3, 1)

    with pytest.raises(ValueError):
        squeeze(x, axis=1)

    t = squeeze(x, axis=2)
    assert t.shape == (1, 3)


def test_result_type():
    x = tensor([2, 3], dtype="i4")
    y = 3
    z = np.array([3, 4], dtype="f4")

    r = result_type(x, y, z)
    e = np.result_type(x.dtype, y, z)
    assert r == e


def test_repeat():
    a = arange(10, chunk_size=2).reshape(2, 5)

    t = repeat(a, 3)
    assert t.shape == (30,)

    t = repeat(a, 3, axis=0)
    assert t.shape == (6, 5)

    t = repeat(a, 3, axis=1)
    assert t.shape == (2, 15)

    t = repeat(a, [3], axis=1)
    assert t.shape == (2, 15)

    t = repeat(a, [3, 4], axis=0)
    assert t.shape == (7, 5)

    with pytest.raises(ValueError):
        repeat(a, [3, 4], axis=1)

    a = tensor(np.random.randn(10), chunk_size=5)

    t = repeat(a, 3)
    t = tile(t)
    assert sum(t.nsplits[0]) == 30

    a = tensor(np.random.randn(100), chunk_size=10)

    t = repeat(a, 3)
    t = tile(t)
    assert sum(t.nsplits[0]) == 300

    a = tensor(np.random.randn(4))
    b = tensor((4,))

    t = repeat(a, b)

    t = tile(t)
    assert np.isnan(t.nsplits[0])


def test_isin():
    element = 2 * arange(4, chunk_size=1).reshape(2, 2)
    test_elements = [1, 2, 4, 8]

    mask = isin(element, test_elements)
    assert mask.shape == (2, 2)
    assert mask.dtype == np.bool_

    mask, element = tile(mask, element)

    assert len(mask.chunks) == len(element.chunks)
    assert len(mask.op.inputs[1].chunks) == 1
    assert mask.chunks[0].inputs[0] is element.chunks[0].data

    element = 2 * arange(4, chunk_size=1).reshape(2, 2)
    test_elements = tensor([1, 2, 4, 8], chunk_size=2)

    mask = isin(element, test_elements, invert=True)
    assert mask.shape == (2, 2)
    assert mask.dtype == np.bool_


def test_create_view():
    arr = ones((10, 20, 30), chunk_size=[4, 3, 5])
    arr2 = transpose(arr)
    assert arr2.op.create_view is True

    arr3 = transpose(arr)
    assert arr3.op.create_view is True

    arr4 = arr.swapaxes(0, 1)
    assert arr4.op.create_view is True

    arr5 = moveaxis(arr, 1, 0)
    assert arr5.op.create_view is True

    arr6 = atleast_1d(1)
    assert arr6.op.create_view is True

    arr7 = atleast_2d([1, 1])
    assert arr7.op.create_view is True

    arr8 = atleast_3d([1, 1])
    assert arr8.op.create_view is True

    arr9 = arr[:3, [1, 2, 3]]
    # no view cuz of fancy indexing
    assert arr9.op.create_view is False

    arr9[0][0][0] = 100
    assert arr9.op.create_view is False

    arr10 = arr[:3, None, :5]
    assert arr10.op.create_view is True

    arr10[0][0][0] = 100
    assert arr10.op.create_view is False

    data = np.array([[[0], [1], [2]]])
    x = tensor(data)

    t = squeeze(x)
    assert t.op.create_view is True

    y = x.reshape(3)
    assert y.op.create_view is True


def test_ravel():
    arr = ones((10, 5), chunk_size=2)
    flat_arr = ravel(arr)
    assert flat_arr.shape == (50,)


def test_searchsorted():
    raw = np.sort(np.random.randint(100, size=(16,)))
    arr = tensor(raw, chunk_size=3).cumsum()

    t1 = searchsorted(arr, 10)

    assert t1.shape == ()
    assert (
        t1.flags["C_CONTIGUOUS"]
        == np.searchsorted(raw.cumsum(), 10).flags["C_CONTIGUOUS"]
    )
    assert (
        t1.flags["F_CONTIGUOUS"]
        == np.searchsorted(raw.cumsum(), 10).flags["F_CONTIGUOUS"]
    )

    t1 = tile(t1)

    assert t1.nsplits == ()
    assert len(t1.chunks) == 1
    assert t1.chunks[0].op.stage == OperandStage.agg

    with pytest.raises(ValueError):
        searchsorted(np.random.randint(10, size=(14, 14)), 1)

    with pytest.raises(ValueError):
        searchsorted(arr, 10, side="both")

    with pytest.raises(ValueError):
        searchsorted(arr.tosparse(), 10)

    raw2 = np.asfortranarray(np.sort(np.random.randint(100, size=(16,))))
    arr = tensor(raw2, chunk_size=3)
    to_search = np.asfortranarray([[1, 2], [3, 4]])

    t1 = searchsorted(arr, to_search)
    expected = np.searchsorted(raw2, to_search)

    assert t1.shape == to_search.shape
    assert t1.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert t1.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_to_gpu():
    x = tensor(np.random.rand(10, 10), chunk_size=3)

    gx = to_gpu(x)

    assert gx.dtype == x.dtype
    assert gx.order == x.order
    assert gx.op.gpu is True

    gx, x = tile(gx, x)

    assert gx.chunks[0].dtype == x.chunks[0].dtype
    assert gx.chunks[0].order == x.chunks[0].order
    assert gx.chunks[0].op.gpu is True


def test_to_cpu():
    x = tensor(np.random.rand(10, 10), chunk_size=3, gpu=True)

    cx = to_cpu(x)

    assert cx.dtype == x.dtype
    assert cx.order == x.order
    assert cx.op.gpu is False

    cx, x = tile(cx, x)

    assert cx.chunks[0].dtype == x.chunks[0].dtype
    assert cx.chunks[0].order == x.chunks[0].order
    assert cx.chunks[0].op.gpu is False


def test_unique():
    x = unique(np.int64(1))

    assert len(x.shape) == 1
    assert np.isnan(x.shape[0])
    assert x.dtype == np.dtype(np.int64)

    x = tile(x)

    assert len(x.chunks) == 1
    assert len(x.chunks[0].shape) == 1
    assert np.isnan(x.chunks[0].shape[0])
    assert x.chunks[0].dtype == np.dtype(np.int64)

    x, indices = unique(0.1, return_index=True)

    assert len(x.shape) == 1
    assert np.isnan(x.shape[0])
    assert x.dtype == np.dtype(np.float64)
    assert len(indices.shape) == 1
    assert np.isnan(indices.shape[0])
    assert indices.dtype == np.dtype(np.intp)

    x, indices = tile(x, indices)

    assert len(x.chunks) == 1
    assert len(x.chunks[0].shape) == 1
    assert np.isnan(x.chunks[0].shape[0])
    assert x.chunks[0].dtype == np.dtype(np.float64)
    assert len(indices.chunks) == 1
    assert len(indices.chunks[0].shape) == 1
    assert np.isnan(indices.chunks[0].shape[0])
    assert indices.chunks[0].dtype == np.dtype(np.intp)

    with pytest.raises(np.AxisError):
        unique(0.1, axis=1)

    raw = np.random.randint(10, size=(10), dtype=np.int64)
    a = tensor(raw, chunk_size=4)

    x = unique(a, aggregate_size=2)

    assert len(x.shape) == len(raw.shape)
    assert np.isnan(x.shape[0])
    assert x.dtype == np.dtype(np.int64)

    x = tile(x)

    assert len(x.chunks) == 2
    assert x.nsplits == ((np.nan, np.nan),)
    for i in range(2):
        assert x.chunks[i].shape == (np.nan,)
        assert x.chunks[i].dtype == raw.dtype

    raw = np.random.randint(10, size=(10, 20), dtype=np.int64)
    a = tensor(raw, chunk_size=(4, 6))

    x, indices, inverse, counts = unique(
        a,
        axis=1,
        aggregate_size=2,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    assert x.shape == (10, np.nan)
    assert x.dtype == np.dtype(np.int64)
    assert indices.shape == (np.nan,)
    assert indices.dtype == np.dtype(np.intp)
    assert inverse.shape == (20,)
    assert inverse.dtype == np.dtype(np.intp)
    assert counts.shape == (np.nan,)
    assert counts.dtype == np.dtype(np.int_)

    x, indices, inverse, counts = tile(x, indices, inverse, counts)

    assert len(x.chunks) == 2
    assert x.nsplits == ((10,), (np.nan, np.nan))
    for i in range(2):
        assert x.chunks[i].shape == (10, np.nan)
        assert x.chunks[i].dtype == raw.dtype
        assert x.chunks[i].index == (0, i)

    assert len(indices.chunks) == 2
    assert indices.nsplits == ((np.nan, np.nan),)
    for i in range(2):
        assert indices.chunks[i].shape == (np.nan,)
        assert indices.chunks[i].dtype == raw.dtype
        assert indices.chunks[i].index == (i,)

    assert len(inverse.chunks) == 4
    assert inverse.nsplits == ((6, 6, 6, 2),)
    for i in range(4):
        assert inverse.chunks[i].shape == ((6, 6, 6, 2)[i],)
        assert inverse.chunks[i].dtype == np.dtype(np.int64)
        assert inverse.chunks[i].index == (i,)

    assert len(counts.chunks) == 2
    assert counts.nsplits == ((np.nan, np.nan),)
    for i in range(2):
        assert counts.chunks[i].shape == (np.nan,)
        assert counts.chunks[i].dtype == np.dtype(np.int_)
        assert counts.chunks[i].index == (i,)


def test_sort():
    a = tensor(np.random.rand(10, 10), chunk_size=(5, 10))

    sa = sort(a)
    assert type(sa.op).__name__ == "TensorSort"

    sa = tile(sa)

    assert len(sa.chunks) == 2
    for c in sa.chunks:
        assert type(c.op).__name__ == "TensorSort"
        assert type(c.inputs[0].op).__name__ == "ArrayDataSource"

    a = tensor(np.random.rand(100), chunk_size=(10))

    sa = sort(a)
    assert type(sa.op).__name__ == "TensorSort"

    sa = tile(sa)

    for c in sa.chunks:
        assert type(c.op).__name__ == "PSRSShuffle"
        assert c.op.stage == OperandStage.reduce
        assert c.shape == (np.nan,)

    a = tensor(
        np.empty((10, 10), dtype=[("id", np.int32), ("size", np.int64)]),
        chunk_size=(10, 5),
    )
    sa = sort(a)
    assert sa.op.order == ["id", "size"]

    with pytest.raises(ValueError):
        sort(a, order=["unknown_field"])

    with pytest.raises(np.AxisError):
        sort(np.random.rand(100), axis=1)

    with pytest.raises(ValueError):
        sort(np.random.rand(100), kind="non_valid_kind")

    with pytest.raises(ValueError):
        sort(np.random.rand(100), parallel_kind="non_valid_parallel_kind")

    with pytest.raises(TypeError):
        sort(np.random.rand(100), psrs_kinds="non_valid_psrs_kinds")

    with pytest.raises(ValueError):
        sort(np.random.rand(100), psrs_kinds=["quicksort"] * 2)

    with pytest.raises(ValueError):
        sort(np.random.rand(100), psrs_kinds=["non_valid_kind"] * 3)

    with pytest.raises(ValueError):
        sort(np.random.rand(100), psrs_kinds=[None, None, None])

    with pytest.raises(ValueError):
        sort(np.random.rand(100), psrs_kinds=["quicksort", "mergesort", None])


def test_partition():
    a = tensor(np.random.rand(10, 10), chunk_size=(5, 10))

    pa = partition(a, [4, 9])
    assert type(pa.op).__name__ == "TensorPartition"

    pa = tile(pa)

    assert len(pa.chunks) == 2
    for c in pa.chunks:
        assert type(c.op).__name__ == "TensorPartition"
        assert type(c.inputs[0].op).__name__ == "ArrayDataSource"

    a = tensor(np.random.rand(100), chunk_size=(10))

    pa = partition(a, 4)
    assert type(pa.op).__name__ == "TensorPartition"

    pa = tile(pa)

    for c in pa.chunks:
        assert type(c.op).__name__ == "PartitionMerged"
        assert c.shape == (np.nan,)

    a = tensor(
        np.empty((10, 10), dtype=[("id", np.int32), ("size", np.int64)]),
        chunk_size=(10, 5),
    )
    pa = partition(a, 3)
    assert pa.op.order == ["id", "size"]

    with pytest.raises(ValueError):
        partition(a, 4, order=["unknown_field"])

    with pytest.raises(np.AxisError):
        partition(np.random.rand(100), 4, axis=1)

    with pytest.raises(ValueError):
        partition(np.random.rand(100), 4, kind="non_valid_kind")

    with pytest.raises(ValueError):
        partition(np.random.rand(10), 10)

    with pytest.raises(TypeError):
        partition(np.random.rand(10), tensor([1.0, 2.0]))

    with pytest.raises(ValueError):
        partition(np.random.rand(10), tensor([[1, 2]]))

    with pytest.raises(ValueError):
        partition(np.random.rand(10), [-11, 2])


def test_topk():
    raw = np.random.rand(20)
    a = tensor(raw, chunk_size=10)

    t = topk(a, 2)
    t = tile(t)
    assert t.op.parallel_kind == "tree"

    t = topk(a, 3)
    t = tile(t)
    assert t.op.parallel_kind == "psrs"

    t = topk(sort(a), 3)
    t = tile(t)
    # k is less than 100
    assert t.op.parallel_kind == "tree"

    with pytest.raises(ValueError):
        topk(a, 3, parallel_kind="unknown")


def test_map_chunk():
    raw = np.random.rand(20)
    a = tensor(raw, chunk_size=10)

    mapped = tile(a.map_chunk(lambda x: x * 0.5))
    assert np.issubdtype(mapped.dtype, np.floating) is True
    assert mapped.shape == (np.nan,)
    assert len(mapped.chunks) == 2

    mapped = tile(a.map_chunk(lambda x: x * 0.5, elementwise=True))
    assert np.issubdtype(mapped.dtype, np.floating) is True
    assert mapped.shape == (20,)
    assert len(mapped.chunks) == 2
