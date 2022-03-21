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

from ....config import option_context
from ....core import tile
from ...base.broadcast_to import TensorBroadcastTo
from ...datasource import ones, tensor, array, empty
from ...datasource.ones import TensorOnes
from ...merge.concatenate import TensorConcatenate
from .. import choose, unravel_index, nonzero, compress, fill_diagonal
from ..setitem import TensorIndexSetValue


def test_bool_indexing():
    t = ones((100, 200, 300))
    indexed = t[t < 2]
    assert len(indexed.shape) == 1
    assert np.isnan(indexed.shape[0])

    t2 = ones((100, 200))
    indexed = t[t2 < 2]
    assert len(indexed.shape) == 2
    assert np.isnan(indexed.shape[0])
    assert indexed.shape[1] == 300

    t2 = ones((100, 200))
    indexed = t[t2 < 2] + 1
    assert len(indexed.shape) == 2
    assert np.isnan(indexed.shape[0])
    assert indexed.shape[1] == 300

    t2 = ones((10, 20))
    rs = np.random.RandomState(0)
    i1 = np.zeros(10, dtype=bool)
    i1[rs.permutation(np.arange(10))[:5]] = True
    i2 = np.zeros(20, dtype=bool)
    i2[rs.permutation(np.arange(20))[:5]] = True
    indexed = t2[i1, i2]
    assert len(indexed.shape) == 1
    assert indexed.shape[0] == 5

    t2 = tile(indexed)
    assert t2.chunks[0].index == (0,)

    t3 = ones((101, 200))
    with pytest.raises(IndexError) as cm:
        _ = t[t3 < 2]  # noqa: F841
    e = cm.value.args[0]
    assert "along dimension 0" in e
    assert "dimension is 100 but corresponding boolean dimension is 101" in e

    t4 = ones((100, 201))
    with pytest.raises(IndexError) as cm:
        _ = t[t4 < 2]  # noqa: F841
    e = cm.value.args[0]
    assert "along dimension 1" in e
    assert "dimension is 200 but corresponding boolean dimension is 201" in e


def test_slice():
    t = ones((100, 200, 300))
    t2 = t[10:30, 199:, -30:303]
    assert t2.shape == (20, 1, 30)

    t3 = t[10:90:4, 20:80:5]
    s1 = len(list(range(100))[10:90:4])
    s2 = len(list(range(200))[20:80:5])
    assert t3.shape == (s1, s2, 300)


def test_fancy_indexing():
    t = ones((100, 200, 300))
    t2 = t[[0, 1], [2, 3]]
    assert t2.shape == (2, 300)

    t3 = t[[[0, 1], [2, 3]], [4, 5]]
    assert t3.shape == (2, 2, 300)

    with pytest.raises(IndexError) as cm:
        _ = t[[1, 2], [3, 4, 5]]  # noqa: F841
    e = cm.value.args[0]
    assert (
        e == "shape mismatch: indexing arrays could not be broadcast "
        "together with shapes (2,) (3,)"
    )

    with pytest.raises(IndexError):
        t[[100]]

    t = ones((100, 200, 300), chunk_size=10)

    # fancy index on numpy ndarrays

    t4 = tile(t[:10, -10:, [13, 244, 151, 242, 34]])
    assert t4.shape == (10, 10, 5)
    assert t4.chunk_shape == (1, 1, 1)

    t5 = tile(t[:10, -10:, [1, 10, 20, 33, 34, 200]])
    assert t5.shape == (10, 10, 6)
    assert t5.chunk_shape == (1, 1, 5)

    t6 = tile(t[[20, 1, 33, 22, 11], :15, [255, 211, 2, 11, 121]])
    assert t6.shape == (5, 15)
    # need a concat, because the fancy indexes are not ascending according to chunk index
    assert t6.chunk_shape == (1, 2)
    assert t6.chunks[0].ndim == 2
    assert t6.nsplits == ((5,), (10, 5))

    t7 = tile(t[[5, 6, 33, 66], :15, [0, 9, 2, 11]])
    assert t7.shape == (4, 15)
    # not need a concat
    assert t7.chunk_shape == (3, 2)
    assert t7.chunks[0].ndim == 2
    assert t7.nsplits == ((2, 1, 1), (10, 5))

    t8 = tile(t[[[5, 33], [66, 6]], :15, [255, 11]])
    assert t8.shape == (2, 2, 15)
    assert t8.chunk_shape == (1, 1, 2)
    assert t8.chunks[0].ndim == 3
    assert t8.nsplits == ((2,), (2,), (10, 5))

    # fancy index on tensors

    t9 = tile(t[:10, -10:, tensor([13, 244, 151, 242, 34], chunk_size=2)])
    assert t9.shape == (10, 10, 5)
    assert t9.chunk_shape == (1, 1, 3)

    t10 = tile(t[:10, -10:, tensor([1, 10, 20, 33, 34, 200], chunk_size=4)])
    assert t10.shape == (10, 10, 6)
    assert t10.chunk_shape == (1, 1, 2)

    t11 = tile(
        t[
            tensor([20, 1, 33, 22, 11], chunk_size=2),
            :15,
            tensor([255, 211, 2, 11, 121], chunk_size=3),
        ]
    )
    assert t11.shape == (5, 15)
    # need a concat, because the fancy indexes are not ascending according to chunk index
    assert t11.chunk_shape == (4, 2)
    assert t11.chunks[0].ndim == 2
    assert t11.nsplits == ((2, 1, 1, 1), (10, 5))

    t12 = tile(t[tensor([5, 6, 33, 66], chunk_size=2), :15, [0, 9, 2, 11]])
    assert t12.shape == (4, 15)
    # not need a concat
    assert t12.chunk_shape == (2, 2)
    assert t12.chunks[0].ndim == 2
    assert t12.nsplits == ((2, 2), (10, 5))

    t13 = tile(t[tensor([[5, 33], [66, 6]]), :15, tensor([255, 11])])
    assert t13.shape == (2, 2, 15)
    assert t13.chunk_shape == (1, 1, 2)
    assert t13.chunks[0].ndim == 3
    assert t13.nsplits == ((2,), (2,), (10, 5))


def test_mixed_indexing():
    t = ones((100, 200, 300, 400))

    with pytest.raises(IndexError):
        _ = t[ones((100, 200), dtype=float)]  # noqa: F841

    t2 = t[ones(100) < 2, ..., 20::101, 2]
    assert len(t2.shape) == 3
    assert np.isnan(t2.shape[0])

    t3 = ones((2, 3, 4, 5))
    t4 = t3[1]
    assert t4.flags["C_CONTIGUOUS"] == np.ones((2, 3, 4, 5))[1].flags["C_CONTIGUOUS"]
    assert t4.flags["F_CONTIGUOUS"] == np.ones((2, 3, 4, 5))[1].flags["F_CONTIGUOUS"]


def test_bool_indexing_tiles():
    t = ones((100, 200, 300), chunk_size=30)
    indexed = t[t < 2]
    indexed, t = tile(indexed, t)

    assert len(indexed.chunks) == 280
    assert indexed.chunks[0].index == (0,)
    assert indexed.chunks[20].index == (20,)
    assert indexed.chunks[20].inputs[0] is t.cix[(0, 2, 0)].data
    assert indexed.chunks[20].inputs[1] is indexed.op.indexes[0].cix[0, 2, 0].data

    t = ones((100, 200, 300), chunk_size=30)
    t2 = ones((100, 200), chunk_size=30)
    indexed2 = t[t2 < 2]
    indexed2, t = tile(indexed2, t)

    assert len(indexed2.chunks) == 280
    assert len(indexed2.chunks[0].shape) == 2
    assert np.isnan(indexed2.chunks[0].shape[0])
    assert indexed2.chunks[0].shape[1] == 30
    assert indexed2.chunks[20].inputs[0] == t.cix[(0, 2, 0)].data
    assert indexed2.chunks[20].inputs[1] == indexed2.op.indexes[0].cix[0, 2].data


def test_slice_tiles():
    t = ones((100, 200, 300), chunk_size=30)
    t2 = t[10:40, 199:, -30:303]
    t, t2 = tile(t, t2)

    assert t2.chunk_shape == (2, 1, 1)
    assert t2.chunks[0].inputs[0] == t.cix[0, -1, -1].data
    assert t2.chunks[0].op.indexes == [slice(10, 30, 1), slice(19, 20, 1), slice(None)]
    assert t2.chunks[0].index == (0, 0, 0)
    assert t2.chunks[1].inputs[0] == t.cix[1, -1, -1].data
    assert t2.chunks[1].op.indexes == [slice(0, 10, 1), slice(19, 20, 1), slice(None)]
    assert t2.chunks[1].index == (1, 0, 0)


def test_indices_indexing_tiles():
    t = ones((10, 20, 30), chunk_size=(2, 20, 30))
    t2 = t[3]
    t, t2 = tile(t, t2)

    assert len(t2.chunks) == 1
    assert t2.chunks[0].inputs[0] is t.cix[1, 0, 0].data
    assert t2.chunks[0].op.indexes[0] == 1

    t = ones((10, 20, 30), chunk_size=(2, 20, 30))
    t3 = t[4]
    t, t3 = tile(t, t3)

    assert len(t3.chunks) == 1
    assert t3.chunks[0].inputs[0] is t.cix[2, 0, 0].data
    assert t3.chunks[0].op.indexes[0] == 0


def test_mixed_indexing_tiles():
    t = ones((100, 200, 300, 400), chunk_size=24)

    cmp = ones(400, chunk_size=24) < 2
    t2 = t[10:90:3, 5, ..., None, cmp]
    t2, cmp = tile(t2, cmp)

    assert t2.shape[:-1] == (27, 300, 1)
    assert np.isnan(t2.shape[-1])
    assert t2.chunk_shape == (4, 13, 1, 17)
    assert t2.chunks[0].op.indexes == [
        slice(10, 24, 3),
        5,
        slice(None),
        None,
        cmp.cix[(0,)].data,
    ]


def test_setitem():
    shape = (10, 20, 30, 40)
    t = ones(shape, chunk_size=5, dtype="i4")
    t[5:20:3, 5, ..., :-5] = 2.2

    assert isinstance(t.op, TensorIndexSetValue)
    assert t.shape == shape
    assert isinstance(t.inputs[0].op.outputs[0].op, TensorOnes)

    t = tile(t)
    assert isinstance(t.chunks[0].op, TensorOnes)
    assert isinstance(t.cix[1, 1, 0, 0].op, TensorIndexSetValue)
    assert t.cix[1, 1, 0, 0].op.value == 2.2

    t2 = ones(shape, chunk_size=5, dtype="i4")
    shape = t2[5:20:3, 5, ..., :-5].shape
    t2[5:20:3, 5, ..., :-5] = ones(shape, chunk_size=4, dtype="i4") * 2

    t2 = tile(t2)
    assert isinstance(t2.chunks[0].op, TensorOnes)
    assert isinstance(t2.cix[1, 1, 0, 0].op, TensorIndexSetValue)
    assert isinstance(t2.cix[1, 1, 0, 0].op.value.op, TensorConcatenate)


def test_setitem_structured():
    # Check to value is properly broadcast for `setitem` on complex record dtype arrays.
    rec_type = np.dtype(
        [
            ("a", np.int32),
            ("b", np.double),
            ("c", np.dtype([("a", np.int16), ("b", np.int64)])),
        ]
    )

    t = ones((4, 5), dtype=rec_type, chunk_size=3)

    # assign tuple to record
    t[1:4, 1] = (3, 4.0, (5, 6))
    tt = tile(t)
    assert tt.cix[0, 0].op.value == (3, 4.0, (5, 6))

    # assign scalar to record
    t[1:4, 2] = 8
    tt = tile(t)
    assert tt.cix[0, 0].op.value == 8

    # assign scalar array to record array with broadcast
    t[1:3] = np.arange(5)
    tt = tile(t)
    slices_op = tt.cix[0, 0].op.value.op
    assert slices_op.slices == [slice(None, None, None), slice(0, 3, None)]
    broadcast_op = slices_op.inputs[0].op.inputs[0].op
    assert isinstance(broadcast_op, TensorBroadcastTo)
    assert broadcast_op.shape == (2, 5)
    np.testing.assert_array_equal(broadcast_op.inputs[0].op.data, np.arange(5))

    # assign scalar array to record array of same shape, no broadcast
    t[2:4] = np.arange(10).reshape(2, 5)
    tt = tile(t)
    slices_op = tt.cix[0, 0].op.value.op
    assert slices_op.slices == [slice(0, 1, None), slice(0, 3, None)]
    np.testing.assert_array_equal(
        slices_op.inputs[0].op.inputs[0].op.data, np.arange(10).reshape(2, 5)
    )


def test_choose():
    with option_context() as options:
        options.chunk_size = 2

        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        a = choose([2, 3, 1, 0], choices)

        a = tile(a)
        assert len(a.chunks) == 2
        assert isinstance(a.chunks[0].op, type(a.op))
        assert len(a.chunks[0].inputs) == 5

        with pytest.raises(TypeError):
            choose([2, 3, 1, 0], choices, out=1)

        with pytest.raises(ValueError):
            choose([2, 3, 1, 0], choices, out=tensor(np.empty((1, 4))))


def test_unravel_index():
    indices = tensor([22, 41, 37], chunk_size=1)
    t = unravel_index(indices, (7, 6))

    assert len(t) == 2

    t = [tile(r) for r in t]

    assert len(t[0].chunks) == 3
    assert len(t[1].chunks) == 3

    with pytest.raises(TypeError):
        unravel_index([22, 41, 37], (7, 6), order="B")


def test_nonzero():
    x = tensor([[1, 0, 0], [0, 2, 0], [1, 1, 0]], chunk_size=2)
    y = nonzero(x)

    assert len(y) == 2

    tile(y[0])


def test_compress():
    a = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(TypeError):
        compress([0, 1], a, axis=0, out=1)

    with pytest.raises(TypeError):
        compress(
            [0, 1],
            array([[1, 2], [3, 4], [5, 6]], dtype="i8"),
            axis=0,
            out=empty((1, 2), dtype="f8"),
        )


def test_operand_key():
    t = ones((10, 2), chunk_size=5)
    t_slice1 = t[:5]
    t_slice2 = t[5:]

    assert t_slice1.op.key != t_slice2.op.key


def test_fill_diagonal():
    a = tensor(np.random.rand(10, 13))
    fill_diagonal(a, 10)

    assert a.shape == (10, 13)

    # must be Tensor
    with pytest.raises(TypeError):
        fill_diagonal(np.random.rand(11, 10), 1)

    # at least 2-d required
    with pytest.raises(ValueError):
        a = tensor(np.random.rand(4))
        fill_diagonal(a, 1)

    # for more than 2-d, shape on each dimension should be equal
    with pytest.raises(ValueError):
        a = tensor(np.random.rand(11, 10, 11))
        fill_diagonal(a, 1)
