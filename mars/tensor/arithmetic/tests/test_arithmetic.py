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

from ....core import enter_mode, tile
from ...datasource import array, ones, tensor, empty
from ...fetch import TensorFetch
from ...linalg import matmul
from ...core import Tensor, SparseTensor
from .. import (
    add,
    subtract,
    truediv,
    log,
    frexp,
    around,
    isclose,
    isfinite,
    negative,
    cos,
    tree_add,
    tree_multiply,
    TensorAdd,
    TensorTreeAdd,
    TensorTreeMultiply,
    TensorSubtract,
    TensorLog,
    TensorIsclose,
    TensorGreaterThan,
)


def test_add():
    t1 = ones((3, 4), chunk_size=2)
    t2 = ones(4, chunk_size=2)
    t3 = t1 + t2
    k1 = t3.key
    assert t3.op.gpu is None
    t1, t2, t3 = tile(t1, t2, t3)
    assert t3.key != k1
    assert t3.shape == (3, 4)
    assert len(t3.chunks) == 4
    assert t3.chunks[0].inputs == [t1.chunks[0].data, t2.chunks[0].data]
    assert t3.chunks[1].inputs == [t1.chunks[1].data, t2.chunks[1].data]
    assert t3.chunks[2].inputs == [t1.chunks[2].data, t2.chunks[0].data]
    assert t3.chunks[3].inputs == [t1.chunks[3].data, t2.chunks[1].data]
    assert t3.op.dtype == np.dtype("f8")
    assert t3.chunks[0].op.dtype == np.dtype("f8")

    t1 = ones((3, 4), chunk_size=2)
    t4 = t1 + 1
    t1, t4 = tile(t1, t4)
    assert t4.shape == (3, 4)
    assert len(t3.chunks) == 4
    assert t4.chunks[0].inputs == [t1.chunks[0].data]
    assert t4.chunks[0].op.rhs == 1
    assert t4.chunks[1].inputs == [t1.chunks[1].data]
    assert t4.chunks[1].op.rhs == 1
    assert t4.chunks[2].inputs == [t1.chunks[2].data]
    assert t4.chunks[2].op.rhs == 1
    assert t4.chunks[3].inputs == [t1.chunks[3].data]
    assert t4.chunks[3].op.rhs == 1

    t5 = add([1, 2, 3, 4], 1)
    tile(t5)
    assert t4.chunks[0].inputs == [t1.chunks[0].data]

    t2 = ones(4, chunk_size=2)
    t6 = ones((3, 4), chunk_size=2, gpu=True)
    t7 = ones(4, chunk_size=2, gpu=True)
    t8 = t6 + t7
    t9 = t6 + t2
    assert t8.op.gpu is True
    t8, t9 = tile(t8, t9)
    assert t8.chunks[0].op.gpu is True
    assert t9.op.gpu is None
    assert t9.chunks[0].op.gpu is None

    # sparse tests
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 + 1
    assert t.op.gpu is None
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True

    t = t1 + 0
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 + t2
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 + t3
    assert t.issparse() is False
    assert type(t) is Tensor

    t = tile(t)
    assert t.chunks[0].op.sparse is False


def test_add_order():
    raw_a = np.random.rand(4, 2)
    raw_b = np.asfortranarray(np.random.rand(4, 2))
    t1 = tensor(raw_a)
    t2 = tensor(raw_b)
    out = tensor(raw_b)

    # C + scalar
    assert (t1 + 1).flags["C_CONTIGUOUS"] == (raw_a + 1).flags["C_CONTIGUOUS"]
    assert (t1 + 1).flags["F_CONTIGUOUS"] == (raw_a + 1).flags["F_CONTIGUOUS"]
    # C + C
    assert (t1 + t1).flags["C_CONTIGUOUS"] == (raw_a + raw_a).flags["C_CONTIGUOUS"]
    assert (t1 + t1).flags["F_CONTIGUOUS"] == (raw_a + raw_a).flags["F_CONTIGUOUS"]
    # F + scalar
    assert (t2 + 1).flags["C_CONTIGUOUS"] == (raw_b + 1).flags["C_CONTIGUOUS"]
    assert (t2 + 1).flags["F_CONTIGUOUS"] == (raw_b + 1).flags["F_CONTIGUOUS"]
    # F + F
    assert (t2 + t2).flags["C_CONTIGUOUS"] == (raw_b + raw_b).flags["C_CONTIGUOUS"]
    assert (t2 + t2).flags["F_CONTIGUOUS"] == (raw_b + raw_b).flags["F_CONTIGUOUS"]
    # C + F
    assert (t1 + t2).flags["C_CONTIGUOUS"] == (raw_a + raw_b).flags["C_CONTIGUOUS"]
    assert (t1 + t2).flags["F_CONTIGUOUS"] == (raw_a + raw_b).flags["F_CONTIGUOUS"]
    # C + C + out
    assert (
        add(t1, t1, out=out).flags["C_CONTIGUOUS"]
        == np.add(raw_a, raw_a, out=np.empty((4, 2), order="F")).flags["C_CONTIGUOUS"]
    )
    assert (
        add(t1, t1, out=out).flags["F_CONTIGUOUS"]
        == np.add(raw_a, raw_a, out=np.empty((4, 2), order="F")).flags["F_CONTIGUOUS"]
    )

    with pytest.raises(TypeError):
        add(t1, 1, order="B")


def test_multiply():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 * 10
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 * t2
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 * t3
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True


def test_divide():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = t1 / 10
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True

    t2 = tensor([[1, 0, 0]], chunk_size=2).tosparse()

    t = t1 / t2
    assert t.issparse() is False
    assert type(t) is Tensor

    t = tile(t)
    assert t.chunks[0].op.sparse is False

    t3 = tensor([1, 1, 1], chunk_size=2)
    t = t1 / t3
    assert t.issparse() is False
    assert type(t) is Tensor

    t = tile(t)
    assert t.chunks[0].op.sparse is False

    t = t3 / t1
    assert t.issparse() is False
    assert type(t) is Tensor

    t = tile(t)
    assert t.chunks[0].op.sparse is False


def test_datatime_arith():
    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    t2 = t1 + np.timedelta64(1)

    assert isinstance(t2.op, TensorAdd)

    t3 = t1 - np.datetime64("2005-02-02")

    assert isinstance(t3.op, TensorSubtract)
    assert (
        t3.dtype
        == (
            np.array(["2005-02-02", "2005-02-03"], dtype=np.datetime64)
            - np.datetime64("2005-02-02")
        ).dtype
    )

    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    subtract(t1, np.datetime64("2005-02-02"), out=empty(t1.shape, dtype=t3.dtype))

    t1 = array([np.datetime64("2005-02-02"), np.datetime64("2005-02-03")])
    add(t1, np.timedelta64(1, "D"), out=t1)


def test_add_with_out():
    t1 = ones((3, 4), chunk_size=2)
    t2 = ones(4, chunk_size=2)

    t3 = add(t1, t2, out=t1)

    assert isinstance(t1.op, TensorAdd)
    assert t1.op.out.key == t1.op.lhs.key
    assert t3 is t1
    assert t3.shape == (3, 4)
    assert t3.op.lhs.extra_params.raw_chunk_size == 2
    assert t3.op.rhs is t2.data
    assert t3.key != t3.op.lhs.key

    t1, t3 = tile(t1, t3)

    assert isinstance(t1.chunks[0].op, TensorAdd)
    assert t1.chunks[0].op.out.key == t1.chunks[0].op.lhs.key

    with pytest.raises(TypeError):
        add(t1, t2, out=1)

    with pytest.raises(ValueError):
        add(t1, t2, out=t2)

    with pytest.raises(TypeError):
        truediv(t1, t2, out=t1.astype("i8"))

    t1 = ones((3, 4), chunk_size=2, dtype=float)
    t2 = ones(4, chunk_size=2, dtype=int)

    t3 = add(t2, 1, out=t1)
    assert t3.shape == (3, 4)
    assert t3.dtype == np.float64


def test_dtype_from_out():
    x = array([-np.inf, 0.0, np.inf])
    y = array([2, 2, 2])

    t3 = isfinite(x, y)
    assert t3.dtype == y.dtype


def test_log_without_where():
    t1 = ones((3, 4), chunk_size=2)

    t2 = log(t1, out=t1)

    assert isinstance(t2.op, TensorLog)
    assert t1.op.out.key == t1.op.input.key
    assert t2 is t1
    assert t2.op.input.extra_params.raw_chunk_size == 2
    assert t2.key != t2.op.input.key

    t3 = empty((3, 4), chunk_size=2)
    t4 = log(t1, out=t3, where=t1 > 0)
    assert isinstance(t4.op, TensorLog)
    assert t4 is t3
    assert t2.op.input.extra_params.raw_chunk_size == 2
    assert t2.key != t2.op.input.key


def test_copy_add():
    t1 = ones((3, 4), chunk_size=2)
    t2 = ones(4, chunk_size=2)
    t3 = t1 + t2
    t3 = tile(t3)

    c = t3.chunks[0]
    inputs = (
        c.op.lhs,
        TensorFetch().new_chunk(
            c.op.rhs.inputs,
            shape=c.op.rhs.shape,
            index=c.op.rhs.index,
            _key=c.op.rhs.key,
        ),
    )
    new_c = c.op.copy().reset_key().new_chunk(inputs, shape=c.shape, _key="new_key")
    assert new_c.key == "new_key"
    assert new_c.inputs[1] is new_c.op.rhs
    assert isinstance(new_c.inputs[1].op, TensorFetch)


def test_compare():
    t1 = ones(4, chunk_size=2) * 2
    t2 = ones(4, chunk_size=2)
    t3 = t1 > t2
    t3 = tile(t3)
    assert len(t3.chunks) == 2
    assert isinstance(t3.op, TensorGreaterThan)


def test_unify_chunk_add():
    t1 = ones(4, chunk_size=2)
    t2 = ones(1, chunk_size=1)

    t3 = t1 + t2
    t1, t2, t3 = tile(t1, t2, t3)

    assert len(t3.chunks) == 2
    assert t3.chunks[0].inputs[0] == t1.chunks[0].data
    assert t3.chunks[0].inputs[1] == t2.chunks[0].data
    assert t3.chunks[1].inputs[0] == t1.chunks[1].data
    assert t3.chunks[1].inputs[1] == t2.chunks[0].data


def test_frexp():
    t1 = ones((3, 4, 5), chunk_size=2)
    t2 = empty((3, 4, 5), dtype=np.float_, chunk_size=2)
    op_type = type(t1.op)

    o1, o2 = frexp(t1)

    assert o1.op is o2.op
    assert o1.dtype != o2.dtype

    o1, o2 = frexp(t1, t1)

    assert o1 is t1
    assert o1.inputs[0] is not t1
    assert isinstance(o1.inputs[0].op, op_type)
    assert o2.inputs[0] is not t1

    o1, o2 = frexp(t1, t2, where=t1 > 0)

    op_type = type(t2.op)
    assert o1 is t2
    assert o1.inputs[0] is not t1
    assert isinstance(o1.inputs[0].op, op_type)
    assert o2.inputs[0] is not t1


def test_frexp_order():
    raw1 = np.asfortranarray(np.random.rand(2, 4))
    t = tensor(raw1)
    o1 = tensor(np.random.rand(2, 4))

    o1, o2 = frexp(t, out1=o1)

    assert (
        o1.flags["C_CONTIGUOUS"]
        == np.frexp(raw1, np.empty((2, 4)))[0].flags["C_CONTIGUOUS"]
    )
    assert (
        o1.flags["F_CONTIGUOUS"]
        == np.frexp(raw1, np.empty((2, 4)))[0].flags["F_CONTIGUOUS"]
    )
    assert o2.flags["C_CONTIGUOUS"] == np.frexp(raw1)[1].flags["C_CONTIGUOUS"]
    assert o2.flags["F_CONTIGUOUS"] == np.frexp(raw1)[1].flags["F_CONTIGUOUS"]


def test_dtype():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    t = truediv(t1, 2, dtype="f8")

    assert t.dtype == np.float64

    with pytest.raises(TypeError):
        truediv(t1, 2, dtype="i4")


def test_negative():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = negative(t1)
    assert t.op.gpu is None
    assert t.issparse() is True
    assert type(t) is SparseTensor

    t = tile(t)
    assert t.chunks[0].op.sparse is True


def test_negative_order():
    raw1 = np.random.rand(4, 2)
    raw2 = np.asfortranarray(np.random.rand(4, 2))
    t1 = tensor(raw1)
    t2 = tensor(raw2)
    t3 = tensor(raw1)
    t4 = tensor(raw2)

    # C
    assert negative(t1).flags["C_CONTIGUOUS"] == np.negative(raw1).flags["C_CONTIGUOUS"]
    assert negative(t1).flags["F_CONTIGUOUS"] == np.negative(raw1).flags["F_CONTIGUOUS"]
    # F
    assert negative(t2).flags["C_CONTIGUOUS"] == np.negative(raw2).flags["C_CONTIGUOUS"]
    assert negative(t2).flags["F_CONTIGUOUS"] == np.negative(raw2).flags["F_CONTIGUOUS"]
    # C + out
    assert (
        negative(t1, out=t4).flags["C_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="F")).flags["C_CONTIGUOUS"]
    )
    assert (
        negative(t1, out=t4).flags["F_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="F")).flags["F_CONTIGUOUS"]
    )
    # F + out
    assert (
        negative(t2, out=t3).flags["C_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="C")).flags["C_CONTIGUOUS"]
    )
    assert (
        negative(t2, out=t3).flags["F_CONTIGUOUS"]
        == np.negative(raw1, out=np.empty((4, 2), order="C")).flags["F_CONTIGUOUS"]
    )

    with pytest.raises(TypeError):
        negative(t1, order="B")


def test_cos():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()

    t = cos(t1)
    assert t.issparse() is True
    assert type(t) is SparseTensor


def test_around():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    t = around(t1, decimals=3)

    assert t.issparse() is False
    assert t.op.decimals == 3

    t = tile(t)

    assert t.chunks[0].op.decimals == 3


def test_isclose():
    t1 = ones((2, 3), dtype="f4", chunk_size=2)

    atol = 1e-4
    rtol = 1e-5
    equal_nan = True

    t = isclose(t1, 2, atol=atol, rtol=rtol, equal_nan=equal_nan)

    assert isinstance(t.op, TensorIsclose)
    assert t.op.atol == atol
    assert t.op.rtol == rtol
    assert t.op.equal_nan == equal_nan

    t = tile(t)

    assert isinstance(t.chunks[0].op, TensorIsclose)
    assert t.chunks[0].op.atol == atol
    assert t.chunks[0].op.rtol == rtol
    assert t.chunks[0].op.equal_nan == equal_nan

    t1 = ones((2, 3), dtype="f4", chunk_size=2)
    t2 = ones((2, 3), dtype="f4", chunk_size=2)

    atol = 1e-4
    rtol = 1e-5
    equal_nan = True

    t = isclose(t1, t2, atol=atol, rtol=rtol, equal_nan=equal_nan)

    assert isinstance(t.op, TensorIsclose)
    assert t.op.atol == atol
    assert t.op.rtol == rtol
    assert t.op.equal_nan == equal_nan

    t = tile(t)

    assert isinstance(t.chunks[0].op, TensorIsclose)
    assert t.chunks[0].op.atol == atol
    assert t.chunks[0].op.rtol == rtol
    assert t.chunks[0].op.equal_nan == equal_nan


def test_matmul():
    a_data = [[1, 0], [0, 1]]
    b_data = [[4, 1], [2, 2]]

    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=1)

    t = matmul(a, b)

    assert t.shape == (2, 2)
    t = tile(t)
    assert t.shape == tuple(sum(s) for s in t.nsplits)

    b_data = [1, 2]
    b = tensor(b_data, chunk_size=1)

    t = matmul(a, b)

    assert t.shape == (2,)
    t = tile(t)
    assert t.shape == tuple(sum(s) for s in t.nsplits)

    t = matmul(b, a)

    assert t.shape == (2,)
    t = tile(t)
    assert t.shape == tuple(sum(s) for s in t.nsplits)

    a_data = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    b_data = np.arange(2 * 2 * 4).reshape((2, 4, 2))

    a = tensor(a_data, chunk_size=1)
    b = tensor(b_data, chunk_size=1)

    t = matmul(a, b)

    assert t.shape == (2, 2, 2)
    t = tile(t)
    assert t.shape == tuple(sum(s) for s in t.nsplits)

    t = matmul(tensor([2j, 3j], chunk_size=1), tensor([2j, 3j], chunk_size=1))

    assert t.shape == ()
    t = tile(t)
    assert t.shape == tuple(sum(s) for s in t.nsplits)

    with pytest.raises(ValueError):
        matmul([1, 2], 3)

    with pytest.raises(ValueError):
        matmul(np.random.randn(2, 3, 4), np.random.randn(3, 4, 3))

    t = matmul(
        tensor(np.random.randn(2, 3, 4), chunk_size=2),
        tensor(np.random.randn(3, 1, 4, 3), chunk_size=3),
    )
    assert t.shape == (3, 2, 3, 3)

    v = ones((100, 100), chunk_size=10)
    tv = matmul(v, v)
    assert tv.shape == (100, 100)
    tv = tile(tv)
    assert tv.shape == tuple(sum(s) for s in tv.nsplits)


def test_tree_arithmetic():
    raws = [np.random.rand(10, 10) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    t = tree_add(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeAdd)
    assert t.issparse() is False
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    t = tree_multiply(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeMultiply)
    assert t.issparse() is False
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    raws = [sps.random(5, 9, density=0.1) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    t = tree_add(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeAdd)
    assert t.issparse() is True
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2

    t = tree_multiply(*tensors, combine_size=4)
    assert isinstance(t.op, TensorTreeMultiply)
    assert t.issparse() is True
    assert len(t.inputs) == 3
    assert len(t.inputs[0].inputs) == 4
    assert len(t.inputs[-1].inputs) == 2


def test_get_set_real():
    a_data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    a = tensor(a_data, chunk_size=2)

    with pytest.raises(ValueError):
        a.real = [2, 4]


def test_build_mode():
    t1 = ones((2, 3), chunk_size=2)
    assert t1 == 2

    with enter_mode(build=True):
        assert t1 != 2
