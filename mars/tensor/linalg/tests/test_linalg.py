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

from .... import tensor as mt
from ....core import tile
from ... import ones, tensor, dot, empty
from ...core import SparseTensor, Tensor
from .. import matmul
from ..inv import TensorInv


def test_qr():
    a = mt.random.rand(9, 6, chunk_size=(3, 6))
    q, r = mt.linalg.qr(a)

    assert q.shape == (9, 6)
    assert r.shape == (6, 6)

    q, r = tile(q, r)

    assert len(q.chunks) == 3
    assert len(r.chunks) == 1
    assert q.nsplits == ((3, 3, 3), (6,))
    assert r.nsplits == ((6,), (6,))

    assert q.chunks[0].shape == (3, 6)
    assert q.chunks[0].inputs[0].shape == (3, 3)
    assert q.chunks[0].inputs[1].shape == (3, 6)

    a = mt.random.rand(18, 6, chunk_size=(9, 6))
    q, r = mt.linalg.qr(a)

    assert q.shape == (18, 6)
    assert r.shape == (6, 6)

    q, r = tile(q, r)

    assert len(q.chunks) == 2
    assert len(r.chunks) == 1
    assert q.nsplits == ((9, 9), (6,))
    assert r.nsplits == ((6,), (6,))

    assert q.chunks[0].shape == (9, 6)
    assert q.chunks[0].inputs[0].shape == (9, 6)
    assert q.chunks[0].inputs[1].shape == (6, 6)

    # for Short-and-Fat QR
    a = mt.random.rand(6, 18, chunk_size=(6, 6))
    q, r = mt.linalg.qr(a, method="sfqr")

    assert q.shape == (6, 6)
    assert r.shape == (6, 18)

    q, r = tile(q, r)

    assert len(q.chunks) == 1
    assert len(r.chunks) == 3
    assert q.nsplits == ((6,), (6,))
    assert r.nsplits == ((6,), (6, 6, 6))

    # chunk width less than height
    a = mt.random.rand(6, 9, chunk_size=(6, 3))
    q, r = mt.linalg.qr(a, method="sfqr")

    assert q.shape == (6, 6)
    assert r.shape == (6, 9)

    q, r = tile(q, r)

    assert len(q.chunks) == 1
    assert len(r.chunks) == 2
    assert q.nsplits == ((6,), (6,))
    assert r.nsplits == ((6,), (6, 3))

    a = mt.random.rand(9, 6, chunk_size=(9, 3))
    q, r = mt.linalg.qr(a, method="sfqr")

    assert q.shape == (9, 6)
    assert r.shape == (6, 6)

    q, r = tile(q, r)

    assert len(q.chunks) == 1
    assert len(r.chunks) == 1
    assert q.nsplits == ((9,), (6,))
    assert r.nsplits == ((6,), (6,))


def test_norm():
    data = np.random.rand(9, 6)

    a = mt.tensor(data, chunk_size=(2, 6))

    for ord in (None, "nuc", np.inf, -np.inf, 0, 1, -1, 2, -2):
        for axis in (0, 1, (0, 1)):
            for keepdims in (True, False):
                try:
                    res = mt.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
                    expect_shape = np.linalg.norm(
                        data, ord=ord, axis=axis, keepdims=keepdims
                    ).shape
                    assert res.shape == expect_shape
                except ValueError:
                    continue


def test_svd():
    a = mt.random.rand(9, 6, chunk_size=(3, 6))
    U, s, V = mt.linalg.svd(a)

    assert U.shape == (9, 6)
    assert s.shape == (6,)
    assert V.shape == (6, 6)

    U, s, V = tile(U, s, V)

    assert len(U.chunks) == 3
    assert U.chunks[0].shape == (3, 6)
    assert len(s.chunks) == 1
    assert s.chunks[0].shape == (6,)
    assert len(V.chunks) == 1
    assert V.chunks[0].shape == (6, 6)

    assert U.chunks[0].inputs[0].shape == (3, 6)
    assert U.chunks[0].inputs[0].inputs[0].shape == (3, 3)
    assert U.chunks[0].inputs[0].inputs[1].shape == (3, 6)

    assert s.ndim == 1
    assert len(s.chunks[0].index) == 1

    a = mt.random.rand(9, 6, chunk_size=(9, 6))
    U, s, V = mt.linalg.svd(a)

    assert U.shape == (9, 6)
    assert s.shape == (6,)
    assert V.shape == (6, 6)

    U, s, V = tile(U, s, V)

    assert len(U.chunks) == 1
    assert U.chunks[0].shape == (9, 6)
    assert len(s.chunks) == 1
    assert s.chunks[0].shape == (6,)
    assert len(V.chunks) == 1
    assert V.chunks[0].shape == (6, 6)

    assert s.ndim == 1
    assert len(s.chunks[0].index) == 1

    a = mt.random.rand(6, 20, chunk_size=10)
    U, s, V = mt.linalg.svd(a)

    assert U.shape == (6, 6)
    assert s.shape == (6,)
    assert V.shape == (6, 20)

    U, s, V = tile(U, s, V)

    assert len(U.chunks) == 1
    assert U.chunks[0].shape == (6, 6)
    assert len(s.chunks) == 1
    assert s.chunks[0].shape == (6,)
    assert len(V.chunks) == 1
    assert V.chunks[0].shape == (6, 20)

    a = mt.random.rand(6, 9, chunk_size=(6, 9))
    U, s, V = mt.linalg.svd(a)

    assert U.shape == (6, 6)
    assert s.shape == (6,)
    assert V.shape == (6, 9)

    rs = mt.random.RandomState(1)

    a = rs.rand(20, 10, chunk_size=10)
    _, s, _ = mt.linalg.svd(a)
    del _
    graph = s.build_graph()
    assert len(graph) == 4


def test_lu():
    a = mt.random.randint(1, 10, (6, 6), chunk_size=3)
    p, l_, u = mt.linalg.lu(a)

    p, l_, u = tile(p, l_, u)

    assert l_.shape == (6, 6)
    assert u.shape == (6, 6)
    assert p.shape == (6, 6)

    a = mt.random.randint(1, 10, (6, 6), chunk_size=(3, 2))
    p, l_, u = mt.linalg.lu(a)
    p, l_, u = tile(p, l_, u)

    assert l_.shape == (6, 6)
    assert u.shape == (6, 6)
    assert p.shape == (6, 6)

    assert p.nsplits == ((3, 3), (3, 3))
    assert l_.nsplits == ((3, 3), (3, 3))
    assert u.nsplits == ((3, 3), (3, 3))

    a = mt.random.randint(1, 10, (7, 7), chunk_size=4)
    p, l_, u = mt.linalg.lu(a)
    p, l_, u = tile(p, l_, u)

    assert l_.shape == (7, 7)
    assert u.shape == (7, 7)
    assert p.shape == (7, 7)

    assert p.nsplits == ((4, 3), (4, 3))
    assert l_.nsplits == ((4, 3), (4, 3))
    assert u.nsplits == ((4, 3), (4, 3))

    a = mt.random.randint(1, 10, (7, 5), chunk_size=4)
    p, l_, u = mt.linalg.lu(a)
    p, l_, u = tile(p, l_, u)

    assert l_.shape == (7, 5)
    assert u.shape == (5, 5)
    assert p.shape == (7, 7)

    a = mt.random.randint(1, 10, (5, 7), chunk_size=4)
    p, l_, u = mt.linalg.lu(a)
    p, l_, u = tile(p, l_, u)

    assert l_.shape == (5, 5)
    assert u.shape == (5, 7)
    assert p.shape == (5, 5)

    # test sparse
    data = sps.csr_matrix(
        [
            [2, 0, 0, 0, 5, 2],
            [0, 6, 1, 0, 0, 6],
            [8, 0, 9, 0, 0, 2],
            [0, 6, 0, 8, 7, 3],
            [7, 0, 6, 1, 7, 0],
            [0, 0, 0, 7, 0, 8],
        ]
    )
    t = mt.tensor(data, chunk_size=3)
    p, l_, u = mt.linalg.lu(t)

    assert p.op.sparse is True
    assert isinstance(p, SparseTensor)
    assert l_.op.sparse is True
    assert isinstance(l_, SparseTensor)
    assert u.op.sparse is True
    assert isinstance(u, SparseTensor)

    p, l_, u = tile(p, l_, u)

    assert all(c.is_sparse() for c in p.chunks) is True
    assert all(c.is_sparse() for c in l_.chunks) is True
    assert all(c.is_sparse() for c in u.chunks) is True


def test_solve():
    a = mt.random.randint(1, 10, (20, 20))
    b = mt.random.randint(1, 10, (20,))
    x = tile(mt.linalg.solve(a, b))

    assert x.shape == (20,)

    a = mt.random.randint(1, 10, (20, 20), chunk_size=5)
    b = mt.random.randint(1, 10, (20, 3), chunk_size=5)
    x = tile(mt.linalg.solve(a, b))

    assert x.shape == (20, 3)

    a = mt.random.randint(1, 10, (20, 20), chunk_size=12)
    b = mt.random.randint(1, 10, (20, 3))
    x = tile(mt.linalg.solve(a, b))

    assert x.shape == (20, 3)
    assert x.nsplits == ((12, 8), (3,))

    # test sparse
    a = sps.csr_matrix(np.random.randint(1, 10, (20, 20)))
    b = mt.random.randint(1, 10, (20,), chunk_size=3)
    x = tile(mt.linalg.solve(a, b))

    assert x.shape == (20,)
    assert x.op.sparse is True
    assert x.chunks[0].op.sparse is True

    a = mt.tensor(a, chunk_size=7)
    b = mt.random.randint(1, 10, (20,))
    x = tile(mt.linalg.solve(a, b))

    assert x.shape == (20,)
    assert x.nsplits == ((7, 7, 6),)

    x = tile(mt.linalg.solve(a, b, sparse=False))
    assert x.op.sparse is False
    assert x.chunks[0].op.sparse is False


def test_inv():
    a = mt.random.randint(1, 10, (20, 20), chunk_size=8)
    a_inv = tile(mt.linalg.inv(a))

    assert a_inv.shape == (20, 20)

    # test 1 chunk
    a = mt.random.randint(1, 10, (20, 20), chunk_size=20)
    a_inv = tile(mt.linalg.inv(a))

    assert a_inv.shape == (20, 20)
    assert len(a_inv.chunks) == 1
    assert isinstance(a_inv.chunks[0].op, TensorInv)

    a = mt.random.randint(1, 10, (20, 20), chunk_size=11)
    a_inv = tile(mt.linalg.inv(a))

    assert a_inv.shape == (20, 20)
    assert a_inv.nsplits == ((11, 9), (11, 9))

    b = a.T.dot(a)
    b_inv = tile(mt.linalg.inv(b))
    assert b_inv.shape == (20, 20)

    # test sparse
    data = sps.csr_matrix(np.random.randint(1, 10, (20, 20)))
    a = mt.tensor(data, chunk_size=10)
    a_inv = tile(mt.linalg.inv(a))

    assert a_inv.shape == (20, 20)

    assert a_inv.op.sparse is True
    assert isinstance(a_inv, SparseTensor)
    assert all(c.is_sparse() for c in a_inv.chunks) is True

    b = a.T.dot(a)
    b_inv = tile(mt.linalg.inv(b))
    assert b_inv.shape == (20, 20)

    assert b_inv.op.sparse is True
    assert isinstance(b_inv, SparseTensor)
    assert all(c.is_sparse() for c in b_inv.chunks) is True

    b_inv = tile(mt.linalg.inv(b, sparse=False))
    assert b_inv.op.sparse is False
    assert not all(c.is_sparse() for c in b_inv.chunks) is True


def test_tensordot():
    from .. import tensordot, dot, inner

    t1 = ones((3, 4, 6), chunk_size=2)
    t2 = ones((4, 3, 5), chunk_size=2)
    t3 = tensordot(t1, t2, axes=((0, 1), (1, 0)))

    assert t3.shape == (6, 5)

    t3 = tile(t3)

    assert t3.shape == (6, 5)
    assert len(t3.chunks) == 9

    a = ones((10000, 20000), chunk_size=5000)
    b = ones((20000, 1000), chunk_size=5000)

    with pytest.raises(ValueError):
        tensordot(a, b)

    a = ones(10, chunk_size=2)
    b = ones((10, 20), chunk_size=2)
    c = dot(a, b)
    assert c.shape == (20,)
    c = tile(c)
    assert c.shape == tuple(sum(s) for s in c.nsplits)

    a = ones((10, 20), chunk_size=2)
    b = ones(20, chunk_size=2)
    c = dot(a, b)
    assert c.shape == (10,)
    c = tile(c)
    assert c.shape == tuple(sum(s) for s in c.nsplits)

    v = ones((100, 100), chunk_size=10)
    tv = v.dot(v)
    assert tv.shape == (100, 100)
    tv = tile(tv)
    assert tv.shape == tuple(sum(s) for s in tv.nsplits)

    a = ones((10, 20), chunk_size=2)
    b = ones((30, 20), chunk_size=2)
    c = inner(a, b)
    assert c.shape == (10, 30)
    c = tile(c)
    assert c.shape == tuple(sum(s) for s in c.nsplits)


def test_dot():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()
    t2 = t1.T

    assert t1.dot(t2).issparse() is True
    assert type(t1.dot(t2)) is SparseTensor
    assert t1.dot(t2, sparse=False).issparse() is False
    assert type(t1.dot(t2, sparse=False)) is Tensor

    with pytest.raises(TypeError):
        dot(t1, t2, out=1)

    with pytest.raises(ValueError):
        dot(t1, t2, empty((3, 6)))

    with pytest.raises(ValueError):
        dot(t1, t2, empty((3, 3), dtype="i4"))

    with pytest.raises(ValueError):
        dot(t1, t2, empty((3, 3), order="F"))

    t1.dot(t2, out=empty((2, 2), dtype=t1.dtype))


def test_matmul():
    t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()
    t2 = t1.T

    t3 = matmul(t1, t2, out=empty((2, 2), dtype=t1.dtype, order="F"))
    assert t3.order.value == "F"

    with pytest.raises(TypeError):
        matmul(t1, t2, out=1)

    with pytest.raises(TypeError):
        matmul(t1, t2, out=empty((2, 2), dtype="?"))

    with pytest.raises(ValueError):
        matmul(t1, t2, out=empty((3, 2), dtype=t1.dtype))

    raw1 = np.asfortranarray(np.random.rand(3, 3))
    raw2 = np.asfortranarray(np.random.rand(3, 3))
    raw3 = np.random.rand(3, 3)

    assert (
        matmul(tensor(raw1), tensor(raw2)).flags["C_CONTIGUOUS"]
        == np.matmul(raw1, raw2).flags["C_CONTIGUOUS"]
    )
    assert (
        matmul(tensor(raw1), tensor(raw2)).flags["F_CONTIGUOUS"]
        == np.matmul(raw1, raw2).flags["F_CONTIGUOUS"]
    )

    assert (
        matmul(tensor(raw1), tensor(raw2), order="A").flags["C_CONTIGUOUS"]
        == np.matmul(raw1, raw2, order="A").flags["C_CONTIGUOUS"]
    )
    assert (
        matmul(tensor(raw1), tensor(raw2), order="A").flags["F_CONTIGUOUS"]
        == np.matmul(raw1, raw2, order="A").flags["F_CONTIGUOUS"]
    )

    assert (
        matmul(tensor(raw1), tensor(raw3), order="A").flags["C_CONTIGUOUS"]
        == np.matmul(raw1, raw3, order="A").flags["C_CONTIGUOUS"]
    )
    assert (
        matmul(tensor(raw1), tensor(raw3), order="A").flags["F_CONTIGUOUS"]
        == np.matmul(raw1, raw3, order="A").flags["F_CONTIGUOUS"]
    )
