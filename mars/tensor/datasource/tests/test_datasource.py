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

import shutil
import tempfile
from copy import copy

import numpy as np
import scipy.sparse as sps
import pytest

try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

from .... import dataframe as md
from ....core import enter_mode, tile
from ... import ones, zeros, tensor, full, arange, diag, linspace, triu, tril, ones_like
from ...core import Tensor, SparseTensor
from .. import (
    array,
    fromtiledb,
    TensorTileDBDataSource,
    fromdense,
    asarray,
    ascontiguousarray,
    asfortranarray,
)
from ..tri import TensorTriu, TensorTril
from ..zeros import TensorZeros
from ..from_dense import DenseToSparse
from ..array import CSRMatrixDataSource
from ..ones import TensorOnes, TensorOnesLike
from ..from_dataframe import from_dataframe


def test_array():
    a = tensor([0, 1, 2], chunk_size=2)

    b = array(a)
    assert a is not b

    c = asarray(a)
    assert a is c


def test_ascontiguousarray():
    # dtype different
    raw_a = np.asfortranarray(np.random.rand(2, 4))
    raw_b = np.ascontiguousarray(raw_a, dtype="f4")

    a = tensor(raw_a, chunk_size=2)
    b = ascontiguousarray(a, dtype="f4")

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]

    # no copy
    raw_a = np.random.rand(2, 4)
    raw_b = np.ascontiguousarray(raw_a)

    a = tensor(raw_a, chunk_size=2)
    b = ascontiguousarray(a)

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]


def test_asfortranarray():
    # dtype different
    raw_a = np.random.rand(2, 4)
    raw_b = np.asfortranarray(raw_a, dtype="f4")

    a = tensor(raw_a, chunk_size=2)
    b = asfortranarray(a, dtype="f4")

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]

    # no copy
    raw_a = np.asfortranarray(np.random.rand(2, 4))
    raw_b = np.asfortranarray(raw_a)

    a = tensor(raw_a, chunk_size=2)
    b = asfortranarray(a)

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]


def test_ones():
    tensor = ones((10, 10, 8), chunk_size=(3, 3, 5))
    tensor = tile(tensor)
    assert tensor.shape == (10, 10, 8)
    assert len(tensor.chunks) == 32

    tensor = ones((10, 3), chunk_size=(4, 2))
    tensor = tile(tensor)
    assert tensor.shape == (10, 3)

    chunk = tensor.cix[1, 1]
    assert tensor.get_chunk_slices(chunk.index) == (slice(4, 8), slice(2, 3))

    tensor = ones((10, 5), chunk_size=(2, 3), gpu=True)
    tensor = tile(tensor)

    assert tensor.op.gpu is True
    assert tensor.chunks[0].op.gpu is True

    tensor1 = ones((10, 10, 8), chunk_size=(3, 3, 5))
    tensor1 = tile(tensor1)

    tensor2 = ones((10, 10, 8), chunk_size=(3, 3, 5))
    tensor2 = tile(tensor2)

    assert tensor1.chunks[0].op.key == tensor2.chunks[0].op.key
    assert tensor1.chunks[0].key == tensor2.chunks[0].key
    assert tensor1.chunks[0].op.key != tensor1.chunks[1].op.key
    assert tensor1.chunks[0].key != tensor1.chunks[1].key

    tensor = ones((2, 3, 4))
    assert len(list(tensor)) == 2

    tensor2 = ones((2, 3, 4), chunk_size=1)
    assert tensor.op.key != tensor2.op.key
    assert tensor.key != tensor2.key

    tensor3 = ones((2, 3, 3))
    assert tensor.op.key != tensor3.op.key
    assert tensor.key != tensor3.key

    # test create chunk op of ones manually
    chunk_op1 = TensorOnes(dtype=tensor.dtype)
    chunk1 = chunk_op1.new_chunk(None, shape=(3, 3), index=(0, 0))
    chunk_op2 = TensorOnes(dtype=tensor.dtype)
    chunk2 = chunk_op2.new_chunk(None, shape=(3, 4), index=(0, 1))
    assert chunk1.op.key != chunk2.op.key
    assert chunk1.key != chunk2.key

    tensor = ones((100, 100), chunk_size=50)
    tensor = tile(tensor)
    assert len({c.op.key for c in tensor.chunks}) == 1
    assert len({c.key for c in tensor.chunks}) == 1


def test_zeros():
    tensor = zeros((2, 3, 4))
    assert len(list(tensor)) == 2
    assert tensor.op.gpu is None

    tensor2 = zeros((2, 3, 4), chunk_size=1)
    # tensor's op key must be equal to tensor2
    assert tensor.op.key != tensor2.op.key
    assert tensor.key != tensor2.key

    tensor3 = zeros((2, 3, 3))
    assert tensor.op.key != tensor3.op.key
    assert tensor.key != tensor3.key

    # test create chunk op of zeros manually
    chunk_op1 = TensorZeros(dtype=tensor.dtype)
    chunk1 = chunk_op1.new_chunk(None, shape=(3, 3), index=(0, 0))
    chunk_op2 = TensorZeros(dtype=tensor.dtype)
    chunk2 = chunk_op2.new_chunk(None, shape=(3, 4), index=(0, 1))
    assert chunk1.op.key != chunk2.op.key
    assert chunk1.key != chunk2.key

    tensor = zeros((100, 100), chunk_size=50)
    tensor = tile(tensor)
    assert len({c.op.key for c in tensor.chunks}) == 1
    assert len({c.key for c in tensor.chunks}) == 1


def test_data_source():
    from ...base.broadcast_to import TensorBroadcastTo

    data = np.random.random((10, 3))
    t = tensor(data, chunk_size=2)
    assert t.op.gpu is None
    t = tile(t)
    assert (t.chunks[0].op.data == data[:2, :2]).all()
    assert (t.chunks[1].op.data == data[:2, 2:3]).all()
    assert (t.chunks[2].op.data == data[2:4, :2]).all()
    assert (t.chunks[3].op.data == data[2:4, 2:3]).all()

    assert t.key == tile(tensor(data, chunk_size=2)).key
    assert t.key != tile(tensor(data, chunk_size=3)).key
    assert t.key != tile(tensor(np.random.random((10, 3)), chunk_size=2)).key

    t = tensor(data, chunk_size=2, gpu=True)
    t = tile(t)

    assert t.op.gpu is True
    assert t.chunks[0].op.gpu is True

    t = full((2, 2), 2, dtype="f4")
    assert t.op.gpu is None
    assert t.shape == (2, 2)
    assert t.dtype == np.float32

    t = full((2, 2), [1.0, 2.0], dtype="f4")
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    assert isinstance(t.op, TensorBroadcastTo)

    with pytest.raises(ValueError):
        full((2, 2), [1.0, 2.0, 3.0], dtype="f4")


def test_ufunc():
    t = ones((3, 10), chunk_size=2)

    x = np.add(t, [[1], [2], [3]])
    assert isinstance(x, Tensor)

    y = np.sum(t, axis=1)
    assert isinstance(y, Tensor)


def test_arange():
    t = arange(10, chunk_size=3)

    assert t.op.gpu is False
    t = tile(t)

    assert t.shape == (10,)
    assert t.nsplits == ((3, 3, 3, 1),)
    assert t.chunks[1].op.start == 3
    assert t.chunks[1].op.stop == 6

    t = arange(0, 10, 3, chunk_size=2)
    t = tile(t)

    assert t.shape == (4,)
    assert t.nsplits == ((2, 2),)
    assert t.chunks[0].op.start == 0
    assert t.chunks[0].op.stop == 6
    assert t.chunks[0].op.step == 3
    assert t.chunks[1].op.start == 6
    assert t.chunks[1].op.stop == 12
    assert t.chunks[1].op.step == 3

    pytest.raises(TypeError, lambda: arange(10, start=0))
    pytest.raises(TypeError, lambda: arange(0, 10, stop=0))
    pytest.raises(TypeError, lambda: arange())
    pytest.raises(
        ValueError, lambda: arange("1066-10-13", dtype=np.datetime64, chunks=3)
    )


def test_diag():
    # test 2-d, shape[0] == shape[1], k == 0
    v = tensor(np.arange(16).reshape(4, 4), chunk_size=2)
    t = diag(v)

    assert t.shape == (4,)
    assert t.op.gpu is None
    t = tile(t)
    assert t.nsplits == ((2, 2),)

    v = tensor(np.arange(16).reshape(4, 4), chunk_size=(2, 3))
    t = diag(v)

    assert t.shape == (4,)
    t = tile(t)
    assert t.nsplits == ((2, 1, 1),)

    # test 1-d, k == 0
    v = tensor(np.arange(3), chunk_size=2)
    t = diag(v, sparse=True)

    assert t.shape == (3, 3)
    t = tile(t)
    assert t.nsplits == ((2, 1), (2, 1))
    assert len([c for c in t.chunks if c.op.__class__.__name__ == "TensorDiag"]) == 2
    assert t.chunks[0].op.sparse is True

    # test 2-d, shape[0] != shape[1]
    v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)
    t = diag(v)

    assert t.shape == np.diag(np.arange(24).reshape(4, 6)).shape
    t = tile(t)
    assert tuple(sum(s) for s in t.nsplits) == t.shape

    v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)

    t = diag(v, k=1)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=1).shape
    t = tile(t)
    assert tuple(sum(s) for s in t.nsplits) == t.shape

    t = diag(v, k=2)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=2).shape
    t = tile(t)
    assert tuple(sum(s) for s in t.nsplits) == t.shape

    t = diag(v, k=-1)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=-1).shape
    t = tile(t)
    assert tuple(sum(s) for s in t.nsplits) == t.shape

    t = diag(v, k=-2)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=-2).shape
    t = tile(t)
    assert tuple(sum(s) for s in t.nsplits) == t.shape

    # test tiled zeros' keys
    a = arange(5, chunk_size=2)
    t = diag(a)
    t = tile(t)
    # 1 and 2 of t.chunks is ones, they have different shapes
    assert t.chunks[1].op.key != t.chunks[2].op.key


def test_linspace():
    a = linspace(2.0, 3.0, num=5, chunk_size=2)

    assert a.shape == (5,)

    a = tile(a)
    assert a.nsplits == ((2, 2, 1),)
    assert a.chunks[0].op.start == 2.0
    assert a.chunks[0].op.stop == 2.25
    assert a.chunks[1].op.start == 2.5
    assert a.chunks[1].op.stop == 2.75
    assert a.chunks[2].op.start == 3.0
    assert a.chunks[2].op.stop == 3.0

    a = linspace(2.0, 3.0, num=5, endpoint=False, chunk_size=2)

    assert a.shape == (5,)

    a = tile(a)
    assert a.nsplits == ((2, 2, 1),)
    assert a.chunks[0].op.start == 2.0
    assert a.chunks[0].op.stop == 2.2
    assert a.chunks[1].op.start == 2.4
    assert a.chunks[1].op.stop == 2.6
    assert a.chunks[2].op.start == 2.8
    assert a.chunks[2].op.stop == 2.8

    _, step = linspace(2.0, 3.0, num=5, chunk_size=2, retstep=True)
    assert step == 0.25


def test_triu_tril():
    a_data = np.arange(12).reshape(4, 3)
    a = tensor(a_data, chunk_size=2)

    t = triu(a)

    assert t.op.gpu is None

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTriu)
    assert isinstance(t.chunks[1].op, TensorTriu)
    assert isinstance(t.chunks[2].op, TensorZeros)
    assert isinstance(t.chunks[3].op, TensorTriu)

    t = triu(a, k=1)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTriu)
    assert isinstance(t.chunks[1].op, TensorTriu)
    assert isinstance(t.chunks[2].op, TensorZeros)
    assert isinstance(t.chunks[3].op, TensorZeros)

    t = triu(a, k=2)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorZeros)
    assert isinstance(t.chunks[1].op, TensorTriu)
    assert isinstance(t.chunks[2].op, TensorZeros)
    assert isinstance(t.chunks[3].op, TensorZeros)

    t = triu(a, k=-1)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTriu)
    assert isinstance(t.chunks[1].op, TensorTriu)
    assert isinstance(t.chunks[2].op, TensorTriu)
    assert isinstance(t.chunks[3].op, TensorTriu)

    t = tril(a)

    assert t.op.gpu is None

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTril)
    assert isinstance(t.chunks[1].op, TensorZeros)
    assert isinstance(t.chunks[2].op, TensorTril)
    assert isinstance(t.chunks[3].op, TensorTril)

    t = tril(a, k=1)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTril)
    assert isinstance(t.chunks[1].op, TensorTril)
    assert isinstance(t.chunks[2].op, TensorTril)
    assert isinstance(t.chunks[3].op, TensorTril)

    t = tril(a, k=-1)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorTril)
    assert isinstance(t.chunks[1].op, TensorZeros)
    assert isinstance(t.chunks[2].op, TensorTril)
    assert isinstance(t.chunks[3].op, TensorTril)

    t = tril(a, k=-2)

    t = tile(t)
    assert len(t.chunks) == 4
    assert isinstance(t.chunks[0].op, TensorZeros)
    assert isinstance(t.chunks[1].op, TensorZeros)
    assert isinstance(t.chunks[2].op, TensorTril)
    assert isinstance(t.chunks[3].op, TensorZeros)


def test_set_tensor_inputs():
    t1 = tensor([1, 2], chunk_size=2)
    t2 = tensor([2, 3], chunk_size=2)
    t3 = t1 + t2

    t1c = copy(t1)
    t2c = copy(t2)

    assert t1c is not t1
    assert t2c is not t2

    assert t3.op.lhs is t1.data
    assert t3.op.rhs is t2.data
    assert t3.op.inputs == [t1.data, t2.data]
    assert t3.inputs == [t1.data, t2.data]

    with pytest.raises(StopIteration):
        t3.inputs = []

    t1 = tensor([1, 2], chunk_size=2)
    t2 = tensor([True, False], chunk_size=2)
    t3 = t1[t2]

    t1c = copy(t1)
    t2c = copy(t2)
    t3c = copy(t3)
    t3c.inputs = [t1c, t2c]

    with enter_mode(build=True):
        assert t3c.op.input is t1c.data
        assert t3c.op.indexes[0] is t2c.data


def test_from_spmatrix():
    t = tensor(sps.csr_matrix([[0, 0, 1], [1, 0, 0]], dtype="f8"), chunk_size=2)

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, CSRMatrixDataSource)
    assert t.issparse() is True
    assert not t.op.gpu

    t = tile(t)
    assert t.chunks[0].index == (0, 0)
    assert isinstance(t.op, CSRMatrixDataSource)
    assert not t.op.gpu
    m = sps.csr_matrix([[0, 0], [1, 0]])
    assert np.array_equal(t.chunks[0].op.indices, m.indices) is True
    assert np.array_equal(t.chunks[0].op.indptr, m.indptr) is True
    assert np.array_equal(t.chunks[0].op.data, m.data) is True
    assert np.array_equal(t.chunks[0].op.shape, m.shape) is True


def test_from_dense():
    t = fromdense(tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2))

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, DenseToSparse)
    assert t.issparse() is True

    t = tile(t)
    assert t.chunks[0].index == (0, 0)
    assert isinstance(t.op, DenseToSparse)


def test_ones_like():
    t1 = tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2).tosparse()
    t = ones_like(t1, dtype="f8")

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, TensorOnesLike)
    assert t.issparse() is True
    assert t.op.gpu is None

    t = tile(t)
    assert t.chunks[0].index == (0, 0)
    assert isinstance(t.op, TensorOnesLike)
    assert t.chunks[0].issparse() is True


def test_from_array():
    x = array([1, 2, 3])
    assert x.shape == (3,)

    y = array([x, x])
    assert y.shape == (2, 3)

    z = array((x, x, x))
    assert z.shape == (3, 3)


@pytest.mark.skipif(tiledb is None, reason="TileDB not installed")
def test_from_tile_db():
    ctx = tiledb.Ctx()

    for sparse in (True, False):
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, name="i", domain=(1, 30), tile=7, dtype=np.int32),
            tiledb.Dim(ctx=ctx, name="j", domain=(1, 20), tile=3, dtype=np.int32),
            tiledb.Dim(ctx=ctx, name="k", domain=(1, 10), tile=4, dtype=np.int32),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(
            ctx=ctx,
            domain=dom,
            sparse=sparse,
            attrs=[tiledb.Attr(ctx=ctx, name="a", dtype=np.float32)],
        )

        tempdir = tempfile.mkdtemp()
        try:
            # create tiledb array
            array_type = tiledb.DenseArray if not sparse else tiledb.SparseArray
            array_type.create(tempdir, schema)

            tensor = fromtiledb(tempdir)
            assert isinstance(tensor.op, TensorTileDBDataSource)
            assert tensor.op.issparse() == sparse
            assert tensor.shape == (30, 20, 10)
            assert tensor.extra_params.raw_chunk_size == (7, 3, 4)
            assert tensor.op.tiledb_config is None
            assert tensor.op.tiledb_uri == tempdir
            assert tensor.op.tiledb_key is None
            assert tensor.op.tiledb_timestamp is None

            tensor = tile(tensor)

            assert len(tensor.chunks) == 105
            assert isinstance(tensor.chunks[0].op, TensorTileDBDataSource)
            assert tensor.chunks[0].op.issparse() == sparse
            assert tensor.chunks[0].shape == (7, 3, 4)
            assert tensor.chunks[0].op.tiledb_config is None
            assert tensor.chunks[0].op.tiledb_uri == tempdir
            assert tensor.chunks[0].op.tiledb_key is None
            assert tensor.chunks[0].op.tiledb_timestamp is None
            assert tensor.chunks[0].op.tiledb_dim_starts == (1, 1, 1)

            # test axis_offsets of chunk op
            assert tensor.chunks[0].op.axis_offsets == (0, 0, 0)
            assert tensor.chunks[1].op.axis_offsets == (0, 0, 4)
            assert tensor.cix[0, 2, 2].op.axis_offsets == (0, 6, 8)
            assert tensor.cix[0, 6, 2].op.axis_offsets == (0, 18, 8)
            assert tensor.cix[4, 6, 2].op.axis_offsets == (28, 18, 8)

            tensor2 = fromtiledb(tempdir, ctx=ctx)
            assert tensor2.op.tiledb_config == ctx.config().dict()

            tensor2 = tile(tensor2)

            assert tensor2.chunks[0].op.tiledb_config == ctx.config().dict()
        finally:
            shutil.rmtree(tempdir)


@pytest.mark.skipif(tiledb is None, reason="TileDB not installed")
def test_dim_start_float():
    ctx = tiledb.Ctx()

    dom = tiledb.Domain(
        tiledb.Dim(ctx=ctx, name="i", domain=(0.0, 6.0), tile=6, dtype=np.float64),
        ctx=ctx,
    )
    schema = tiledb.ArraySchema(
        ctx=ctx,
        domain=dom,
        sparse=True,
        attrs=[tiledb.Attr(ctx=ctx, name="a", dtype=np.float32)],
    )

    tempdir = tempfile.mkdtemp()
    try:
        # create tiledb array
        tiledb.SparseArray.create(tempdir, schema)

        with pytest.raises(ValueError):
            fromtiledb(tempdir, ctx=ctx)
    finally:
        shutil.rmtree(tempdir)


def test_from_dataframe():
    mdf = md.DataFrame(
        {"a": [0, 1, 2], "b": [3, 4, 5], "c": [0.1, 0.2, 0.3]},
        index=["c", "d", "e"],
        chunk_size=2,
    )
    tensor = from_dataframe(mdf)
    assert tensor.shape == (3, 3)
    assert np.float64 == tensor.dtype
