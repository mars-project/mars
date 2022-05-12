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

import tempfile
import shutil
import os
import time

import numpy as np
import pandas as pd
import pytest

try:
    import scipy.sparse as sps
except ImportError:
    sps = None
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None
try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

from .... import tensor as mt
from .... import dataframe as md
from ....lib.sparse import SparseNDArray
from ....tests.core import require_cupy
from ....utils import lazy_import
from ...lib import nd_grid
from .. import (
    tensor,
    ones_like,
    zeros,
    zeros_like,
    full,
    full_like,
    arange,
    empty,
    empty_like,
    diag,
    diagflat,
    eye,
    linspace,
    meshgrid,
    indices,
    triu,
    tril,
    from_dataframe,
    fromtiledb,
    fromhdf5,
    fromzarr,
)

cupy = lazy_import("cupy")


@require_cupy
def test_array_gpu_execution(setup_gpu):
    raw = cupy.random.rand(20, 30)
    t = tensor(raw, dtype="f8", chunk_size=10)

    res = t.execute().fetch()
    expected = raw.astype("f8")
    cupy.testing.assert_array_equal(res, expected)


def test_create_sparse_execution(setup):
    mat = sps.csr_matrix([[0, 0, 2], [2, 0, 0]])
    t = tensor(mat, dtype="f8", chunk_size=2)

    res = t.execute().fetch()
    assert isinstance(res, SparseNDArray)
    assert res.dtype == np.float64
    np.testing.assert_array_equal(res.toarray(), mat.toarray())

    t2 = ones_like(t, dtype="f4")

    res = t2.execute().fetch()
    expected = sps.csr_matrix([[0, 0, 1], [1, 0, 0]])
    assert isinstance(res, SparseNDArray)
    assert res.dtype == np.float32
    np.testing.assert_array_equal(res.toarray(), expected.toarray())

    t3 = tensor(np.array([[0, 0, 2], [2, 0, 0]]), chunk_size=2).tosparse()

    res = t3.execute().fetch()
    assert isinstance(res, SparseNDArray)
    assert res.dtype == np.int_
    np.testing.assert_array_equal(res.toarray(), mat.toarray())

    # test missing argument
    t4 = tensor(np.array([[0, 0, 2], [2, 0, 0]]), chunk_size=2).tosparse(missing=2)
    t4 = t4 + 1
    expected = mat.toarray()
    raw = expected.copy()
    expected[raw == 0] += 1
    expected[raw != 0] = 0

    res = t4.execute().fetch()
    assert isinstance(res, SparseNDArray)
    assert res.dtype == np.int_
    np.testing.assert_array_equal(res.toarray(), expected)

    # test missing argument that is np.nan
    t5 = tensor(
        np.array([[np.nan, np.nan, 2], [2, np.nan, -999]]), chunk_size=2
    ).tosparse(missing=[np.nan, -999])
    t5 = (t5 + 1).todense(fill_value=np.nan)
    expected = mat.toarray().astype(float)
    expected[expected != 0] += 1
    expected[expected == 0] = np.nan

    res = t5.execute().fetch()
    assert res.dtype == np.float64
    np.testing.assert_array_equal(res, expected)


def test_zeros_execution(setup):
    t = zeros((20, 30), dtype="i8", chunk_size=10)

    res = t.execute().fetch()
    np.testing.assert_array_equal(res, np.zeros((20, 30), dtype="i8"))
    assert res[0].dtype == np.int64

    t2 = zeros_like(t)
    res = t2.execute().fetch()
    np.testing.assert_array_equal(res, np.zeros((20, 30), dtype="i8"))
    assert res[0].dtype == np.int64

    t = zeros((20, 30), dtype="i4", chunk_size=5, sparse=True)
    res = t.execute().fetch()

    assert res[0].nnz == 0

    t = zeros((20, 30), dtype="i8", chunk_size=6, order="F")
    res = t.execute().fetch()
    expected = np.zeros((20, 30), dtype="i8", order="F")
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_empty_execution(setup):
    t = empty((20, 30), dtype="i8", chunk_size=5)

    res = t.execute().fetch()
    assert res.shape == (20, 30)
    assert res.dtype == np.int64

    t = empty((20, 30), chunk_size=10)

    res = t.execute().fetch()
    assert res.shape == (20, 30)
    assert res.dtype == np.float64

    t2 = empty_like(t)
    res = t2.execute().fetch()
    assert res.shape == (20, 30)
    assert res.dtype == np.float64

    t = empty((20, 30), dtype="i8", chunk_size=5, order="F")

    res = t.execute().fetch()
    expected = np.empty((20, 30), dtype="i8", order="F")
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_full_execution(setup):
    t = full((2, 2), 1, dtype="f4", chunk_size=1)

    res = t.execute().fetch()
    np.testing.assert_array_equal(res, np.full((2, 2), 1, dtype="f4"))

    t = full((2, 2), [1, 2], dtype="f8", chunk_size=1)

    res = t.execute().fetch()
    np.testing.assert_array_equal(res, np.full((2, 2), [1, 2], dtype="f8"))

    t = full((2, 2), 1, dtype="f4", chunk_size=1, order="F")

    res = t.execute().fetch()
    expected = np.full((2, 2), 1, dtype="f4", order="F")
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    t2 = full_like(t, 10, order="F")

    res = t2.execute().fetch()
    expected = np.full((2, 2), 10, dtype="f4", order="F")
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_arange_execution(setup):
    t = arange(1, 20, 3, chunk_size=2)

    res = t.execute().fetch()
    assert np.array_equal(res, np.arange(1, 20, 3)) is True

    t = arange(1, 20, 0.3, chunk_size=4)

    res = t.execute().fetch()
    expected = np.arange(1, 20, 0.3)
    assert np.allclose(res, expected) is True

    t = arange(1.0, 1.8, 0.3, chunk_size=4)

    res = t.execute().fetch()
    expected = np.arange(1.0, 1.8, 0.3)
    assert np.allclose(res, expected) is True

    t = arange("1066-10-13", "1066-10-31", dtype=np.datetime64, chunk_size=3)

    res = t.execute().fetch()
    expected = np.arange("1066-10-13", "1066-10-31", dtype=np.datetime64)
    assert np.array_equal(res, expected) is True


def test_diag_execution(setup):
    # 2-d  6 * 6
    a = arange(36, chunk_size=5).reshape(6, 6)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(6, 6))
    np.testing.assert_equal(res, expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(6, 6), k=1)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(6, 6), k=3)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(6, 6), k=-2)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-5)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(6, 6), k=-5)
    np.testing.assert_equal(res, expected)

    # 2-d  6 * 6 sparse, no tensor
    a = sps.rand(6, 6, density=0.1)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(a.toarray())
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=1)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=3)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=-2)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-5)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=-5)
    np.testing.assert_equal(res.toarray(), expected)

    # 2-d  6 * 6 sparse, from tensor
    raw_a = sps.rand(6, 6, density=0.1)
    a = tensor(raw_a, chunk_size=2)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray())
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=1)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=3)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=-2)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-5)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=-5)
    np.testing.assert_equal(res.toarray(), expected)

    # 2-d  4 * 9
    a = arange(36, chunk_size=2).reshape(4, 9)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(4, 9))
    np.testing.assert_equal(res, expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(4, 9), k=1)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(4, 9), k=3)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(4, 9), k=-2)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-3)
    res = d.execute().fetch()
    expected = np.diag(np.arange(36).reshape(4, 9), k=-3)
    np.testing.assert_equal(res, expected)

    # 2-d  4 * 9 sparse, no tensor
    a = sps.rand(4, 9, density=0.1)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(a.toarray())
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=1)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=3)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=-2)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-3)
    res = d.execute().fetch()
    expected = np.diag(a.toarray(), k=-3)
    np.testing.assert_equal(res.toarray(), expected)

    # 2-d  4 * 9 sparse, from tensor
    raw_a = sps.rand(4, 9, density=0.1)
    a = tensor(raw_a, chunk_size=2)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray())
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=1)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=3)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=-2)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-3)
    res = d.execute().fetch()
    expected = np.diag(raw_a.toarray(), k=-3)
    np.testing.assert_equal(res.toarray(), expected)

    # 1-d
    a = arange(5, chunk_size=2)

    d = diag(a)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5))
    np.testing.assert_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] is True
    assert res.flags["F_CONTIGUOUS"] is False

    d = diag(a, k=1)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=1)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=3)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=3)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-2)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=-2)
    np.testing.assert_equal(res, expected)

    d = diag(a, k=-3)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=-3)
    np.testing.assert_equal(res, expected)

    d = diag(a, sparse=True)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5))
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=1, sparse=True)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=2, sparse=True)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-2, sparse=True)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=-2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    d = diag(a, k=-3, sparse=True)
    res = d.execute().fetch()
    expected = np.diag(np.arange(5), k=-3)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)


def test_diagflat_execution(setup):
    a = diagflat([[1, 2], [3, 4]], chunk_size=1)

    res = a.execute().fetch()
    expected = np.diagflat([[1, 2], [3, 4]])
    np.testing.assert_equal(res, expected)

    d = tensor([[1, 2], [3, 4]], chunk_size=1)
    a = diagflat(d)

    res = a.execute().fetch()
    expected = np.diagflat([[1, 2], [3, 4]])
    np.testing.assert_equal(res, expected)

    a = diagflat([1, 2], 1, chunk_size=1)

    res = a.execute().fetch()
    expected = np.diagflat([1, 2], 1)
    np.testing.assert_equal(res, expected)

    d = tensor([[1, 2]], chunk_size=1)
    a = diagflat(d, 1)

    res = a.execute().fetch()
    expected = np.diagflat([1, 2], 1)
    np.testing.assert_equal(res, expected)


def test_eye_execution(setup):
    t = eye(5, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5)
    np.testing.assert_equal(res, expected)

    t = eye(5, k=1, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=1)
    np.testing.assert_equal(res, expected)

    t = eye(5, k=2, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=2)
    np.testing.assert_equal(res, expected)

    t = eye(5, k=-1, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=-1)
    np.testing.assert_equal(res, expected)

    t = eye(5, k=-3, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=-3)
    np.testing.assert_equal(res, expected)

    t = eye(5, M=3, k=1, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=3, k=1)
    np.testing.assert_equal(res, expected)

    t = eye(5, M=3, k=-3, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=3, k=-3)
    np.testing.assert_equal(res, expected)

    t = eye(5, M=7, k=1, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=7, k=1)
    np.testing.assert_equal(res, expected)

    t = eye(5, M=8, k=-3, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=8, k=-3)
    np.testing.assert_equal(res, expected)

    t = eye(2, dtype=int)

    res = t.execute().fetch()
    assert res.dtype == np.int_

    # test sparse
    t = eye(5, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, k=1, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, k=2, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, k=-1, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=-1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, k=-3, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, k=-3)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, M=3, k=1, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=3, k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, M=3, k=-3, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=3, k=-3)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, M=7, k=1, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=7, k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, M=8, k=-3, sparse=True, chunk_size=2)

    res = t.execute().fetch()
    expected = np.eye(5, M=8, k=-3)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res.toarray(), expected)

    t = eye(5, M=9, k=-3, chunk_size=2, order="F")

    res = t.execute().fetch()
    assert res.flags["C_CONTIGUOUS"] is True
    assert res.flags["F_CONTIGUOUS"] is False


def test_linspace_execution(setup):
    a = linspace(2.0, 9.0, num=11, chunk_size=3)

    res = a.execute().fetch()
    expected = np.linspace(2.0, 9.0, num=11)
    np.testing.assert_allclose(res, expected)

    a = linspace(2.0, 9.0, num=11, endpoint=False, chunk_size=3)

    res = a.execute().fetch()
    expected = np.linspace(2.0, 9.0, num=11, endpoint=False)
    np.testing.assert_allclose(res, expected)

    a = linspace(2.0, 9.0, num=11, chunk_size=3, dtype=int)

    res = a.execute().fetch()
    assert res.dtype == np.int_


def test_meshgrid_execution(setup):
    a = arange(5, chunk_size=2)
    b = arange(6, 12, chunk_size=3)
    c = arange(12, 19, chunk_size=4)

    A, B, C = meshgrid(a, b, c)

    A_res = A.execute().fetch()
    A_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[0]
    np.testing.assert_equal(A_res, A_expected)

    B_res = B.execute().fetch()
    B_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[1]
    np.testing.assert_equal(B_res, B_expected)

    C_res = C.execute().fetch()
    C_expected = np.meshgrid(np.arange(5), np.arange(6, 12), np.arange(12, 19))[2]
    np.testing.assert_equal(C_res, C_expected)

    A, B, C = meshgrid(a, b, c, indexing="ij")

    A_res = A.execute().fetch()
    A_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij"
    )[0]
    np.testing.assert_equal(A_res, A_expected)

    B_res = B.execute().fetch()
    B_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij"
    )[1]
    np.testing.assert_equal(B_res, B_expected)

    C_res = C.execute().fetch()
    C_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij"
    )[2]
    np.testing.assert_equal(C_res, C_expected)

    A, B, C = meshgrid(a, b, c, sparse=True)

    A_res = A.execute().fetch()
    A_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True
    )[0]
    np.testing.assert_equal(A_res, A_expected)

    B_res = B.execute().fetch()
    B_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True
    )[1]
    np.testing.assert_equal(B_res, B_expected)

    C_res = C.execute().fetch()
    C_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), sparse=True
    )[2]
    np.testing.assert_equal(C_res, C_expected)

    A, B, C = meshgrid(a, b, c, indexing="ij", sparse=True)

    A_res = A.execute().fetch()
    A_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij", sparse=True
    )[0]
    np.testing.assert_equal(A_res, A_expected)

    B_res = B.execute().fetch()
    B_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij", sparse=True
    )[1]
    np.testing.assert_equal(B_res, B_expected)

    C_res = C.execute().fetch()
    C_expected = np.meshgrid(
        np.arange(5), np.arange(6, 12), np.arange(12, 19), indexing="ij", sparse=True
    )[2]
    np.testing.assert_equal(C_res, C_expected)


def test_indices_execution(setup):
    grid = indices((2, 3), chunk_size=1)

    res = grid.execute().fetch()
    expected = np.indices((2, 3))
    np.testing.assert_equal(res, expected)

    res = grid[0].execute().fetch()
    np.testing.assert_equal(res, expected[0])

    res = grid[1].execute().fetch()
    np.testing.assert_equal(res, expected[1])


def test_triu_execution(setup):
    a = arange(24, chunk_size=2).reshape(2, 3, 4)

    t = triu(a)

    res = t.execute().fetch()
    expected = np.triu(np.arange(24).reshape(2, 3, 4))
    np.testing.assert_equal(res, expected)

    t = triu(a, k=1)

    res = t.execute().fetch()
    expected = np.triu(np.arange(24).reshape(2, 3, 4), k=1)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=2)

    res = t.execute().fetch()
    expected = np.triu(np.arange(24).reshape(2, 3, 4), k=2)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=-1)

    res = t.execute().fetch()
    expected = np.triu(np.arange(24).reshape(2, 3, 4), k=-1)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=-2)

    res = t.execute().fetch()
    expected = np.triu(np.arange(24).reshape(2, 3, 4), k=-2)
    np.testing.assert_equal(res, expected)

    # test sparse
    a = arange(12, chunk_size=2).reshape(3, 4).tosparse()

    t = triu(a)

    res = t.execute().fetch()
    expected = np.triu(np.arange(12).reshape(3, 4))
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=1)

    res = t.execute().fetch()
    expected = np.triu(np.arange(12).reshape(3, 4), k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=2)

    res = t.execute().fetch()
    expected = np.triu(np.arange(12).reshape(3, 4), k=2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=-1)

    res = t.execute().fetch()
    expected = np.triu(np.arange(12).reshape(3, 4), k=-1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = triu(a, k=-2)

    res = t.execute().fetch()
    expected = np.triu(np.arange(12).reshape(3, 4), k=-2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    raw = np.asfortranarray(np.random.rand(10, 7))
    a = tensor(raw, chunk_size=3)

    t = triu(a, k=-2)

    res = t.execute().fetch()
    expected = np.triu(raw, k=-2)
    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]


def test_tril_execution(setup):
    a = arange(24, chunk_size=2).reshape(2, 3, 4)

    t = tril(a)

    res = t.execute().fetch()
    expected = np.tril(np.arange(24).reshape(2, 3, 4))
    np.testing.assert_equal(res, expected)

    t = tril(a, k=1)

    res = t.execute().fetch()
    expected = np.tril(np.arange(24).reshape(2, 3, 4), k=1)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=2)

    res = t.execute().fetch()
    expected = np.tril(np.arange(24).reshape(2, 3, 4), k=2)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=-1)

    res = t.execute().fetch()
    expected = np.tril(np.arange(24).reshape(2, 3, 4), k=-1)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=-2)

    res = t.execute().fetch()
    expected = np.tril(np.arange(24).reshape(2, 3, 4), k=-2)
    np.testing.assert_equal(res, expected)

    a = arange(12, chunk_size=2).reshape(3, 4).tosparse()

    t = tril(a)

    res = t.execute().fetch()
    expected = np.tril(np.arange(12).reshape(3, 4))
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=1)

    res = t.execute().fetch()
    expected = np.tril(np.arange(12).reshape(3, 4), k=1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=2)

    res = t.execute().fetch()
    expected = np.tril(np.arange(12).reshape(3, 4), k=2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=-1)

    res = t.execute().fetch()
    expected = np.tril(np.arange(12).reshape(3, 4), k=-1)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)

    t = tril(a, k=-2)

    res = t.execute().fetch()
    expected = np.tril(np.arange(12).reshape(3, 4), k=-2)
    assert isinstance(res, SparseNDArray)
    np.testing.assert_equal(res, expected)


def test_index_trick_execution(setup):
    mgrid = nd_grid()
    t = mgrid[0:5, 0:5]

    res = t.execute().fetch()
    expected = np.lib.index_tricks.nd_grid()[0:5, 0:5]
    np.testing.assert_equal(res, expected)

    t = mgrid[-1:1:5j]

    res = t.execute().fetch()
    expected = np.lib.index_tricks.nd_grid()[-1:1:5j]
    np.testing.assert_equal(res, expected)

    ogrid = nd_grid(sparse=True)

    t = ogrid[0:5, 0:5]

    res = [o.execute().fetch() for o in t]
    expected = np.lib.index_tricks.nd_grid(sparse=True)[0:5, 0:5]
    [np.testing.assert_equal(r, e) for r, e in zip(res, expected)]


@pytest.mark.skipif(tiledb is None, reason="tiledb not installed")
def test_read_tile_db_execution(setup):
    ctx = tiledb.Ctx()

    tempdir = tempfile.mkdtemp()
    try:
        # create TileDB dense array
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32),
            tiledb.Dim(ctx=ctx, domain=(0, 90), tile=22, dtype=np.int32),
            tiledb.Dim(ctx=ctx, domain=(0, 9), tile=8, dtype=np.int32),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(
            ctx=ctx,
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)],
        )
        tiledb.DenseArray.create(tempdir, schema)

        expected = np.random.rand(100, 91, 10)
        with tiledb.DenseArray(uri=tempdir, ctx=ctx, mode="w") as arr:
            arr.write_direct(expected)

        a = fromtiledb(tempdir, ctx=ctx)
        result = a.execute().fetch()

        np.testing.assert_allclose(expected, result)
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # create 2-d TileDB sparse array
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(0, 99), tile=30, dtype=np.int32),
            tiledb.Dim(ctx=ctx, domain=(2, 11), tile=8, dtype=np.int32),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(
            ctx=ctx,
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)],
        )
        tiledb.SparseArray.create(tempdir, schema)

        expected = sps.rand(100, 10, density=0.01)
        with tiledb.SparseArray(uri=tempdir, ctx=ctx, mode="w") as arr:
            I, J = expected.row, expected.col + 2
            arr[I, J] = {arr.attr(0).name: expected.data}

        a = fromtiledb(tempdir, ctx=ctx)
        result = a.execute().fetch()

        np.testing.assert_allclose(expected.toarray(), result.toarray())
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # create 1-d TileDB sparse array
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32), ctx=ctx
        )
        schema = tiledb.ArraySchema(
            ctx=ctx,
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)],
        )
        tiledb.SparseArray.create(tempdir, schema)

        expected = sps.rand(1, 100, density=0.05)
        with tiledb.SparseArray(uri=tempdir, ctx=ctx, mode="w") as arr:
            arr[expected.col + 1] = expected.data

        a = fromtiledb(tempdir, ctx=ctx)
        result = a.execute().fetch()

        np.testing.assert_allclose(expected.toarray()[0], result.toarray())
    finally:
        shutil.rmtree(tempdir)

    tempdir = tempfile.mkdtemp()
    try:
        # create TileDB dense array with column-major
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(1, 100), tile=30, dtype=np.int32),
            tiledb.Dim(ctx=ctx, domain=(0, 90), tile=22, dtype=np.int32),
            tiledb.Dim(ctx=ctx, domain=(0, 9), tile=8, dtype=np.int32),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(
            ctx=ctx,
            domain=dom,
            sparse=False,
            cell_order="F",
            attrs=[tiledb.Attr(ctx=ctx, dtype=np.float64)],
        )
        tiledb.DenseArray.create(tempdir, schema)

        expected = np.asfortranarray(np.random.rand(100, 91, 10))
        with tiledb.DenseArray(uri=tempdir, ctx=ctx, mode="w") as arr:
            arr.write_direct(expected)

        a = fromtiledb(tempdir, ctx=ctx)
        result = a.execute().fetch()

        np.testing.assert_allclose(expected, result)
        assert result.flags["F_CONTIGUOUS"] is True
        assert result.flags["C_CONTIGUOUS"] is False
    finally:
        shutil.rmtree(tempdir)


def test_from_dataframe_execution(setup):
    mdf = md.DataFrame(
        {"angle": [0, 3, 4], "degree": [360, 180, 360]},
        index=["circle", "triangle", "rectangle"],
    )
    tensor_result = from_dataframe(mdf).execute().fetch()
    tensor_expected = mt.tensor([[0, 360], [3, 180], [4, 360]]).execute().fetch()
    np.testing.assert_equal(tensor_result, tensor_expected)

    # test up-casting
    mdf2 = md.DataFrame({"a": [0.1, 0.2, 0.3], "b": [1, 2, 3]})
    tensor_result2 = from_dataframe(mdf2).execute().fetch()
    np.testing.assert_equal(tensor_result2[0].dtype, np.dtype("float64"))
    tensor_expected2 = mt.tensor([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]]).execute().fetch()
    np.testing.assert_equal(tensor_result2, tensor_expected2)

    raw = [[0.1, 0.2, 0.4], [0.4, 0.7, 0.3]]
    mdf3 = md.DataFrame(raw, columns=list("abc"), chunk_size=2)
    tensor_result3 = from_dataframe(mdf3).execute().fetch()
    np.testing.assert_array_equal(tensor_result3, np.asarray(raw))
    assert tensor_result3.flags["F_CONTIGUOUS"] is True
    assert tensor_result3.flags["C_CONTIGUOUS"] is False

    # test from series
    series = md.Series([1, 2, 3])
    tensor_result = series.to_tensor().execute().fetch()
    np.testing.assert_array_equal(tensor_result, np.array([1, 2, 3]))

    series = md.Series(range(10), chunk_size=3)
    tensor_result = series.to_tensor().execute().fetch()
    np.testing.assert_array_equal(tensor_result, np.arange(10))

    # test from index
    index = md.Index(pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
    tensor_result = index.to_tensor(extract_multi_index=True).execute().fetch()
    np.testing.assert_array_equal(tensor_result, np.arange(6).reshape((3, 2)))

    index = md.Index(pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
    tensor_result = index.to_tensor(extract_multi_index=False).execute().fetch()
    np.testing.assert_array_equal(
        tensor_result, pd.MultiIndex.from_tuples([(0, 1), (2, 3), (4, 5)]).to_series()
    )


@pytest.mark.skipif(h5py is None, reason="h5py not installed")
def test_read_hdf5_execution(setup):
    test_array = np.random.RandomState(0).rand(20, 10)
    group_name = "test_group"
    dataset_name = "test_dataset"

    with pytest.raises(TypeError):
        fromhdf5(object())

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, f"test_read_{int(time.time())}.hdf5")
        with h5py.File(filename, "w") as f:
            g = f.create_group(group_name)
            g.create_dataset(dataset_name, chunks=(7, 4), data=test_array)

        # test filename
        r = fromhdf5(filename, group=group_name, dataset=dataset_name)

        result = r.execute().fetch()
        np.testing.assert_array_equal(result, test_array)
        assert r.extra_params["raw_chunk_size"] == (7, 4)

        with pytest.raises(ValueError):
            fromhdf5(filename)

        with pytest.raises(ValueError):
            fromhdf5(filename, dataset="non_exist")

        with h5py.File(filename, "r") as f:
            # test file
            r = fromhdf5(f, group=group_name, dataset=dataset_name)

            result = r.execute().fetch()
            np.testing.assert_array_equal(result, test_array)

            with pytest.raises(ValueError):
                fromhdf5(f)

            with pytest.raises(ValueError):
                fromhdf5(f, dataset="non_exist")

            # test dataset
            ds = f[f"{group_name}/{dataset_name}"]
            r = fromhdf5(ds)

            result = r.execute().fetch()
            np.testing.assert_array_equal(result, test_array)


@pytest.mark.skipif(zarr is None, reason="zarr not installed")
def test_read_zarr_execution(setup):
    session = setup

    test_array = np.random.RandomState(0).rand(20, 10)
    group_name = "test_group"
    dataset_name = "test_dataset"

    with pytest.raises(TypeError):
        fromzarr(object())

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"test_read_{int(time.time())}.zarr")

        group = zarr.group(path)
        arr = group.array(group_name + "/" + dataset_name, test_array, chunks=(7, 4))

        r = fromzarr(arr)

        result = r.execute().fetch()
        np.testing.assert_array_equal(result, test_array)
        assert len(session._session._tileable_to_fetch[r.data].chunks) > 1

        arr = zarr.open_array(f"{path}/{group_name}/{dataset_name}")
        r = fromzarr(arr)

        result = r.execute().fetch()
        np.testing.assert_array_equal(result, test_array)
        assert len(session._session._tileable_to_fetch[r.data].chunks) > 1

        r = fromzarr(path, group=group_name, dataset=dataset_name)

        result = r.execute().fetch()
        np.testing.assert_array_equal(result, test_array)
        assert len(session._session._tileable_to_fetch[r.data].chunks) > 1

        r = fromzarr(path + "/" + group_name + "/" + dataset_name)

        result = r.execute().fetch()
        np.testing.assert_array_equal(result, test_array)
        assert len(session._session._tileable_to_fetch[r.data].chunks) > 1
