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

from ... import sparse as mls
from .. import SparseNDArray, SparseVector, SparseMatrix
from ..core import issparse


s1_data = sps.csr_matrix([[1, 0, 1], [0, 0, 1]])
s2_data = sps.csr_matrix([[0, 1, 1], [1, 0, 1]])
v1_data = np.random.rand(3)
v1 = sps.csr_matrix(v1_data)
v2_data = np.random.rand(2)
v2 = sps.csr_matrix(v2_data)
d1 = np.array([1, 2, 3])


def assertArrayEqual(a, b, almost=False):
    if issparse(a):
        a = a.toarray()
    else:
        a = np.asarray(a)
    if issparse(b):
        b = b.toarray()
    else:
        b = np.asarray(b)
    if not almost:
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_almost_equal(a, b)


def test_sparse_creation():
    s = SparseNDArray(s1_data)
    assert s.ndim == 2
    assert isinstance(s, SparseMatrix)
    assertArrayEqual(s.toarray(), s1_data.A)
    assertArrayEqual(s.todense(), s1_data.A)

    v = SparseNDArray(v1, shape=(3,))
    assert s.ndim
    assert isinstance(v, SparseVector)
    assert v.shape == (3,)
    assertArrayEqual(v.todense(), v1_data)
    assertArrayEqual(v.toarray(), v1_data)
    assertArrayEqual(v, v1_data)


def test_sparse_add():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 + s2, s1 + s2)
    assertArrayEqual(s1 + d1, s1 + d1)
    assertArrayEqual(d1 + s1, d1 + s1)
    r = sps.csr_matrix(((s1.data + 1), s1.indices, s1.indptr), s1.shape)
    assertArrayEqual(s1 + 1, r)
    r = sps.csr_matrix(((1 + s1.data), s1.indices, s1.indptr), s1.shape)
    assertArrayEqual(1 + s1, r)

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v + v, v1_data + v1_data)
    assertArrayEqual(v + d1, v1_data + d1)
    assertArrayEqual(d1 + v, d1 + v1_data)
    r = sps.csr_matrix(((v1.data + 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v + 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 + v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 + v, r.toarray().reshape(3))


def test_sparse_subtract():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 - s2, s1 - s2)
    assertArrayEqual(s1 - d1, s1 - d1)
    assertArrayEqual(d1 - s1, d1 - s1)
    r = sps.csr_matrix(((s1.data - 1), s1.indices, s1.indptr), s1.shape)
    assertArrayEqual(s1 - 1, r)
    r = sps.csr_matrix(((1 - s1.data), s1.indices, s1.indptr), s1.shape)
    assertArrayEqual(1 - s1, r)

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v - v, v1_data - v1_data)
    assertArrayEqual(v - d1, v1_data - d1)
    assertArrayEqual(d1 - v, d1 - v1_data)
    r = sps.csr_matrix(((v1.data - 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v - 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 - v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 - v, r.toarray().reshape(3))


def test_sparse_multiply():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 * s2, s1_data.multiply(s2_data))
    assertArrayEqual(s1 * d1, s1_data.multiply(d1))
    assertArrayEqual(d1 * s1, s1_data.multiply(d1))
    assertArrayEqual(s1 * 2, s1 * 2)
    assertArrayEqual(2 * s1, s1 * 2)

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v * v, v1_data * v1_data)
    assertArrayEqual(v * d1, v1_data * d1)
    assertArrayEqual(d1 * v, d1 * v1_data)
    r = sps.csr_matrix(((v1.data * 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v * 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 * v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 * v, r.toarray().reshape(3))


def test_sparse_divide():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 / s2, s1 / s2)
    assertArrayEqual(s1 / d1, s1 / d1)
    assertArrayEqual(d1 / s1, d1 / s1.toarray())
    assertArrayEqual(s1 / 2, s1 / 2)
    assertArrayEqual(2 / s1, 2 / s1.toarray())

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v / v, v1_data / v1_data)
    assertArrayEqual(v / d1, v1_data / d1)
    assertArrayEqual(d1 / v, d1 / v1_data)
    r = sps.csr_matrix(((v1.data / 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v / 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 / v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 / v, r.toarray().reshape(3))


def test_sparse_floor_divide():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 // s2, s1.toarray() // s2.toarray())
    assertArrayEqual(s1 // d1, s1.toarray() // d1)
    assertArrayEqual(d1 // s1, d1 // s1.toarray())
    assertArrayEqual(s1 // 2, s1.toarray() // 2)
    assertArrayEqual(2 // s1, 2 // s1.toarray())

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v // v, v1_data // v1_data)
    assertArrayEqual(v // d1, v1_data // d1)
    assertArrayEqual(d1 // v, d1 // v1_data)
    r = sps.csr_matrix(((v1.data // 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v // 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 // v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 // v, r.toarray().reshape(3))


def test_sparse_power():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1**s2, s1.toarray() ** s2.toarray())
    assertArrayEqual(s1**d1, s1.toarray() ** d1)
    assertArrayEqual(d1**s1, d1 ** s1.toarray())
    assertArrayEqual(s1**2, s1_data.power(2))
    assertArrayEqual(2**s1, 2 ** s1.toarray())

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v**v, v1_data**v1_data)
    assertArrayEqual(v**d1, v1_data**d1)
    assertArrayEqual(d1**v, d1**v1_data)
    r = sps.csr_matrix(((v1.data**1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v**1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1**v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1**v, r.toarray().reshape(3))


def test_sparse_mod():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    assertArrayEqual(s1 % s2, s1.toarray() % s2.toarray())
    assertArrayEqual(s1 % d1, s1.toarray() % d1)
    assertArrayEqual(d1 % s1, d1 % s1.toarray())
    assertArrayEqual(s1 % 2, s1.toarray() % 2)
    assertArrayEqual(2 % s1, 2 % s1.toarray())

    # test sparse vector
    v = SparseNDArray(v1, shape=(3,))
    assertArrayEqual(v % v, v1_data % v1_data)
    assertArrayEqual(v % d1, v1_data % d1)
    assertArrayEqual(d1 % v, d1 % v1_data)
    r = sps.csr_matrix(((v1.data % 1), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(v % 1, r.toarray().reshape(3))
    r = sps.csr_matrix(((1 % v1.data), v1.indices, v1.indptr), v1.shape)
    assertArrayEqual(1 % v, r.toarray().reshape(3))


def test_sparse_bin():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)
    v = SparseNDArray(v1, shape=(3,))

    for method in (
        "fmod",
        "logaddexp",
        "logaddexp2",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "hypot",
        "arctan2",
    ):
        lm, rm = getattr(mls, method), getattr(np, method)
        assertArrayEqual(lm(s1, s2), rm(s1.toarray(), s2.toarray()))
        assertArrayEqual(lm(s1, d1), rm(s1.toarray(), d1))
        assertArrayEqual(lm(d1, s1), rm(d1, s1.toarray()))
        r1 = sps.csr_matrix((rm(s1.data, 2), s1.indices, s1.indptr), s1.shape)
        assertArrayEqual(lm(s1, 2), r1)
        r2 = sps.csr_matrix((rm(2, s1.data), s1.indices, s1.indptr), s1.shape)
        assertArrayEqual(lm(2, s1), r2)

        # test sparse
        assertArrayEqual(lm(v, v), rm(v1_data, v1_data))
        assertArrayEqual(lm(v, d1), rm(v1_data, d1))
        assertArrayEqual(lm(d1, v), rm(d1, v1_data))
        assertArrayEqual(lm(v, 2), rm(v1_data, 2))
        assertArrayEqual(lm(2, v), rm(2, v1_data))


def test_sparse_unary():
    s1 = SparseNDArray(s1_data)
    v = SparseNDArray(v1, shape=(3,))

    for method in (
        "negative",
        "positive",
        "absolute",
        "abs",
        "fabs",
        "rint",
        "sign",
        "conj",
        "exp",
        "exp2",
        "log",
        "log2",
        "log10",
        "expm1",
        "log1p",
        "sqrt",
        "square",
        "cbrt",
        "reciprocal",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "deg2rad",
        "rad2deg",
        "angle",
        "isnan",
        "isinf",
        "signbit",
        "sinc",
        "isreal",
        "isfinite",
    ):
        lm, rm = getattr(mls, method), getattr(np, method)
        r = sps.csr_matrix((rm(s1.data), s1.indices, s1.indptr), s1.shape)
        assertArrayEqual(lm(s1), r)
        assertArrayEqual(lm(v), rm(v1_data))


def test_sparse_dot():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)
    v1_s = SparseNDArray(v1, shape=(3,))
    v2_s = SparseNDArray(v2, shape=(2,))

    assertArrayEqual(mls.dot(s1, s2.T), s1.dot(s2.T))
    assertArrayEqual(s1.dot(d1), s1.dot(d1))
    assertArrayEqual(d1.dot(s1.T), d1.dot(s1.T.toarray()))

    assertArrayEqual(s1 @ s2.T, s1_data @ s2_data.T)

    assertArrayEqual(mls.tensordot(s1, s2.T, axes=(1, 0)), s1.dot(s2.T))
    assertArrayEqual(mls.tensordot(s1, d1, axes=(1, -1)), s1.dot(d1))
    assertArrayEqual(mls.tensordot(d1, s1.T, axes=(0, 0)), d1.dot(s1.T.toarray()))

    assertArrayEqual(mls.dot(s1, v1_s), s1.dot(v1_data))
    assertArrayEqual(mls.dot(s2, v1_s), s2.dot(v1_data))
    assertArrayEqual(mls.dot(v2_s, s1), v2_data.dot(s1_data.A))
    assertArrayEqual(mls.dot(v2_s, s2), v2_data.dot(s2_data.A))
    assertArrayEqual(mls.dot(v1_s, v1_s), v1_data.dot(v1_data), almost=True)
    assertArrayEqual(mls.dot(v2_s, v2_s), v2_data.dot(v2_data), almost=True)

    assertArrayEqual(mls.dot(v2_s, s1, sparse=False), v2_data.dot(s1_data.A))
    assertArrayEqual(mls.dot(v1_s, v1_s, sparse=False), v1_data.dot(v1_data))


def test_sparse_sum():
    s1 = SparseNDArray(s1_data)
    v = SparseNDArray(v1, shape=(3,))
    assert s1.sum() == s1.sum()
    np.testing.assert_array_equal(s1.sum(axis=1), np.asarray(s1.sum(axis=1)).reshape(2))
    np.testing.assert_array_equal(s1.sum(axis=0), np.asarray(s1.sum(axis=0)).reshape(3))
    np.testing.assert_array_equal(v.sum(), np.asarray(v1_data.sum()))


def test_sparse_setitem():
    s1 = SparseNDArray(s1_data.copy())
    s1[1:2, 1] = [2]
    ss1 = s1_data.tolil()
    ss1[1:2, 1] = [2]
    np.testing.assert_array_equal(s1.toarray(), ss1.toarray())

    v = SparseVector(v1, shape=(3,))
    v[1:2] = [2]
    vv1 = v1_data.copy()
    vv1[1:2] = [2]
    np.testing.assert_array_equal(v.toarray(), vv1)


def test_sparse_maximum():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    np.testing.assert_array_equal(s1.maximum(s2).toarray(), s1.maximum(s2).toarray())

    v = SparseVector(v1, shape=(3,))
    np.testing.assert_array_equal(v.maximum(d1), np.maximum(v1_data, d1))


def test_sparse_minimum():
    s1 = SparseNDArray(s1_data)
    s2 = SparseNDArray(s2_data)

    np.testing.assert_array_equal(s1.minimum(s2).toarray(), s1.minimum(s2).toarray())

    v = SparseVector(v1, shape=(3,))
    np.testing.assert_array_equal(v.minimum(d1), np.minimum(v1_data, d1))


def test_sparse_fill_diagonal():
    s1 = sps.random(100, 11, density=0.3, format="csr", random_state=0)

    # fill scalar
    arr = SparseNDArray(s1)
    arr.fill_diagonal(3)

    expected = s1.copy().A
    np.fill_diagonal(expected, 3)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill scalar, wrap=True
    arr = SparseNDArray(s1)
    arr.fill_diagonal(3, wrap=True)

    expected = s1.copy().A
    np.fill_diagonal(expected, 3, wrap=True)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill list
    arr = SparseNDArray(s1)
    arr.fill_diagonal([1, 2, 3])

    expected = s1.copy().A
    np.fill_diagonal(expected, [1, 2, 3])

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill list, wrap=True
    arr = SparseNDArray(s1)
    arr.fill_diagonal([1, 2, 3], wrap=True)

    expected = s1.copy().A
    np.fill_diagonal(expected, [1, 2, 3], wrap=True)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill long list
    val = np.random.RandomState(0).rand(101)
    arr = SparseNDArray(s1)
    arr.fill_diagonal(val)

    expected = s1.copy().A
    np.fill_diagonal(expected, val)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill long list, wrap=True
    val = np.random.RandomState(0).rand(101)
    arr = SparseNDArray(s1)
    arr.fill_diagonal(val, wrap=True)

    expected = s1.copy().A
    np.fill_diagonal(expected, val, wrap=True)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill ndarray
    val = np.random.RandomState(0).rand(3, 4)
    arr = SparseNDArray(s1)
    arr.fill_diagonal(val)

    expected = s1.copy().A
    np.fill_diagonal(expected, val)

    np.testing.assert_array_equal(arr.toarray(), expected)

    # fill ndarray, wrap=True
    val = np.random.RandomState(0).rand(3, 4)
    arr = SparseNDArray(s1)
    arr.fill_diagonal(val, wrap=True)

    expected = s1.copy().A
    np.fill_diagonal(expected, val, wrap=True)

    np.testing.assert_array_equal(arr.toarray(), expected)
