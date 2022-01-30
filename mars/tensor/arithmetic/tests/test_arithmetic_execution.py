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

import functools
import operator

import numpy as np
import scipy.sparse as sps
import pytest

from ....config import option_context
from ....session import execute, fetch
from ....tests.core import require_cupy
from ....utils import ignore_warning
from ...datasource import ones, tensor, zeros
from .. import (
    add,
    cos,
    truediv,
    frexp,
    modf,
    clip,
    isclose,
    arctan2,
    tree_add,
    tree_multiply,
)


def _nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def _get_func(op):
    if isinstance(op, str):
        return getattr(np, op)
    return op


def _get_sparse_func(op):
    from ....lib.sparse.core import issparse

    if isinstance(op, str):
        op = getattr(np, op)

    def func(*args):
        new_args = []
        for arg in args:
            if issparse(arg):
                new_args.append(arg.toarray())
            else:
                new_args.append(arg)

        return op(*new_args)

    return func


def toarray(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def test_base_execution(setup):
    arr = ones((10, 8), chunk_size=2)
    arr2 = arr + 1

    res = arr2.execute().fetch()

    np.testing.assert_array_equal(res, np.ones((10, 8)) + 1)

    data = np.random.random((10, 8, 3))
    arr = tensor(data, chunk_size=2)
    arr2 = arr + 1

    res = arr2.execute().fetch()
    np.testing.assert_array_equal(res, data + 1)


def test_base_order_execution(setup):
    raw = np.asfortranarray(np.random.rand(5, 6))
    arr = tensor(raw, chunk_size=3)

    res = (arr + 1).execute().fetch()
    np.testing.assert_array_equal(res, raw + 1)
    assert res.flags["C_CONTIGUOUS"] is False
    assert res.flags["F_CONTIGUOUS"] is True

    res2 = add(arr, 1, order="C").execute().fetch()
    np.testing.assert_array_equal(res2, np.add(raw, 1, order="C"))
    assert res2.flags["C_CONTIGUOUS"] is True
    assert res2.flags["F_CONTIGUOUS"] is False


def test_ufunc_execution(setup):
    from .. import (
        UNARY_UFUNC,
        BIN_UFUNC,
        arccosh,
        invert,
        mod,
        fmod,
        bitand,
        bitor,
        bitxor,
        lshift,
        rshift,
        ldexp,
    )

    _sp_unary_ufunc = {arccosh, invert}
    _sp_bin_ufunc = {mod, fmod, bitand, bitor, bitxor, lshift, rshift, ldexp}

    data1 = np.random.random((5, 6, 2))
    data2 = np.random.random((5, 6, 2))
    rand = np.random.random()
    arr1 = tensor(data1, chunk_size=3)
    arr2 = tensor(data2, chunk_size=3)

    _new_unary_ufunc = UNARY_UFUNC - _sp_unary_ufunc
    for func in _new_unary_ufunc:
        res_tensor = func(arr1)
        assert res_tensor.dtype is not None
        res = res_tensor.execute().fetch()
        expected = _get_func(res_tensor.op._func_name)(data1)
        np.testing.assert_array_almost_equal(res, expected)

    _new_bin_ufunc = BIN_UFUNC - _sp_bin_ufunc
    for func in _new_bin_ufunc:
        res_tensor1 = func(arr1, arr2)
        assert res_tensor1.dtype is not None
        res_tensor2 = func(arr1, rand)
        assert res_tensor2.dtype is not None
        res_tensor3 = func(rand, arr1)
        assert res_tensor3.dtype is not None

        res1 = res_tensor1.execute().fetch()
        res2 = res_tensor2.execute().fetch()
        res3 = res_tensor3.execute().fetch()

        expected1 = _get_func(res_tensor1.op._func_name)(data1, data2)
        expected2 = _get_func(res_tensor1.op._func_name)(data1, rand)
        expected3 = _get_func(res_tensor1.op._func_name)(rand, data1)

        np.testing.assert_array_almost_equal(res1, expected1)
        np.testing.assert_array_almost_equal(res2, expected2)
        np.testing.assert_array_almost_equal(res3, expected3)

    data1 = np.random.randint(2, 10, size=(10, 10, 10))
    data2 = np.random.randint(2, 10, size=(10, 10, 10))
    rand = np.random.randint(1, 10)
    arr1 = tensor(data1, chunk_size=6)
    arr2 = tensor(data2, chunk_size=6)

    for func in _sp_unary_ufunc:
        res_tensor = func(arr1)
        assert res_tensor.dtype is not None
        res = res_tensor.execute().fetch()
        expected = _get_func(res_tensor.op._func_name)(data1)
        np.testing.assert_array_almost_equal(res, expected)

    for func in _sp_bin_ufunc:
        res_tensor1 = func(arr1, arr2)
        assert res_tensor1.dtype is not None
        res_tensor2 = func(arr1, rand)
        assert res_tensor2.dtype is not None
        res_tensor3 = func(rand, arr1)
        assert res_tensor3.dtype is not None

        res1 = res_tensor1.execute().fetch()
        res2 = res_tensor2.execute().fetch()
        res3 = res_tensor3.execute().fetch()

        expected1 = _get_func(res_tensor1.op._func_name)(data1, data2)
        expected2 = _get_func(res_tensor1.op._func_name)(data1, rand)
        expected3 = _get_func(res_tensor1.op._func_name)(rand, data1)

        np.testing.assert_array_almost_equal(res1, expected1)
        np.testing.assert_array_almost_equal(res2, expected2)
        np.testing.assert_array_almost_equal(res3, expected3)


def test_sparse_ufunc_execution(setup):
    from .. import add, square, arccosh, mod

    _normal_unary_ufunc = [square]
    _normal_bin_ufunc = [add]
    _sp_unary_ufunc = [arccosh]
    _sp_bin_ufunc = [mod]

    data1 = sps.random(5, 9, density=0.1)
    data2 = sps.random(5, 9, density=0.2)
    rand = np.random.random()
    arr1 = tensor(data1, chunk_size=3)
    arr2 = tensor(data2, chunk_size=3)

    for func in _normal_unary_ufunc:
        res_tensor = func(arr1)
        res = res_tensor.execute().fetch()
        expected = _get_sparse_func(res_tensor.op._func_name)(data1)
        _nan_equal(toarray(res[0]), expected)

    for func in _normal_bin_ufunc:
        res_tensor1 = func(arr1, arr2)
        res_tensor2 = func(arr1, rand)
        res_tensor3 = func(rand, arr1)

        res1 = res_tensor1.execute().fetch()
        res2 = res_tensor2.execute().fetch()
        res3 = res_tensor3.execute().fetch()

        expected1 = _get_sparse_func(res_tensor1.op._func_name)(data1, data2)
        expected2 = _get_sparse_func(res_tensor1.op._func_name)(data1, rand)
        expected3 = _get_sparse_func(res_tensor1.op._func_name)(rand, data1)

        _nan_equal(toarray(res1[0]), expected1)
        _nan_equal(toarray(res2[0]), expected2)
        _nan_equal(toarray(res3[0]), expected3)

    data1 = np.random.randint(2, 10, size=(10, 10))
    data2 = np.random.randint(2, 10, size=(10, 10))
    rand = np.random.randint(1, 10)
    arr1 = tensor(data1, chunk_size=3).tosparse()
    arr2 = tensor(data2, chunk_size=3).tosparse()

    for func in _sp_unary_ufunc:
        res_tensor = func(arr1)
        res = res_tensor.execute().fetch()
        expected = _get_sparse_func(res_tensor.op._func_name)(data1)
        _nan_equal(toarray(res[0]), expected)

    for func in _sp_bin_ufunc:
        res_tensor1 = func(arr1, arr2)
        res_tensor2 = func(arr1, rand)
        res_tensor3 = func(rand, arr1)

        res1 = res_tensor1.execute().fetch()
        res2 = res_tensor2.execute().fetch()
        res3 = res_tensor3.execute().fetch()
        expected1 = _get_sparse_func(res_tensor1.op._func_name)(data1, data2)
        expected2 = _get_sparse_func(res_tensor1.op._func_name)(data1, rand)
        expected3 = _get_sparse_func(res_tensor1.op._func_name)(rand, data1)

        _nan_equal(toarray(res1[0]), expected1)
        _nan_equal(toarray(res2[0]), expected2)
        _nan_equal(toarray(res3[0]), expected3)


def test_add_with_out_execution(setup):
    data1 = np.random.random((5, 9, 4))
    data2 = np.random.random((9, 4))

    arr1 = tensor(data1.copy(), chunk_size=3)
    arr2 = tensor(data2.copy(), chunk_size=3)

    add(arr1, arr2, out=arr1)
    res = arr1.execute().fetch()
    np.testing.assert_array_equal(res, data1 + data2)

    arr1 = tensor(data1.copy(), chunk_size=3)
    arr2 = tensor(data2.copy(), chunk_size=3)

    arr3 = add(arr1, arr2, out=arr1.astype("i4"), casting="unsafe")
    res = arr3.execute().fetch()
    np.testing.assert_array_equal(res, (data1 + data2).astype("i4"))

    arr1 = tensor(data1.copy(), chunk_size=3)
    arr2 = tensor(data2.copy(), chunk_size=3)

    arr3 = truediv(arr1, arr2, out=arr1, where=arr2 > 0.5)
    res = arr3.execute().fetch()
    np.testing.assert_array_equal(
        res, np.true_divide(data1, data2, out=data1.copy(), where=data2 > 0.5)
    )

    arr1 = tensor(data1.copy(), chunk_size=4)
    arr2 = tensor(data2.copy(), chunk_size=4)

    arr3 = add(arr1, arr2, where=arr1 > 0.5)
    res = arr3.execute().fetch()
    expected = np.add(data1, data2, where=data1 > 0.5)
    np.testing.assert_array_equal(res[data1 > 0.5], expected[data1 > 0.5])

    arr1 = tensor(data1.copy(), chunk_size=4)

    arr3 = add(arr1, 1, where=arr1 > 0.5)
    res = arr3.execute().fetch()
    expected = np.add(data1, 1, where=data1 > 0.5)
    np.testing.assert_array_equal(res[data1 > 0.5], expected[data1 > 0.5])

    arr1 = tensor(data2.copy(), chunk_size=3)

    arr3 = add(arr1[:5, :], 1, out=arr1[-5:, :])
    res = arr3.execute().fetch()
    expected = np.add(data2[:5, :], 1)
    np.testing.assert_array_equal(res, expected)


def test_arctan2_execution(setup):
    x = tensor(1)  # scalar
    y = arctan2(x, x)

    assert y.issparse() is False
    result = y.execute().fetch()
    np.testing.assert_equal(result, np.arctan2(1, 1))

    y = arctan2(0, x)

    assert y.issparse() is False
    result = y.execute().fetch()
    np.testing.assert_equal(result, np.arctan2(0, 1))

    raw1 = np.array([[0, 1, 2]])
    raw2 = sps.csr_matrix([[0, 1, 0]])
    y = arctan2(raw1, raw2)

    assert y.issparse() is False
    result = y.execute().fetch()
    np.testing.assert_equal(result, np.arctan2(raw1, raw2.A))

    y = arctan2(raw2, raw2)

    assert y.issparse() is True
    result = y.execute().fetch()
    np.testing.assert_equal(result, np.arctan2(raw2.A, raw2.A))

    y = arctan2(0, raw2)

    assert y.issparse() is True
    result = y.execute().fetch()
    np.testing.assert_equal(result, np.arctan2(0, raw2.A))


def test_frexp_execution(setup):
    data1 = np.random.RandomState(0).randint(0, 100, (5, 9, 6))

    arr1 = tensor(data1.copy(), chunk_size=4)

    o1, o2 = frexp(arr1)
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.frexp(data1))
    np.testing.assert_array_almost_equal(res, expected)

    arr1 = tensor(data1.copy(), chunk_size=4)
    o1 = zeros(data1.shape, chunk_size=4)
    o2 = zeros(data1.shape, dtype="i8", chunk_size=4)
    frexp(arr1, o1, o2)
    res1, res2 = fetch(*execute(o1, o2))

    res = res1 * 2**res2
    np.testing.assert_array_almost_equal(res, data1, decimal=3)

    data1 = sps.random(5, 9, density=0.1)

    arr1 = tensor(data1.copy(), chunk_size=4)

    o1, o2 = frexp(arr1)
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.frexp(data1.toarray()))
    np.testing.assert_equal(res.toarray(), expected)


def test_frexp_order_execution(setup):
    data1 = np.random.RandomState(0).random((5, 9))
    t = tensor(data1, chunk_size=3)

    o1, o2 = frexp(t, order="F")
    res1, res2 = execute(o1, o2)
    expected1, expected2 = np.frexp(data1, order="F")
    np.testing.assert_allclose(res1, expected1)
    assert res1.flags["F_CONTIGUOUS"] is True
    assert res1.flags["C_CONTIGUOUS"] is False
    np.testing.assert_allclose(res2, expected2)
    assert res2.flags["F_CONTIGUOUS"] is True
    assert res2.flags["C_CONTIGUOUS"] is False


def test_modf_execution(setup):
    data1 = np.random.random((5, 9))

    arr1 = tensor(data1.copy(), chunk_size=3)

    o1, o2 = modf(arr1)
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.modf(data1))
    np.testing.assert_array_almost_equal(res, expected)

    o1, o2 = modf([0, 3.5])
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.modf([0, 3.5]))
    np.testing.assert_array_almost_equal(res, expected)

    arr1 = tensor(data1.copy(), chunk_size=3)
    o1 = zeros(data1.shape, chunk_size=3)
    o2 = zeros(data1.shape, chunk_size=3)
    modf(arr1, o1, o2)
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.modf(data1))
    np.testing.assert_array_almost_equal(res, expected)

    data1 = sps.random(5, 9, density=0.1)

    arr1 = tensor(data1.copy(), chunk_size=3)

    o1, o2 = modf(arr1)
    o = o1 + o2

    res = o.execute().fetch()
    expected = sum(np.modf(data1.toarray()))
    np.testing.assert_equal(res.toarray(), expected)


def test_modf_order_execution(setup):
    data1 = np.random.random((5, 9))
    t = tensor(data1, chunk_size=3)

    o1, o2 = modf(t, order="F")
    res1, res2 = execute(o1, o2)
    expected1, expected2 = np.modf(data1, order="F")
    np.testing.assert_allclose(res1, expected1)
    assert res1.flags["F_CONTIGUOUS"] is True
    assert res1.flags["C_CONTIGUOUS"] is False
    np.testing.assert_allclose(res2, expected2)
    assert res2.flags["F_CONTIGUOUS"] is True
    assert res2.flags["C_CONTIGUOUS"] is False


def test_clip_execution(setup):
    a_data = np.arange(10)

    a = tensor(a_data.copy(), chunk_size=3)

    b = clip(a, 1, 8)

    res = b.execute().fetch()
    expected = np.clip(a_data, 1, 8)
    np.testing.assert_array_equal(res, expected)

    a = tensor(a_data.copy(), chunk_size=3)
    clip(a, 3, 6, out=a)

    res = a.execute().fetch()
    expected = np.clip(a_data, 3, 6)
    np.testing.assert_array_equal(res, expected)

    a = tensor(a_data.copy(), chunk_size=3)
    a_min_data = np.random.randint(1, 10, size=(10,))
    a_max_data = np.random.randint(1, 10, size=(10,))
    a_min = tensor(a_min_data)
    a_max = tensor(a_max_data)
    clip(a, a_min, a_max, out=a)

    res = a.execute().fetch()
    expected = np.clip(a_data, a_min_data, a_max_data)
    np.testing.assert_array_equal(res, expected)

    with option_context() as options:
        options.chunk_size = 3

        a = tensor(a_data.copy(), chunk_size=3)
        b = clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)

        res = b.execute().fetch()
        expected = np.clip(a_data, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
        np.testing.assert_array_equal(res, expected)

        # test sparse clip
        a_data = sps.csr_matrix([[0, 2, 8], [0, 0, -1]])
        a = tensor(a_data, chunk_size=3)
        b_data = sps.csr_matrix([[0, 3, 0], [1, 0, -2]])

        c = clip(a, b_data, 4)

        res = c.execute().fetch()
        expected = np.clip(a_data.toarray(), b_data.toarray(), 4)
        np.testing.assert_array_equal(res, expected)


def test_clip_order_execution(setup):
    a_data = np.asfortranarray(np.random.rand(4, 8))

    a = tensor(a_data, chunk_size=3)

    b = clip(a, 0.2, 0.8)

    res = b.execute().fetch()
    expected = np.clip(a_data, 0.2, 0.8)

    np.testing.assert_allclose(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False


def test_around_execution(setup):
    data = np.random.randn(10, 20)
    x = tensor(data, chunk_size=3)

    t = x.round(2)

    res = t.execute().fetch()
    expected = np.around(data, decimals=2)

    np.testing.assert_allclose(res, expected)

    data = sps.random(10, 20, density=0.2)
    x = tensor(data, chunk_size=3)

    t = x.round(2)

    res = t.execute().fetch()
    expected = np.around(data.toarray(), decimals=2)

    np.testing.assert_allclose(res.toarray(), expected)


def test_around_order_execution(setup):
    data = np.asfortranarray(np.random.rand(10, 20))
    x = tensor(data, chunk_size=3)

    t = x.round(2)

    res = t.execute().fetch()
    expected = np.around(data, decimals=2)

    np.testing.assert_allclose(res, expected)
    assert res.flags["F_CONTIGUOUS"] is True
    assert res.flags["C_CONTIGUOUS"] is False


def test_cos_order_execution(setup):
    data = np.asfortranarray(np.random.rand(3, 5))
    x = tensor(data, chunk_size=2)

    t = cos(x)

    res = t.execute().fetch()
    np.testing.assert_allclose(res, np.cos(data))
    assert res.flags["C_CONTIGUOUS"] is False
    assert res.flags["F_CONTIGUOUS"] is True

    t2 = cos(x, order="C")

    res2 = t2.execute().fetch()
    np.testing.assert_allclose(res2, np.cos(data, order="C"))
    assert res2.flags["C_CONTIGUOUS"] is True
    assert res2.flags["F_CONTIGUOUS"] is False


def test_is_close_execution(setup):
    data = np.array([1.05, 1.0, 1.01, np.nan])
    data2 = np.array([1.04, 1.0, 1.03, np.nan])

    x = tensor(data, chunk_size=2)
    y = tensor(data2, chunk_size=3)

    z = isclose(x, y, atol=0.01)

    res = z.execute().fetch()
    expected = np.isclose(data, data2, atol=0.01)
    np.testing.assert_equal(res, expected)

    z = isclose(x, y, atol=0.01, equal_nan=True)

    res = z.execute().fetch()
    expected = np.isclose(data, data2, atol=0.01, equal_nan=True)
    np.testing.assert_equal(res, expected)

    # test tensor with scalar
    z = isclose(x, 1.0, atol=0.01)
    res = z.execute().fetch()
    expected = np.isclose(data, 1.0, atol=0.01)
    np.testing.assert_equal(res, expected)
    z = isclose(1.0, y, atol=0.01)
    res = z.execute().fetch()
    expected = np.isclose(1.0, data2, atol=0.01)
    np.testing.assert_equal(res, expected)
    z = isclose(1.0, 2.0, atol=0.01)
    res = z.execute().fetch()
    expected = np.isclose(1.0, 2.0, atol=0.01)
    np.testing.assert_equal(res, expected)

    # test sparse
    data = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
    data2 = sps.csr_matrix(np.array([0, 1.0, 1.03, np.nan]))

    x = tensor(data, chunk_size=2)
    y = tensor(data2, chunk_size=3)

    z = isclose(x, y, atol=0.01)

    res = z.execute().fetch()
    expected = np.isclose(data.toarray(), data2.toarray(), atol=0.01)
    np.testing.assert_equal(res, expected)

    z = isclose(x, y, atol=0.01, equal_nan=True)

    res = z.execute().fetch()
    expected = np.isclose(data.toarray(), data2.toarray(), atol=0.01, equal_nan=True)
    np.testing.assert_equal(res, expected)


@ignore_warning
def test_dtype_execution(setup):
    a = ones((10, 20), dtype="f4", chunk_size=5)

    c = truediv(a, 2, dtype="f8")

    res = c.execute().fetch()
    assert res.dtype == np.float64

    c = truediv(a, 0, dtype="f8")
    res = c.execute().fetch()
    assert np.isinf(res[0, 0])

    with pytest.raises(FloatingPointError):
        with np.errstate(divide="raise"):
            c = truediv(a, 0, dtype="f8")
            _ = c.execute().fetch()  # noqa: F841


def test_set_get_real_execution(setup):
    a_data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    a = tensor(a_data, chunk_size=2)

    res = a.real.execute().fetch()
    expected = a_data.real

    np.testing.assert_equal(res, expected)

    a.real = 9

    res = a.execute().fetch()
    expected = a_data.copy()
    expected.real = 9

    np.testing.assert_equal(res, expected)

    a.real = np.array([9, 8, 7])

    res = a.execute().fetch()
    expected = a_data.copy()
    expected.real = np.array([9, 8, 7])

    np.testing.assert_equal(res, expected)

    # test sparse
    a_data = np.array([[1 + 2j, 3 + 4j, 0], [0, 0, 0]])
    a = tensor(sps.csr_matrix(a_data))

    res = a.real.execute().fetch().toarray()
    expected = a_data.real

    np.testing.assert_equal(res, expected)

    a.real = 9

    res = a.execute().fetch().toarray()
    expected = a_data.copy()
    expected.real = 9

    np.testing.assert_equal(res, expected)

    a.real = np.array([9, 8, 7])

    res = a.execute().fetch().toarray()
    expected = a_data.copy()
    expected.real = np.array([9, 8, 7])

    np.testing.assert_equal(res, expected)


def test_set_get_imag_execution(setup):
    a_data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    a = tensor(a_data, chunk_size=2)

    res = a.imag.execute().fetch()
    expected = a_data.imag

    np.testing.assert_equal(res, expected)

    a.imag = 9

    res = a.execute().fetch()
    expected = a_data.copy()
    expected.imag = 9

    np.testing.assert_equal(res, expected)

    a.imag = np.array([9, 8, 7])

    res = a.execute().fetch()
    expected = a_data.copy()
    expected.imag = np.array([9, 8, 7])

    np.testing.assert_equal(res, expected)

    # test sparse
    a_data = np.array([[1 + 2j, 3 + 4j, 0], [0, 0, 0]])
    a = tensor(sps.csr_matrix(a_data))

    res = a.imag.execute().fetch().toarray()
    expected = a_data.imag

    np.testing.assert_equal(res, expected)

    a.imag = 9

    res = a.execute().fetch().toarray()
    expected = a_data.copy()
    expected.imag = 9

    np.testing.assert_equal(res, expected)

    a.imag = np.array([9, 8, 7])

    res = a.execute().fetch().toarray()
    expected = a_data.copy()
    expected.imag = np.array([9, 8, 7])

    np.testing.assert_equal(res, expected)


def test_tree_arithmetic_execution(setup):
    raws = [np.random.rand(10, 10) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    res = tree_add(*tensors, 1.0).execute().fetch()
    np.testing.assert_array_almost_equal(
        res, 1.0 + functools.reduce(operator.add, raws)
    )

    res = tree_multiply(*tensors, 2.0).execute().fetch()
    np.testing.assert_array_almost_equal(
        res, 2.0 * functools.reduce(operator.mul, raws)
    )

    raws = [sps.random(5, 9, density=0.1) for _ in range(10)]
    tensors = [tensor(a, chunk_size=3) for a in raws]

    res = tree_add(*tensors).execute().fetch()
    np.testing.assert_array_almost_equal(
        res.toarray(), functools.reduce(operator.add, raws).toarray()
    )


@require_cupy
def test_cupy_execution(setup_gpu):
    a_data = np.random.rand(10, 10)
    b_data = np.random.rand(10, 10)

    a = tensor(a_data, gpu=True, chunk_size=3)
    b = tensor(b_data, gpu=True, chunk_size=3)
    res_binary = (a + b).execute().fetch()
    np.testing.assert_array_equal(res_binary.get(), (a_data + b_data))

    res_unary = cos(a).execute().fetch()
    np.testing.assert_array_almost_equal(res_unary.get(), np.cos(a_data))
