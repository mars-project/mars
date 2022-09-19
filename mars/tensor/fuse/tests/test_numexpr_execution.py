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

from ....utils import ignore_warning
from ...arithmetic import abs as mt_abs
from ...datasource import tensor, arange
from ...reduction import sum as mt_sum


def test_base_execution(setup):
    rs = np.random.RandomState(0)
    raw1 = rs.randint(10, size=(10, 10, 10))
    raw2 = rs.randint(10, size=(10, 10, 10))
    arr1 = tensor(raw1, chunk_size=5)
    arr2 = tensor(raw2, chunk_size=5)

    arr3 = arr1 + arr2 + 10
    arr4 = 10 + arr1 + arr2
    res3 = arr3.execute().fetch()
    res3_cmp = arr4.execute().fetch()
    np.testing.assert_array_equal(res3, res3_cmp)

    a = arange(10)
    b = arange(10) * 0.1
    raw_a = np.arange(10)
    raw_b = np.arange(10) * 0.1
    c = a * b - 4.1 * a > 2.5 * b
    res4_cmp = raw_a * raw_b - 4.1 * raw_a > 2.5 * raw_b
    res4 = c.execute().fetch()
    np.testing.assert_array_equal(res4, res4_cmp)

    c = mt_sum(1) * (-1)
    r = c.execute().fetch()
    assert r == -1

    c = -mt_abs(mt_sum(mt_abs(-1)))
    r = c.execute().fetch()
    assert r == -1


def _gen_pairs(seq):
    test_seq = np.random.RandomState(0).permutation(seq)
    for i in range(0, len(seq), 2):
        j = (i + 1) % len(seq)
        yield test_seq[i], test_seq[j]


@ignore_warning
def test_unary_execution(setup):
    from ...arithmetic import UNARY_UFUNC, arccosh, invert, sin, conj, logical_not

    _sp_unary_ufunc = {arccosh, invert, conj, logical_not}
    _new_unary_ufunc = list(UNARY_UFUNC - _sp_unary_ufunc)[:3]

    def _normalize_by_sin(func1, func2, arr):
        return func1(abs(sin((func2(arr)))))

    tested = set()
    rs = np.random.RandomState(0)
    for func1, func2 in _gen_pairs(_new_unary_ufunc):
        raw = rs.random((8, 8, 8))
        arr1 = tensor(raw, chunk_size=4)

        arr2 = _normalize_by_sin(func1, func2, arr1)
        res = arr2.execute()
        res_cmp = arr2.execute(fuse_enabled=False)
        np.testing.assert_allclose(res[0], res_cmp[0])
        tested.update([func1, func2])
    # make sure all functions tested
    assert tested == set(_new_unary_ufunc)

    raw = rs.randint(100, size=(8, 8, 8))
    arr1 = tensor(raw, chunk_size=4)
    arr2 = arccosh(1 + abs(invert(arr1)))
    res = arr2.execute(fuse_enabled=False).fetch()
    res_cmp = arccosh(1 + abs(~raw))
    np.testing.assert_array_almost_equal(res[0], res_cmp[0])


@ignore_warning
def test_bin_execution(setup):
    from ...arithmetic import (
        BIN_UFUNC,
        mod,
        fmod,
        bitand,
        bitor,
        bitxor,
        lshift,
        rshift,
        ldexp,
        logical_and,
        logical_or,
    )

    _sp_bin_ufunc = [
        mod,
        fmod,
        bitand,
        bitor,
        bitxor,
        lshift,
        rshift,
        logical_and,
        logical_or,
    ]
    _new_bin_ufunc = list(BIN_UFUNC - set(_sp_bin_ufunc) - {ldexp})

    tested = set()
    rs = np.random.RandomState(0)
    for func1, func2 in _gen_pairs(_new_bin_ufunc):
        raw = rs.random((9, 9, 9))
        arr1 = tensor(raw, chunk_size=5)

        arr2 = func1(1, func2(2, arr1))
        res = arr2.execute().fetch()
        res_cmp = arr2.execute(fuse_enabled=False).fetch()
        np.testing.assert_array_almost_equal(res, res_cmp)
        tested.update([func1, func2])
    # make sure all functions tested
    assert tested == set(_new_bin_ufunc)

    tested = set()
    for func1, func2 in _gen_pairs(_sp_bin_ufunc):
        raw = rs.randint(1, 100, size=(10, 10, 10))
        arr1 = tensor(raw, chunk_size=6)

        arr2 = func1(10, func2(arr1, 5))
        res = arr2.execute().fetch()
        res_cmp = arr2.execute(fuse_enabled=False).fetch()
        np.testing.assert_array_almost_equal(res, res_cmp)
        tested.update([func1, func2])
    # make sure all functions tested
    assert tested == set(_sp_bin_ufunc)


def test_reduction_execution(setup):
    rs = np.random.RandomState(0)
    raw1 = rs.randint(5, size=(8, 8, 8))
    raw2 = rs.randint(5, size=(8, 8, 8))
    arr1 = tensor(raw1, chunk_size=4)
    arr2 = tensor(raw2, chunk_size=4)

    res1 = (arr1 + 1).sum(keepdims=True).execute().fetch()
    res2 = (arr1 + 1).prod(keepdims=True).execute().fetch()
    np.testing.assert_array_equal((raw1 + 1).sum(keepdims=True), res1)
    np.testing.assert_array_equal((raw1 + 1).prod(keepdims=True), res2)

    res1 = (arr1 + 1).sum(axis=1).execute().fetch()
    res2 = (arr1 + 1).prod(axis=1).execute().fetch()
    res3 = (arr1 + 1).max(axis=1).execute().fetch()
    res4 = (arr1 + 1).min(axis=1).execute().fetch()
    np.testing.assert_array_equal((raw1 + 1).sum(axis=1), res1)
    np.testing.assert_array_equal((raw1 + 1).prod(axis=1), res2)
    np.testing.assert_array_equal((raw1 + 1).max(axis=1), res3)
    np.testing.assert_array_equal((raw1 + 1).min(axis=1), res4)

    raw3 = raw2 - raw1 + 10
    arr3 = -arr1 + arr2 + 10

    res1 = arr3.sum(axis=(0, 1)).execute().fetch()
    res2 = arr3.prod(axis=(0, 1)).execute().fetch()
    res3 = arr3.max(axis=(0, 1)).execute().fetch()
    res4 = arr3.min(axis=(0, 1)).execute().fetch()
    np.testing.assert_array_equal(raw3.sum(axis=(0, 1)), res1)
    np.testing.assert_array_equal(raw3.prod(axis=(0, 1)), res2)
    np.testing.assert_array_equal(raw3.max(axis=(0, 1)), res3)
    np.testing.assert_array_equal(raw3.min(axis=(0, 1)), res4)


def test_bool_reduction_execution(setup):
    rs = np.random.RandomState(0)
    raw = rs.randint(5, size=(8, 8, 8))
    arr = tensor(raw, chunk_size=4)

    res = (arr > 3).sum(axis=1).execute().fetch()
    np.testing.assert_array_equal(res, (raw > 3).sum(axis=1))

    res = (arr > 3).sum().execute().fetch()
    np.testing.assert_array_equal(res, (raw > 3).sum())


def test_order_execution(setup):
    rs = np.random.RandomState(0)
    raw = np.asfortranarray(rs.rand(4, 5, 6))
    arr = tensor(raw, chunk_size=3)

    res = (arr * 3 + 1).execute().fetch()
    expected = raw * 3 + 1

    np.testing.assert_array_equal(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]
