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

from ....utils import ignore_warning
from ...datasource import ones, tensor
from .. import (
    mean,
    nansum,
    nanmax,
    nanmin,
    nanmean,
    nanprod,
    nanargmax,
    nanargmin,
    nanvar,
    nanstd,
    count_nonzero,
    allclose,
    array_equal,
    var,
    std,
    nancumsum,
    nancumprod,
)


def test_sum_prod_execution(setup):
    arr = ones((10, 8), chunk_size=6)
    assert 80 == arr.sum().execute().fetch()
    np.testing.assert_array_equal(
        arr.sum(axis=0).execute().fetch(), np.full((8,), fill_value=10)
    )

    arr = ones((3, 3), chunk_size=2)
    assert 512 == (arr * 2).prod().execute().fetch()
    np.testing.assert_array_equal(
        (arr * 2).prod(axis=0).execute().fetch(), np.full((3,), fill_value=8)
    )

    raw = sps.random(10, 20, density=0.1)
    arr = tensor(raw, chunk_size=3)
    res = arr.sum().execute().fetch()

    assert pytest.approx(res) == raw.sum()

    # test order
    raw = np.asfortranarray(np.random.rand(10, 20, 30))
    arr = tensor(raw, chunk_size=13)
    arr2 = arr.sum(axis=-1)

    res = arr2.execute().fetch()
    expected = raw.sum(axis=-1)
    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    # test string dtype
    a = tensor(list("abcdefghi"), dtype=object)
    assert a.sum().execute().fetch() == "abcdefghi"
    a = tensor(list("abcdefghi"), dtype=object, chunk_size=2)
    assert a.sum().execute().fetch() == "abcdefghi"


def test_max_min_execution(setup):
    raw = np.random.randint(10000, size=(10, 10, 10))

    arr = tensor(raw, chunk_size=3)

    assert raw.max() == arr.max().execute().fetch()
    assert raw.min() == arr.min().execute().fetch()

    np.testing.assert_array_equal(raw.max(axis=0), arr.max(axis=0).execute().fetch())
    assert arr.max(axis=0).issparse() is False
    np.testing.assert_array_equal(raw.min(axis=0), arr.min(axis=0).execute().fetch())
    assert arr.min(axis=0).issparse() is False

    np.testing.assert_array_equal(
        raw.max(axis=(1, 2)), arr.max(axis=(1, 2)).execute().fetch()
    )
    np.testing.assert_array_equal(
        raw.min(axis=(1, 2)), arr.min(axis=(1, 2)).execute().fetch()
    )

    raw = sps.random(10, 10, density=0.5)

    arr = tensor(raw, chunk_size=3)

    assert raw.max() == arr.max().execute().fetch()
    assert raw.min() == arr.min().execute().fetch()

    np.testing.assert_almost_equal(
        raw.max(axis=1).A.ravel(), arr.max(axis=1).execute().fetch().toarray()
    )
    assert arr.max(axis=1).issparse() is True
    np.testing.assert_almost_equal(
        raw.min(axis=1).A.ravel(), arr.min(axis=1).execute().fetch().toarray()
    )
    assert arr.min(axis=1).issparse() is True

    # test string dtype
    a = tensor(list("abcdefghi"), dtype=object)
    assert a.max().execute().fetch() == "i"
    a = tensor(list("abcdefghi"), dtype=object, chunk_size=2)
    assert a.max().execute().fetch() == "i"

    # test empty chunks
    raw = np.arange(3, 10)
    arr = tensor(np.arange(0, 10), chunk_size=3)
    arr = arr[arr >= 3]
    assert raw.max() == arr.max().execute().fetch()
    assert raw.min() == arr.min().execute().fetch()


def test_all_any_execution(setup):
    raw1 = np.zeros((10, 15))
    raw2 = np.ones((10, 15))
    raw3 = np.array(
        [
            [True, False, True, False],
            [True, True, True, True],
            [False, False, False, False],
            [False, True, False, True],
        ]
    )

    arr1 = tensor(raw1, chunk_size=3)
    arr2 = tensor(raw2, chunk_size=3)
    arr3 = tensor(raw3, chunk_size=4)

    assert not arr1.all().execute().fetch()
    assert arr2.all().execute().fetch()
    assert not arr1.any().execute().fetch()
    np.testing.assert_array_equal(raw3.all(axis=1), arr3.all(axis=1).execute().fetch())
    np.testing.assert_array_equal(raw3.any(axis=0), arr3.any(axis=0).execute().fetch())

    raw = sps.random(10, 10, density=0.5) > 0.5

    arr = tensor(raw, chunk_size=3)

    assert raw.A.all() == arr.all().execute().fetch()
    assert raw.A.any() == arr.any().execute().fetch()

    # test string dtype
    a = tensor(list("abcdefghi"), dtype=object)
    assert a.all().execute().fetch() == "i"
    a = tensor(list("abcdefghi"), dtype=object, chunk_size=2)
    assert a.any().execute().fetch() == "a"


def test_mean_execution(setup):
    raw1 = np.random.random((20, 25))
    raw2 = np.random.randint(10, size=(20, 25))

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.mean().execute().fetch()
    expected1 = raw1.mean()
    np.testing.assert_allclose(res1, expected1)

    res2 = arr1.mean(axis=0).execute().fetch()
    expected2 = raw1.mean(axis=0)
    assert np.allclose(res2, expected2) is True

    res3 = arr1.mean(axis=1, keepdims=True).execute().fetch()
    expected3 = raw1.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    arr2 = tensor(raw2, chunk_size=6)

    res1 = arr2.mean().execute().fetch()
    expected1 = raw2.mean()
    assert res1 == expected1

    res2 = arr2.mean(axis=0).execute().fetch()
    expected2 = raw2.mean(axis=0)
    np.testing.assert_allclose(res2, expected2)

    res3 = arr2.mean(axis=1, keepdims=True).execute().fetch()
    expected3 = raw2.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    raw1 = sps.random(20, 25, density=0.1)

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.mean().execute().fetch()
    expected1 = raw1.mean()
    np.testing.assert_allclose(res1, expected1)

    arr2 = tensor(raw1, chunk_size=30)

    res1 = arr2.mean().execute().fetch()
    expected1 = raw1.mean()
    np.testing.assert_allclose(res1, expected1)

    arr = mean(1)
    assert arr.execute().fetch() == 1

    with pytest.raises(TypeError):
        tensor(list("abcdefghi"), dtype=object).mean().execute()


def test_var_execution(setup):
    raw1 = np.random.random((20, 25))
    raw2 = np.random.randint(10, size=(20, 25))

    arr0 = tensor(raw1, chunk_size=25)

    res1 = arr0.var().execute().fetch()
    expected1 = raw1.var()
    np.testing.assert_allclose(res1, expected1)

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.var().execute().fetch()
    expected1 = raw1.var()
    np.testing.assert_allclose(res1, expected1)

    res2 = arr1.var(axis=0).execute().fetch()
    expected2 = raw1.var(axis=0)
    np.testing.assert_allclose(res2, expected2)

    res3 = arr1.var(axis=1, keepdims=True).execute().fetch()
    expected3 = raw1.var(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    arr2 = tensor(raw2, chunk_size=6)

    res1 = arr2.var().execute().fetch()
    expected1 = raw2.var()
    assert pytest.approx(res1) == expected1

    res2 = arr2.var(axis=0).execute().fetch()
    expected2 = raw2.var(axis=0)
    np.testing.assert_allclose(res2, expected2)

    res3 = arr2.var(axis=1, keepdims=True).execute().fetch()
    expected3 = raw2.var(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    res4 = arr2.var(ddof=1).execute().fetch()
    expected4 = raw2.var(ddof=1)
    assert pytest.approx(res4) == expected4

    raw1 = sps.random(20, 25, density=0.1)

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.var().execute().fetch()
    expected1 = raw1.toarray().var()
    np.testing.assert_allclose(res1, expected1)

    arr2 = tensor(raw1, chunk_size=30)

    res1 = arr2.var().execute().fetch()
    expected1 = raw1.toarray().var()
    np.testing.assert_allclose(res1, expected1)

    arr = var(1)
    assert arr.execute().fetch() == 0


def test_std_execution(setup):
    raw1 = np.random.random((20, 25))
    raw2 = np.random.randint(10, size=(20, 25))

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.std().execute().fetch()
    expected1 = raw1.std()
    np.testing.assert_allclose(res1, expected1)

    res2 = arr1.std(axis=0).execute().fetch()
    expected2 = raw1.std(axis=0)
    np.testing.assert_allclose(res2, expected2)

    res3 = arr1.std(axis=1, keepdims=True).execute().fetch()
    expected3 = raw1.std(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    arr2 = tensor(raw2, chunk_size=6)

    res1 = arr2.std().execute().fetch()
    expected1 = raw2.std()
    assert pytest.approx(res1) == expected1

    res2 = arr2.std(axis=0).execute().fetch()
    expected2 = raw2.std(axis=0)
    np.testing.assert_allclose(res2, expected2)

    res3 = arr2.std(axis=1, keepdims=True).execute().fetch()
    expected3 = raw2.std(axis=1, keepdims=True)
    np.testing.assert_allclose(res3, expected3)

    res4 = arr2.std(ddof=1).execute().fetch()
    expected4 = raw2.std(ddof=1)
    assert pytest.approx(res4) == expected4

    raw1 = sps.random(20, 25, density=0.1)

    arr1 = tensor(raw1, chunk_size=6)

    res1 = arr1.std().execute().fetch()
    expected1 = raw1.toarray().std()
    np.testing.assert_allclose(res1, expected1)

    arr2 = tensor(raw1, chunk_size=30)

    res1 = arr2.std().execute().fetch()
    expected1 = raw1.toarray().std()
    np.testing.assert_allclose(res1, expected1)

    arr = std(1)
    assert arr.execute().fetch() == 0


def test_arg_reduction(setup):
    raw = np.random.random((20, 20, 20))

    arr = tensor(raw, chunk_size=6)

    assert raw.argmax() == arr.argmax().execute().fetch()
    assert raw.argmin() == arr.argmin().execute().fetch()

    np.testing.assert_array_equal(
        raw.argmax(axis=0), arr.argmax(axis=0).execute().fetch()
    )
    np.testing.assert_array_equal(
        raw.argmin(axis=0), arr.argmin(axis=0).execute().fetch()
    )

    raw_format = sps.random(20, 20, density=0.1, format="lil")

    random_min = np.random.randint(0, 200)
    random_max = np.random.randint(200, 400)
    raw_format[np.unravel_index(random_min, raw_format.shape)] = -1
    raw_format[np.unravel_index(random_max, raw_format.shape)] = 2

    raw = raw_format.tocoo()
    arr = tensor(raw, chunk_size=6)

    assert raw.argmax() == arr.argmax().execute().fetch()
    assert raw.argmin() == arr.argmin().execute().fetch()

    # test order
    raw = np.asfortranarray(np.random.rand(10, 20, 30))
    arr = tensor(raw, chunk_size=13)
    arr2 = arr.argmax(axis=-1)

    res = arr2.execute().fetch()
    expected = raw.argmax(axis=-1)
    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    with pytest.raises(TypeError):
        tensor(list("abcdefghi"), dtype=object).argmax().execute()


@ignore_warning
def test_nan_reduction(setup):
    raw = np.random.choice(a=[0, 1, np.nan], size=(10, 10), p=[0.3, 0.4, 0.3])

    arr = tensor(raw, chunk_size=6)

    assert np.nansum(raw) == nansum(arr).execute().fetch()
    assert np.nanprod(raw) == nanprod(arr).execute().fetch()
    assert np.nanmax(raw) == nanmax(arr).execute().fetch()
    assert np.nanmin(raw) == nanmin(arr).execute().fetch()
    assert np.nanmean(raw) == nanmean(arr).execute().fetch()
    assert pytest.approx(np.nanvar(raw)) == nanvar(arr).execute().fetch()
    assert (
        pytest.approx(np.nanvar(raw, ddof=1)) == nanvar(arr, ddof=1).execute().fetch()
    )
    assert pytest.approx(np.nanstd(raw)) == nanstd(arr).execute().fetch()
    assert (
        pytest.approx(np.nanstd(raw, ddof=1)) == nanstd(arr, ddof=1).execute().fetch()
    )

    arr = tensor(raw, chunk_size=10)

    assert np.nansum(raw) == nansum(arr).execute().fetch()
    assert np.nanprod(raw) == nanprod(arr).execute().fetch()
    assert np.nanmax(raw) == nanmax(arr).execute().fetch()
    assert np.nanmin(raw) == nanmin(arr).execute().fetch()
    assert np.nanmean(raw) == nanmean(arr).execute().fetch()
    assert pytest.approx(np.nanvar(raw)) == nanvar(arr).execute().fetch()
    assert (
        pytest.approx(np.nanvar(raw, ddof=1)) == nanvar(arr, ddof=1).execute().fetch()
    )
    assert pytest.approx(np.nanstd(raw)) == nanstd(arr).execute().fetch()
    assert (
        pytest.approx(np.nanstd(raw, ddof=1)) == nanstd(arr, ddof=1).execute().fetch()
    )

    raw = np.random.random((10, 10))
    raw[:3, :3] = np.nan
    arr = tensor(raw, chunk_size=6)
    assert np.nanargmin(raw) == nanargmin(arr).execute().fetch()
    assert np.nanargmax(raw) == nanargmax(arr).execute().fetch()

    raw = np.full((10, 10), np.nan)
    arr = tensor(raw, chunk_size=6)

    assert 0 == nansum(arr).execute().fetch()
    assert 1 == nanprod(arr).execute().fetch()
    assert np.isnan(nanmax(arr).execute().fetch())
    assert np.isnan(nanmin(arr).execute().fetch())
    assert np.isnan(nanmean(arr).execute().fetch())
    with pytest.raises(ValueError):
        _ = nanargmin(arr).execute()  # noqa: F841
    with pytest.raises(ValueError):
        _ = nanargmax(arr).execute()  # noqa: F841

    raw = sps.random(10, 10, density=0.1, format="csr")
    raw[:3, :3] = np.nan
    arr = tensor(raw, chunk_size=6)

    assert pytest.approx(np.nansum(raw.A)) == nansum(arr).execute().fetch()
    assert pytest.approx(np.nanprod(raw.A)) == nanprod(arr).execute().fetch()
    assert pytest.approx(np.nanmax(raw.A)) == nanmax(arr).execute().fetch()
    assert pytest.approx(np.nanmin(raw.A)) == nanmin(arr).execute().fetch()
    assert pytest.approx(np.nanmean(raw.A)) == nanmean(arr).execute().fetch()
    assert pytest.approx(np.nanvar(raw.A)) == nanvar(arr).execute().fetch()
    assert (
        pytest.approx(np.nanvar(raw.A, ddof=1)) == nanvar(arr, ddof=1).execute().fetch()
    )
    assert pytest.approx(np.nanstd(raw.A)) == nanstd(arr).execute().fetch()
    assert (
        pytest.approx(np.nanstd(raw.A, ddof=1)) == nanstd(arr, ddof=1).execute().fetch()
    )

    arr = nansum(1)
    assert arr.execute().fetch() == 1


def test_cum_reduction(setup):
    raw = np.random.randint(5, size=(8, 8, 8))

    arr = tensor(raw, chunk_size=6)

    res1 = arr.cumsum(axis=1).execute().fetch()
    res2 = arr.cumprod(axis=1).execute().fetch()
    expected1 = raw.cumsum(axis=1)
    expected2 = raw.cumprod(axis=1)
    np.testing.assert_array_equal(res1, expected1)
    np.testing.assert_array_equal(res2, expected2)

    raw = sps.random(8, 8, density=0.1)

    arr = tensor(raw, chunk_size=6)

    res1 = arr.cumsum(axis=1).execute().fetch()
    res2 = arr.cumprod(axis=1).execute().fetch()
    expected1 = raw.A.cumsum(axis=1)
    expected2 = raw.A.cumprod(axis=1)
    assert np.allclose(res1, expected1)
    assert np.allclose(res2, expected2)

    # test order
    raw = np.asfortranarray(np.random.rand(10, 20, 30))
    arr = tensor(raw, chunk_size=13)
    arr2 = arr.cumsum(axis=-1)

    res = arr2.execute().fetch()
    expected = raw.cumsum(axis=-1)
    np.testing.assert_allclose(res, expected)
    assert res.flags["C_CONTIGUOUS"] == expected.flags["C_CONTIGUOUS"]
    assert res.flags["F_CONTIGUOUS"] == expected.flags["F_CONTIGUOUS"]

    # test string dtype
    a = tensor(list("abcdefghi"), dtype=object)
    np.testing.assert_array_equal(
        a.cumsum().execute().fetch(),
        np.cumsum(np.array(list("abcdefghi"), dtype=object)),
    )
    a = tensor(list("abcdefghi"), dtype=object, chunk_size=2)
    np.testing.assert_array_equal(
        a.cumsum().execute().fetch(),
        np.cumsum(np.array(list("abcdefghi"), dtype=object)),
    )

    # test empty chunks
    raw = np.random.rand(100)
    arr = tensor(raw, chunk_size=((0, 100),))
    res = arr.cumsum().execute().fetch()
    expected = raw.cumsum()
    np.testing.assert_allclose(res, expected)
    res = arr.cumprod().execute().fetch()
    expected = raw.cumprod()
    np.testing.assert_allclose(res, expected)


def test_nan_cum_reduction(setup):
    raw = np.random.randint(5, size=(8, 8, 8)).astype(float)
    raw[:2, 2:4, 4:6] = np.nan

    arr = tensor(raw, chunk_size=6)

    res1 = nancumsum(arr, axis=1).execute().fetch()
    res2 = nancumprod(arr, axis=1).execute().fetch()
    expected1 = np.nancumsum(raw, axis=1)
    expected2 = np.nancumprod(raw, axis=1)
    np.testing.assert_array_equal(res1, expected1)
    np.testing.assert_array_equal(res2, expected2)

    raw = sps.random(8, 8, density=0.1, format="lil")
    raw[:2, 2:4] = np.nan

    arr = tensor(raw, chunk_size=6)

    res1 = nancumsum(arr, axis=1).execute().fetch()
    res2 = nancumprod(arr, axis=1).execute().fetch()
    expected1 = np.nancumsum(raw.A, axis=1)
    expected2 = np.nancumprod(raw.A, axis=1)
    assert np.allclose(res1, expected1) is True
    assert np.allclose(res2, expected2) is True


def test_out_reduction_execution(setup):
    raw = np.random.randint(5, size=(8, 8, 8))

    arr = tensor(raw, chunk_size=6)
    arr2 = ones((8, 8), dtype="i8", chunk_size=6)
    arr.sum(axis=1, out=arr2)

    res = arr2.execute().fetch()
    expected = raw.sum(axis=1)

    np.testing.assert_array_equal(res, expected)


def test_out_cum_reduction_execution(setup):
    raw = np.random.randint(5, size=(8, 8, 8))

    arr = tensor(raw, chunk_size=6)
    arr.cumsum(axis=0, out=arr)

    res = arr.execute().fetch()
    expected = raw.cumsum(axis=0)

    np.testing.assert_array_equal(res, expected)


def test_count_nonzero_execution(setup):
    raw = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]

    arr = tensor(raw, chunk_size=5)
    t = count_nonzero(arr)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw)
    np.testing.assert_equal(res, expected)

    arr = tensor(raw, chunk_size=2)
    t = count_nonzero(arr)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw)
    np.testing.assert_equal(res, expected)

    t = count_nonzero(arr, axis=0)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw, axis=0)
    np.testing.assert_equal(res, expected)

    t = count_nonzero(arr, axis=1)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw, axis=1)
    np.testing.assert_equal(res, expected)

    raw = sps.csr_matrix(raw)

    arr = tensor(raw, chunk_size=2)
    t = count_nonzero(arr)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw.A)
    np.testing.assert_equal(res, expected)

    t = count_nonzero(arr, axis=0)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw.A, axis=0)
    np.testing.assert_equal(res, expected)

    t = count_nonzero(arr, axis=1)

    res = t.execute().fetch()
    expected = np.count_nonzero(raw.A, axis=1)
    np.testing.assert_equal(res, expected)

    # test string dtype
    a = tensor(list("abcdefghi"), dtype=object)
    assert count_nonzero(a).execute().fetch() == 9
    a = tensor(list("abcdefghi"), dtype=object, chunk_size=2)
    assert count_nonzero(a).execute().fetch() == 9


def test_allclose_execution(setup):
    a = tensor([1e10, 1e-7], chunk_size=1)
    b = tensor([1.00001e10, 1e-8], chunk_size=1)

    t = allclose(a, b)

    res = t.execute().fetch()
    assert res is False

    a = tensor([1e10, 1e-8], chunk_size=1)
    b = tensor([1.00001e10, 1e-9], chunk_size=1)

    t = allclose(a, b)

    res = t.execute().fetch()
    assert res is True

    a = tensor([1.0, np.nan], chunk_size=1)
    b = tensor([1.0, np.nan], chunk_size=1)

    t = allclose(a, b, equal_nan=True)

    res = t.execute().fetch()
    assert res is True

    a = tensor(sps.csr_matrix([[1e10, 1e-7], [0, 0]]), chunk_size=1)
    b = tensor(sps.csr_matrix([[1.00001e10, 1e-8], [0, 0]]), chunk_size=1)

    t = allclose(a, b)

    res = t.execute().fetch()
    assert res is False

    # test string dtype
    with pytest.raises(TypeError):
        a = tensor(list("abcdefghi"), dtype=object)
        allclose(a, a).execute()


def test_array_equal(setup):
    a = ones((10, 5), chunk_size=4)
    b = ones((10, 5), chunk_size=5)

    c = array_equal(a, b)

    assert c.execute().fetch()
