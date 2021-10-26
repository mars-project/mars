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
from ...datasource import arange, tensor, empty
from ...base import sort
from ...merge import stack
from ...reduction import all as tall
from .. import (
    average,
    bincount,
    cov,
    corrcoef,
    ptp,
    digitize,
    histogram_bin_edges,
    histogram,
    quantile,
    percentile,
    median,
)
from ..quantile import INTERPOLATION_TYPES


def test_average_execution(setup):
    data = arange(1, 5, chunk_size=1)
    t = average(data)

    res = t.execute().fetch()
    expected = np.average(np.arange(1, 5))
    assert res == expected

    t = average(arange(1, 11, chunk_size=2), weights=arange(10, 0, -1, chunk_size=2))

    res = t.execute().fetch()
    expected = np.average(range(1, 11), weights=range(10, 0, -1))
    assert res == expected

    data = arange(6, chunk_size=2).reshape((3, 2))
    t = average(data, axis=1, weights=tensor([1.0 / 4, 3.0 / 4], chunk_size=2))

    res = t.execute().fetch()
    expected = np.average(
        np.arange(6).reshape(3, 2), axis=1, weights=(1.0 / 4, 3.0 / 4)
    )
    np.testing.assert_equal(res, expected)

    with pytest.raises(TypeError):
        average(data, weights=tensor([1.0 / 4, 3.0 / 4], chunk_size=2))


def test_cov_execution(setup):
    data = np.array([[0, 2], [1, 1], [2, 0]]).T
    x = tensor(data, chunk_size=1)

    t = cov(x)

    res = t.execute().fetch()
    expected = np.cov(data)
    np.testing.assert_equal(res, expected)

    data_x = [-2.1, -1, 4.3]
    data_y = [3, 1.1, 0.12]
    x = tensor(data_x, chunk_size=1)
    y = tensor(data_y, chunk_size=1)

    X = stack((x, y), axis=0)
    t = cov(x, y)
    r = tall(t == cov(X))
    assert r.execute().fetch()


def test_corrcoef_execution(setup):
    data_x = [-2.1, -1, 4.3]
    data_y = [3, 1.1, 0.12]
    x = tensor(data_x, chunk_size=1)
    y = tensor(data_y, chunk_size=1)

    t = corrcoef(x, y)

    res = t.execute().fetch()
    expected = np.corrcoef(data_x, data_y)
    np.testing.assert_equal(res, expected)


def test_ptp_execution(setup):
    x = arange(4, chunk_size=1).reshape(2, 2)

    t = ptp(x, axis=0)

    res = t.execute().fetch()
    expected = np.ptp(np.arange(4).reshape(2, 2), axis=0)
    np.testing.assert_equal(res, expected)

    t = ptp(x, axis=1)

    res = t.execute().fetch()
    expected = np.ptp(np.arange(4).reshape(2, 2), axis=1)
    np.testing.assert_equal(res, expected)

    t = ptp(x)

    res = t.execute().fetch()
    expected = np.ptp(np.arange(4).reshape(2, 2))
    np.testing.assert_equal(res, expected)


def test_digitize_execution(setup):
    data = np.array([0.2, 6.4, 3.0, 1.6])
    x = tensor(data, chunk_size=2)
    bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    inds = digitize(x, bins)

    res = inds.execute().fetch()
    expected = np.digitize(data, bins)
    np.testing.assert_equal(res, expected)

    b = tensor(bins, chunk_size=2)
    inds = digitize(x, b)

    res = inds.execute().fetch()
    expected = np.digitize(data, bins)
    np.testing.assert_equal(res, expected)

    data = np.array([1.2, 10.0, 12.4, 15.5, 20.0])
    x = tensor(data, chunk_size=2)
    bins = np.array([0, 5, 10, 15, 20])
    inds = digitize(x, bins, right=True)

    res = inds.execute().fetch()
    expected = np.digitize(data, bins, right=True)
    np.testing.assert_equal(res, expected)

    inds = digitize(x, bins, right=False)

    res = inds.execute().fetch()
    expected = np.digitize(data, bins, right=False)
    np.testing.assert_equal(res, expected)

    data = sps.random(10, 1, density=0.1) * 12
    x = tensor(data, chunk_size=2)
    bins = np.array([1.0, 2.0, 2.5, 4.0, 10.0])
    inds = digitize(x, bins)

    res = inds.execute().fetch()
    expected = np.digitize(data.toarray(), bins, right=False)
    np.testing.assert_equal(res.toarray(), expected)


@ignore_warning
def test_histogram_bin_edges_execution(setup):
    rs = np.random.RandomState(0)

    raw = rs.randint(10, size=(20,))
    a = tensor(raw, chunk_size=6)

    # range provided
    for range_ in [(0, 10), (3, 11), (3, 7)]:
        bin_edges = histogram_bin_edges(a, range=range_)
        result = bin_edges.execute().fetch()
        expected = np.histogram_bin_edges(raw, range=range_)
        np.testing.assert_array_equal(result, expected)

    raw2 = rs.randint(10, size=(1,))
    b = tensor(raw2)
    raw3 = rs.randint(10, size=(0,))
    c = tensor(raw3)
    for t, r in [(a, raw), (b, raw2), (c, raw3), (sort(a), raw)]:
        test_bins = [
            10,
            "stone",
            "auto",
            "doane",
            "fd",
            "rice",
            "scott",
            "sqrt",
            "sturges",
        ]
        for bins in test_bins:
            bin_edges = histogram_bin_edges(t, bins=bins)
            result = bin_edges.execute().fetch()
            expected = np.histogram_bin_edges(r, bins=bins)
            np.testing.assert_array_equal(result, expected)

        test_bins = [[0, 4, 8], tensor([0, 4, 8], chunk_size=2)]
        for bins in test_bins:
            bin_edges = histogram_bin_edges(t, bins=bins)
            result = bin_edges.execute().fetch()
            expected = np.histogram_bin_edges(r, bins=[0, 4, 8])
            np.testing.assert_array_equal(result, expected)

        raw = np.arange(5)
        a = tensor(raw, chunk_size=3)
        bin_edges = histogram_bin_edges(a)
        result = bin_edges.execute().fetch()
        expected = np.histogram_bin_edges(raw)
        assert bin_edges.shape == expected.shape
        np.testing.assert_array_equal(result, expected)


@ignore_warning
def test_histogram_execution(setup):
    rs = np.random.RandomState(0)

    raw = rs.randint(10, size=(20,))
    a = tensor(raw, chunk_size=6)
    raw_weights = rs.random(20)
    weights = tensor(raw_weights, chunk_size=8)

    # range provided
    for range_ in [(0, 10), (3, 11), (3, 7)]:
        bin_edges = histogram(a, range=range_)[0]
        result = bin_edges.execute().fetch()
        expected = np.histogram(raw, range=range_)[0]
        np.testing.assert_array_equal(result, expected)

    for wt in (raw_weights, weights):
        for density in (True, False):
            bins = [1, 4, 6, 9]
            bin_edges = histogram(a, bins=bins, weights=wt, density=density)[0]
            result = bin_edges.execute().fetch()
            expected = np.histogram(
                raw, bins=bins, weights=raw_weights, density=density
            )[0]
            np.testing.assert_almost_equal(result, expected)

    raw2 = rs.randint(10, size=(1,))
    b = tensor(raw2)
    raw3 = rs.randint(10, size=(0,))
    c = tensor(raw3)
    for t, r in [(a, raw), (b, raw2), (c, raw3), (sort(a), raw)]:
        for density in (True, False):
            test_bins = [
                10,
                "stone",
                "auto",
                "doane",
                "fd",
                "rice",
                "scott",
                "sqrt",
                "sturges",
            ]
            for bins in test_bins:
                hist = histogram(t, bins=bins, density=density)[0]
                result = hist.execute().fetch()
                expected = np.histogram(r, bins=bins, density=density)[0]
                np.testing.assert_array_equal(result, expected)

            test_bins = [[0, 4, 8], tensor([0, 4, 8], chunk_size=2)]
            for bins in test_bins:
                hist = histogram(t, bins=bins, density=density)[0]
                result = hist.execute().fetch()
                expected = np.histogram(r, bins=[0, 4, 8], density=density)[0]
                np.testing.assert_array_equal(result, expected)

        # test unknown shape
        raw4 = rs.rand(10)
        d = tensor(raw4, chunk_size=6)
        d = d[d < 0.9]
        hist = histogram(d)
        result = hist.execute().fetch()[0]
        expected = np.histogram(raw4[raw4 < 0.9])[0]
        np.testing.assert_array_equal(result, expected)

        raw5 = np.arange(3, 10)
        e = arange(10, chunk_size=6)
        e = e[e >= 3]
        hist = histogram(e)
        result = hist.execute().fetch()[0]
        expected = np.histogram(raw5)[0]
        np.testing.assert_array_equal(result, expected)


def test_quantile_execution(setup):
    # test 1 chunk, 1-d
    raw = np.random.rand(20)
    a = tensor(raw, chunk_size=20)

    raw2 = raw.copy()
    raw2[np.random.RandomState(0).randint(raw.size, size=3)] = np.nan
    a2 = tensor(raw2, chunk_size=20)

    for q in [np.random.RandomState(0).rand(), np.random.RandomState(0).rand(5)]:
        for interpolation in INTERPOLATION_TYPES:
            for keepdims in [True, False]:
                r = quantile(a, q, interpolation=interpolation, keepdims=keepdims)

                result = r.execute().fetch()
                expected = np.quantile(
                    raw, q, interpolation=interpolation, keepdims=keepdims
                )

                np.testing.assert_array_equal(result, expected)

                r2 = quantile(a2, q, interpolation=interpolation, keepdims=keepdims)

                result = r2.execute().fetch()
                expected = np.quantile(
                    raw2, q, interpolation=interpolation, keepdims=keepdims
                )

                np.testing.assert_array_equal(result, expected)

    # test 1 chunk, 2-d
    raw = np.random.rand(20, 10)
    a = tensor(raw, chunk_size=20)

    raw2 = raw.copy()
    raw2.flat[np.random.RandomState(0).randint(raw.size, size=3)] = np.nan
    a2 = tensor(raw2, chunk_size=20)

    for q in [np.random.RandomState(0).rand(), np.random.RandomState(0).rand(5)]:
        for interpolation in INTERPOLATION_TYPES:
            for keepdims in [True, False]:
                for axis in [None, 0, 1]:
                    r = quantile(
                        a, q, axis=axis, interpolation=interpolation, keepdims=keepdims
                    )

                    result = r.execute().fetch()
                    expected = np.quantile(
                        raw,
                        q,
                        axis=axis,
                        interpolation=interpolation,
                        keepdims=keepdims,
                    )

                    np.testing.assert_array_equal(result, expected)

                    r2 = quantile(
                        a2, q, axis=axis, interpolation=interpolation, keepdims=keepdims
                    )

                    result = r2.execute().fetch()
                    expected = np.quantile(
                        raw2,
                        q,
                        axis=axis,
                        interpolation=interpolation,
                        keepdims=keepdims,
                    )

                    np.testing.assert_array_equal(result, expected)

    # test multi chunks, 1-d
    raw = np.random.rand(20)
    a = tensor(raw, chunk_size=6)

    raw2 = raw.copy()
    raw2[np.random.RandomState(0).randint(raw.size, size=3)] = np.nan
    a2 = tensor(raw2, chunk_size=20)

    for q in [np.random.RandomState(0).rand(), np.random.RandomState(0).rand(5)]:
        for interpolation in INTERPOLATION_TYPES:
            for keepdims in [True, False]:
                r = quantile(a, q, interpolation=interpolation, keepdims=keepdims)

                result = r.execute().fetch()
                expected = np.quantile(
                    raw, q, interpolation=interpolation, keepdims=keepdims
                )

                np.testing.assert_almost_equal(result, expected)

                r2 = quantile(a2, q, interpolation=interpolation, keepdims=keepdims)

                result = r2.execute().fetch()
                expected = np.quantile(
                    raw2, q, interpolation=interpolation, keepdims=keepdims
                )

                np.testing.assert_almost_equal(result, expected)

    # test multi chunk, 2-d
    raw = np.random.rand(20, 10)
    a = tensor(raw, chunk_size=(12, 6))

    raw2 = raw.copy()
    raw2.flat[np.random.RandomState(0).randint(raw.size, size=3)] = np.nan
    a2 = tensor(raw2, chunk_size=(12, 6))

    for q in [np.random.RandomState(0).rand(), np.random.RandomState(0).rand(5)]:
        for interpolation in INTERPOLATION_TYPES:
            for keepdims in [True, False]:
                for axis in [None, 0, 1]:
                    r = quantile(
                        a, q, axis=axis, interpolation=interpolation, keepdims=keepdims
                    )

                    result = r.execute().fetch()
                    expected = np.quantile(
                        raw,
                        q,
                        axis=axis,
                        interpolation=interpolation,
                        keepdims=keepdims,
                    )

                    np.testing.assert_almost_equal(result, expected)

                    r2 = quantile(
                        a2, q, axis=axis, interpolation=interpolation, keepdims=keepdims
                    )

                    result = r2.execute().fetch()
                    expected = np.quantile(
                        raw2,
                        q,
                        axis=axis,
                        interpolation=interpolation,
                        keepdims=keepdims,
                    )

                    np.testing.assert_almost_equal(result, expected)

    # test out, 1 chunk
    raw = np.random.rand(20)
    q = np.random.rand(11)
    a = tensor(raw, chunk_size=20)
    out = empty((5, 11))
    quantile(a, q, out=out)

    result = out.execute().fetch()
    expected = np.quantile(raw, q, out=np.empty((5, 11)))
    np.testing.assert_array_equal(result, expected)

    # test out, multi chunks
    raw = np.random.rand(20)
    q = np.random.rand(11)
    a = tensor(raw, chunk_size=6)
    out = empty((5, 11))
    quantile(a, q, out=out)

    result = out.execute().fetch()
    expected = np.quantile(raw, q, out=np.empty((5, 11)))
    np.testing.assert_almost_equal(result, expected)

    # test q which is a tensor
    q_raw = np.random.RandomState(0).rand(5)
    q = tensor(q_raw, chunk_size=6)

    r = quantile(a, q, axis=None)

    result = r.execute().fetch()
    expected = np.quantile(raw, q_raw, axis=None)

    np.testing.assert_almost_equal(result, expected)

    with pytest.raises(ValueError):
        q[0] = 1.1
        r = quantile(a, q, axis=None)
        _ = r.execute()


def test_percentile_execution(setup):
    raw = np.random.rand(20, 10)
    q = np.random.RandomState(0).randint(100, size=11)
    a = tensor(raw, chunk_size=7)
    r = percentile(a, q)

    result = r.execute().fetch()
    expected = np.percentile(raw, q)
    np.testing.assert_almost_equal(result, expected)

    mq = tensor(q)

    r = percentile(a, mq)
    result = r.execute().fetch()

    np.testing.assert_almost_equal(result, expected)


def test_median_execution(setup):
    raw = np.random.rand(20, 10)
    a = tensor(raw, chunk_size=7)
    r = median(a)

    result = r.execute().fetch()
    expected = np.median(raw)

    np.testing.assert_array_equal(result, expected)

    r = median(a, axis=1)

    result = r.execute().fetch()
    expected = np.median(raw, axis=1)

    np.testing.assert_array_equal(result, expected)


def test_bincount_execution(setup):
    rs = np.random.RandomState(0)
    raw = rs.randint(0, 9, (100,))
    raw[raw == 3] = 0
    raw_weights = rs.rand(100)

    # test non-chunked
    a = tensor(raw)
    result = bincount(a).execute().fetch()
    expected = np.bincount(raw)
    np.testing.assert_array_equal(result, expected)

    weights = tensor(raw_weights)
    result = bincount(a, weights=weights).execute().fetch()
    expected = np.bincount(raw, weights=raw_weights)
    np.testing.assert_array_equal(result, expected)

    # test chunked
    a = tensor(raw, chunk_size=13)
    result = bincount(a, chunk_size_limit=5).execute().fetch()
    expected = np.bincount(raw)
    np.testing.assert_array_equal(result, expected)

    # test minlength
    a = tensor(raw, chunk_size=13)
    result = bincount(a, chunk_size_limit=5, minlength=15).execute().fetch()
    expected = np.bincount(raw, minlength=15)
    np.testing.assert_array_equal(result, expected)

    # test with gap
    raw1 = np.concatenate([raw, [20]])
    a = tensor(raw1, chunk_size=13)
    result = bincount(a, chunk_size_limit=5).execute().fetch()
    expected = np.bincount(raw1)
    np.testing.assert_array_equal(result, expected)

    # test with weights
    a = tensor(raw, chunk_size=13)
    weights = tensor(raw_weights, chunk_size=15)
    result = bincount(a, chunk_size_limit=5, weights=weights).execute().fetch()
    expected = np.bincount(raw, weights=raw_weights)
    np.testing.assert_array_almost_equal(result, expected)

    # test errors
    a = tensor(raw, chunk_size=13)
    with pytest.raises(TypeError, match="cast array data"):
        bincount(a.astype(float)).execute()
    with pytest.raises(ValueError, match="1 dimension"):
        bincount(np.array([[1, 2], [3, 4]])).execute()
    with pytest.raises(ValueError, match="be negative"):
        bincount(a, minlength=-1).execute()
    with pytest.raises(ValueError, match="the same length"):
        bincount([-1, 1, 2, 3], weights=[3, 4]).execute()
    with pytest.raises(ValueError, match="negative elements"):
        bincount(tensor([-1, 1, 2, 3], chunk_size=2)).execute()
