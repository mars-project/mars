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

from collections import namedtuple

import numpy as np
import pytest

from .... import tensor
from ....core import tile
from ....lib.sparse.core import issparse
from ...datasource import tensor as from_ndarray


def test_rand_execution(setup):
    arr = tensor.random.rand(10, 20, chunk_size=8, dtype="f4")
    res = arr.execute().fetch()
    assert res.shape == (10, 20)
    assert res.dtype == np.float32
    np.testing.assert_array_less(arr, 1)
    np.testing.assert_array_less(0, arr)


def test_randint_execution(setup):
    arr = tensor.random.randint(0, 2, size=(10, 30), chunk_size=8)

    res = arr.execute().fetch()
    assert res.shape == (10, 30)
    np.testing.assert_array_less(arr, 2)
    np.testing.assert_array_less(-1, res)


def test_random_integers_execution(setup):
    rs = tensor.random.RandomState(0)
    arr1 = rs.random_integers(0, 10, size=(10, 20), chunk_size=8)
    rs = tensor.random.RandomState(0)
    arr2 = rs.random_integers(0, 10, size=(10, 20), chunk_size=8)

    res1 = arr1.execute().fetch()
    res2 = arr2.execute().fetch()

    np.testing.assert_array_almost_equal(res1, res2)


def test_choice_execution(setup):
    # test 1 chunk, get integer
    a = tensor.random.RandomState(0).choice(10)
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 10)
    np.testing.assert_array_less(0, res)

    a = tensor.random.RandomState(0).choice(10)
    seed = tile(a).chunks[0].op.seed
    expected = np.random.RandomState(seed).choice(10)
    np.testing.assert_array_equal(res, expected)

    # test 1 chunk, integer
    a = tensor.random.RandomState(0).choice(10, (4, 3))
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 10)
    np.testing.assert_array_less(-1, res)
    b = tensor.random.RandomState(0).choice(10, (4, 3))
    res2 = b.execute().fetch()
    np.testing.assert_array_equal(res, res2)

    # test 1 chunk, ndarray
    raw = np.random.RandomState(0).rand(10)
    a = tensor.random.RandomState(0).choice(raw, (4, 3))
    res = a.execute().fetch()
    seed = tile(a).chunks[0].op.seed
    expected = np.random.RandomState(seed).choice(raw, (4, 3))
    np.testing.assert_array_equal(res, expected)

    # test with replacement, integer
    a = tensor.random.RandomState(0).choice(20, (7, 4), chunk_size=4)
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 20)
    np.testing.assert_array_less(-1, res)
    b = tensor.random.RandomState(0).choice(20, (7, 4), chunk_size=4)
    res2 = b.execute().fetch()
    np.testing.assert_array_equal(res, res2)

    # test with replacement, ndarray
    raw = np.random.RandomState(0).rand(20)
    t = tensor.array(raw, chunk_size=8)
    a = tensor.random.RandomState(0).choice(t, (7, 4), chunk_size=4)
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 20)
    np.testing.assert_array_less(-1, res)
    b = tensor.random.RandomState(0).choice(t, (7, 4), chunk_size=4)
    res2 = b.execute().fetch()
    np.testing.assert_array_equal(res, res2)

    # test without replacement, integer
    a = tensor.random.RandomState(0).choice(100, (7, 2), chunk_size=2, replace=False)
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 100)
    np.testing.assert_array_less(-1, res)
    assert len(np.unique(res))
    b = tensor.random.RandomState(0).choice(100, (7, 2), chunk_size=2, replace=False)
    res2 = b.execute().fetch()
    np.testing.assert_array_equal(res, res2)

    # test without replacement, ndarray
    raw = np.random.RandomState(0).rand(100)
    t = tensor.array(raw, chunk_size=47)
    a = tensor.random.RandomState(0).choice(t, (7, 2), chunk_size=2, replace=False)
    res = a.execute().fetch()
    np.testing.assert_array_less(res, 100)
    np.testing.assert_array_less(-1, res)
    assert len(np.unique(res))
    b = tensor.random.RandomState(0).choice(t, (7, 2), chunk_size=2, replace=False)
    res2 = b.execute().fetch()
    np.testing.assert_array_equal(res, res2)

    # test p
    raw = np.random.RandomState(0).rand(5)
    p = [0.3, 0.2, 0.1, 0.3, 0.1]
    a = tensor.random.RandomState(0).choice(raw, 3, p=p)
    res = a.execute().fetch()
    expected = np.random.RandomState(tile(a).chunks[0].op.seed).choice(raw, 3, p=p)
    np.testing.assert_array_equal(res, expected)


def test_sparse_randint_execution(setup):
    # size_executor = ExecutorForTest(sync_provider_type=ExecutorForTest.SyncProviderType.MOCK)

    arr = tensor.random.randint(
        1, 2, size=(30, 50), density=0.1, chunk_size=20, dtype="f4"
    )
    # size_res = size_executor.execute_tensor(arr, mock=True)
    # assert pytest.approx(arr.nbytes * 0.1) == sum(tp[0] for tp in size_res)

    res = arr.execute().fetch()
    assert issparse(res) is True
    assert res.shape == (30, 50)
    np.testing.assert_array_less(res.data, 2)
    np.testing.assert_array_less(0, res.data)
    assert (res >= 1).toarray().sum() == pytest.approx(30 * 50 * 0.1, abs=20)


random_test_options = namedtuple("random_test_options", ["func_name", "args", "kwargs"])

random_params = [
    random_test_options("beta", ([1, 2], [3, 4]), dict(chunk_size=2)),
    random_test_options("binomial", (10, 0.5, 100), dict(chunk_size=50)),
    random_test_options("chisquare", (2, 100), dict(chunk_size=50)),
    random_test_options("dirichlet", ((10, 5, 3), 100), dict(chunk_size=50)),
    random_test_options("exponential", (1.0, 100), dict(chunk_size=50)),
    random_test_options("f", (1.0, 2.0, 100), dict(chunk_size=50)),
    random_test_options("gamma", (1.0, 2.0, 100), dict(chunk_size=50)),
    random_test_options("geometric", (1.0, 100), dict(chunk_size=50)),
    random_test_options("gumbel", (0.5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("hypergeometric", (10, 20, 15, 100), dict(chunk_size=50)),
    random_test_options("laplace", (0.5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("logistic", (0.5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("lognormal", (0.5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("logseries", (0.5, 100), dict(chunk_size=50)),
    random_test_options("multinomial", (10, [0.2, 0.5, 0.3], 100), dict(chunk_size=50)),
    random_test_options(
        "multivariate_normal", ([1, 2], [[1, 0], [0, 1]], 100), dict(chunk_size=50)
    ),
    random_test_options("negative_binomial", (5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("noncentral_chisquare", (0.5, 1.0, 100), dict(chunk_size=50)),
    random_test_options("noncentral_f", (1.5, 1.0, 1.1, 100), dict(chunk_size=50)),
    random_test_options("pareto", (1.0, 100), dict(chunk_size=50)),
    random_test_options("poisson", (1.0, 100), dict(chunk_size=50)),
    random_test_options("power", (1.0, 100), dict(chunk_size=50)),
    random_test_options("rayleigh", (1.0, 100), dict(chunk_size=50)),
    random_test_options("standard_cauchy", (100,), dict(chunk_size=50)),
    random_test_options("standard_exponential", (100,), dict(chunk_size=50)),
    random_test_options("standard_gamma", (1.0, 100), dict(chunk_size=50)),
    random_test_options("standard_normal", (100,), dict(chunk_size=50)),
    random_test_options("standard_t", (1.0, 100), dict(chunk_size=50)),
    random_test_options("triangular", (0.1, 0.2, 0.3, 100), dict(chunk_size=50)),
    random_test_options("uniform", (0.1, 0.2, 100), dict(chunk_size=50)),
    random_test_options("vonmises", (0.1, 0.2, 100), dict(chunk_size=50)),
    random_test_options("wald", (0.1, 0.2, 100), dict(chunk_size=50)),
    random_test_options("weibull", (0.1, 100), dict(chunk_size=50)),
    random_test_options("zipf", (1.1, 100), dict(chunk_size=50)),
]


@pytest.mark.parametrize("test_opts", random_params)
def test_random_execute(setup, test_opts):
    rs = tensor.random.RandomState(0)
    arr1 = getattr(rs, test_opts.func_name)(*test_opts.args, **test_opts.kwargs)
    rs = tensor.random.RandomState(0)
    arr2 = getattr(rs, test_opts.func_name)(*test_opts.args, **test_opts.kwargs)
    assert np.array_equal(arr1.execute().fetch(), arr2.execute().fetch())


def test_permutation_execute(setup):
    rs = tensor.random.RandomState(0)
    x = rs.permutation(10)
    res = x.execute().fetch()
    assert not np.all(res[:-1] < res[1:])
    np.testing.assert_array_equal(np.sort(res), np.arange(10))

    arr = from_ndarray([1, 4, 9, 12, 15], chunk_size=2)
    x = rs.permutation(arr)
    res = x.execute().fetch()
    assert not np.all(res[:-1] < res[1:])
    np.testing.assert_array_equal(np.sort(res), np.asarray([1, 4, 9, 12, 15]))

    arr = from_ndarray(np.arange(48).reshape(12, 4), chunk_size=2)
    # axis = 0
    x = rs.permutation(arr)
    res = x.execute().fetch()
    assert not np.all(res[:-1] < res[1:])
    np.testing.assert_array_equal(np.sort(res, axis=0), np.arange(48).reshape(12, 4))
    # axis != 0
    x2 = rs.permutation(arr, axis=1)
    res = x2.execute().fetch()
    assert not np.all(res[:, :-1] < res[:, 1:])
    np.testing.assert_array_equal(np.sort(res, axis=1), np.arange(48).reshape(12, 4))
