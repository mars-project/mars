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

import numpy as np
import pytest
import scipy
from scipy.stats import (
    entropy as sp_entropy,
    power_divergence as sp_power_divergence,
    chisquare as sp_chisquare,
    ttest_rel as sp_ttest_rel,
    ttest_ind as sp_ttest_ind,
    ttest_ind_from_stats as sp_ttest_ind_from_stats,
    ttest_1samp as sp_ttest_1samp,
)

from mars.lib.version import parse as parse_version
from mars.tensor import tensor
from mars.tensor.stats import (
    entropy, power_divergence, chisquare,
    ttest_ind, ttest_rel, ttest_1samp, ttest_ind_from_stats,
)


def test_entropy_execution(setup):
    rs = np.random.RandomState(0)
    a = rs.rand(10)

    t1 = tensor(a, chunk_size=4)
    r = entropy(t1)

    result = r.execute().fetch()
    expected = sp_entropy(a)
    np.testing.assert_array_almost_equal(result, expected)

    b = rs.rand(10)
    base = 3.1

    t2 = tensor(b, chunk_size=4)
    r = entropy(t1, t2, base)

    result = r.execute().fetch()
    expected = sp_entropy(a, b, base)
    np.testing.assert_array_almost_equal(result, expected)

    b = rs.rand(10)
    base = 3.1

    t2 = tensor(b, chunk_size=4)
    r = entropy(t1, t2, base)

    result = r.execute().fetch()
    expected = sp_entropy(a, b, base)
    np.testing.assert_array_almost_equal(result, expected)

    r = entropy(t1, t2, t1.sum())

    result = r.execute().fetch()
    expected = sp_entropy(a, b, a.sum())
    np.testing.assert_array_almost_equal(result, expected)

    with pytest.raises(ValueError):
        entropy(t1, t2[:7])


def test_power_divergence_execution(setup):
    f_obs_raw = np.array([16, 18, 16, 14, 12, 12])
    f_exp_raw = np.array([16, 16, 16, 16, 16, 8])

    f_obs = tensor(f_obs_raw, chunk_size=4)
    f_exp = tensor(f_exp_raw, chunk_size=4)

    with pytest.raises(ValueError):
        power_divergence(f_obs, f_exp, lambda_='non-exist-lambda')

    r = power_divergence(f_obs, lambda_='pearson')
    result = r.execute().fetch()

    expected = sp_power_divergence(f_obs_raw, lambda_='pearson')
    np.testing.assert_almost_equal(expected[0], result[0])
    np.testing.assert_almost_equal(expected[1], result[1])

    modes = [
        None,
        'pearson',
        'log-likelihood',
        'mod-log-likelihood',
        'neyman',
    ]

    for mode in modes:
        r = power_divergence(f_obs, f_exp, lambda_=mode)
        result = r.execute().fetch()

        expected = sp_power_divergence(
            f_obs_raw, f_exp_raw, lambda_=mode)
        np.testing.assert_almost_equal(expected[0], result[0])
        np.testing.assert_almost_equal(expected[1], result[1])


def test_chisquare_execution(setup):
    f_obs_raw = np.array([16, 18, 16, 14, 12, 12])
    f_exp_raw = np.array([16, 16, 16, 16, 16, 8])

    f_obs = tensor(f_obs_raw, chunk_size=4)
    f_exp = tensor(f_exp_raw, chunk_size=4)

    r = chisquare(f_obs, f_exp)
    result = r.execute().fetch()

    expected = sp_chisquare(f_obs_raw, f_exp_raw)
    np.testing.assert_almost_equal(expected[0], result[0])
    np.testing.assert_almost_equal(expected[1], result[1])


def test_t_test_execution(setup):
    if parse_version(scipy.__version__) >= parse_version('1.6.0'):
        alternatives = ['less', 'greater', 'two-sided']

        mt_from_stats = lambda a, b, alternative=None, equal_var=True: ttest_ind_from_stats(
            a.mean(), a.std(), a.shape[0], b.mean(), b.std(), b.shape[0],
            alternative=alternative, equal_var=equal_var)
        sp_from_stats = lambda a, b, alternative=None, equal_var=True: sp_ttest_ind_from_stats(
            a.mean(), a.std(), a.shape[0], b.mean(), b.std(), b.shape[0],
            alternative=alternative, equal_var=equal_var)
    else:
        alternatives = ['two-sided']

        mt_from_stats = lambda a, b, equal_var=True: ttest_ind_from_stats(
            a.mean(), a.std(), a.shape[0], b.mean(), b.std(), b.shape[0],
            equal_var=equal_var)
        sp_from_stats = lambda a, b, equal_var=True: sp_ttest_ind_from_stats(
            a.mean(), a.std(), a.shape[0], b.mean(), b.std(), b.shape[0],
            equal_var=equal_var)

    funcs = [
        (ttest_rel, sp_ttest_rel),
        (
            functools.partial(ttest_ind, equal_var=True),
            functools.partial(sp_ttest_ind, equal_var=True),
        ),
        (
            functools.partial(ttest_ind, equal_var=False),
            functools.partial(sp_ttest_ind, equal_var=False),
        ),
        (
            functools.partial(mt_from_stats, equal_var=True),
            functools.partial(sp_from_stats, equal_var=True),
        ),
        (
            functools.partial(mt_from_stats, equal_var=False),
            functools.partial(sp_from_stats, equal_var=False),
        ),
        (ttest_1samp, sp_ttest_1samp),
    ]

    fa_raw = np.array([16, 18, 16, 14, 12, 12])
    fb_raw = np.array([16, 16, 16, 16, 16, 8])

    fa = tensor(fa_raw, chunk_size=4)
    fb = tensor(fb_raw, chunk_size=4)

    for mt_func, sp_func in funcs:
        if parse_version(scipy.__version__) >= parse_version('1.6.0'):
            with pytest.raises(ValueError):
                mt_func(fa, fb, alternative='illegal-alternative')

        for alt in alternatives:
            if parse_version(scipy.__version__) >= parse_version('1.6.0'):
                r = mt_func(fa, fb, alternative=alt)
            else:
                r = mt_func(fa, fb)
            result = r.execute().fetch()

            if parse_version(scipy.__version__) >= parse_version('1.6.0'):
                expected = sp_func(fa_raw, fb_raw, alternative=alt)
            else:
                expected = sp_func(fa_raw, fb_raw)
            np.testing.assert_almost_equal(expected[0], result[0])
            np.testing.assert_almost_equal(expected[1], result[1])
