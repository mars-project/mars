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
from scipy import __version__ as sp_version
from scipy.stats import (
    ttest_ind as sp_ttest_ind,
    ttest_ind_from_stats as sp_ttest_ind_from_stats,
    ttest_rel as sp_ttest_rel,
    ttest_1samp as sp_ttest_1samp,
)
from scipy.stats import distributions as sp_distributions

from ...core import ExecutableTuple
from ...lib.version import parse as parse_version
from ..arithmetic import (
    divide as mt_divide, sqrt as mt_sqrt, absolute as mt_abs,
    isnan as mt_isnan,
)
from ..base import where as mt_where
from ..reduction import (
    var as mt_var, mean as mt_mean,
)
from ..utils import implement_scipy


def _equal_var_ttest_denom(v1, n1, v2, n2):
    df = n1 + n2 - 2.0
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    denom = mt_sqrt(svar * (1.0 / n1 + 1.0 / n2))  # XXX: np -> da
    return df, denom


def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = mt_where(mt_isnan(df), 1, df)
    denom = mt_sqrt(vn1 + vn2)
    return df, denom


def _ttest_ind_from_stats(mean1, mean2, denom, df, alternative):

    d = mean1 - mean2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mt_divide(d, denom)
    t, prob = _ttest_finish(df, t, alternative)

    return t, prob


def _ttest_finish(df, t, alternative):
    """Common code between all 3 t-test functions."""
    if alternative != 'two-sided' and parse_version(sp_version) < parse_version('1.6.0'):  # pragma: no cover
        raise ValueError("alternative must be 'two-sided' with scipy prior to 1.6.0")

    if alternative == 'less':
        prob = t.map_chunk(sp_distributions.t.cdf, args=(df,))
    elif alternative == 'greater':
        prob = t.map_chunk(sp_distributions.t.sf, args=(df,))
    elif alternative == 'two-sided':
        prob = mt_abs(t).map_chunk(sp_distributions.t.sf, args=(df,)) * 2
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    if t.ndim == 0:
        t = t[()]
    return t, prob


Ttest_1sampResult = namedtuple('Ttest_1sampResult', ('statistic', 'pvalue'))


@implement_scipy(sp_ttest_1samp)
def ttest_1samp(a, popmean, axis=0, nan_policy="propagate", alternative="two-sided"):
    if nan_policy != "propagate":
        raise NotImplementedError(
            "`nan_policy` other than 'propagate' have not been implemented."
        )
    n = a.shape[axis]
    df = n - 1

    d = a.mean(axis=axis) - popmean
    v = a.var(axis=axis, ddof=1)
    denom = mt_sqrt(v / float(n))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = mt_divide(d, denom)
    t, prob = _ttest_finish(df, t, alternative)
    return ExecutableTuple(Ttest_1sampResult(t, prob))


Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))


@implement_scipy(sp_ttest_ind)
def ttest_ind(a, b, axis=0, equal_var=True, alternative="two-sided"):
    v1 = mt_var(a, axis, ddof=1)
    v2 = mt_var(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if equal_var:
        df, denom = _equal_var_ttest_denom(v1, n1, v2, n2)
    else:
        df, denom = _unequal_var_ttest_denom(v1, n1, v2, n2)

    res = _ttest_ind_from_stats(mt_mean(a, axis), mt_mean(b, axis), denom,
                                df, alternative)

    return ExecutableTuple(Ttest_indResult(*res))


@implement_scipy(sp_ttest_ind_from_stats)
def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2,
                         equal_var=True, alternative="two-sided"):
    if equal_var:
        df, denom = _equal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2)
    else:
        df, denom = _unequal_var_ttest_denom(std1**2, nobs1,
                                             std2**2, nobs2)

    res = _ttest_ind_from_stats(mean1, mean2, denom, df, alternative)
    return ExecutableTuple(Ttest_indResult(*res))


Ttest_relResult = namedtuple('Ttest_relResult', ('statistic', 'pvalue'))


@implement_scipy(sp_ttest_rel)
def ttest_rel(a, b, axis=0, nan_policy="propagate", alternative="two-sided"):
    if nan_policy != "propagate":
        raise NotImplementedError(
            "`nan_policy` other than 'propagate' have not been implemented."
        )

    n = a.shape[axis]
    df = float(n - 1)

    d = (a - b).astype(np.float64)
    v = mt_var(d, axis, ddof=1)
    dm = mt_mean(d, axis)
    denom = mt_sqrt(v / float(n))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = mt_divide(dm, denom)
    t, prob = _ttest_finish(df, t, alternative)

    return ExecutableTuple(Ttest_relResult(t, prob))
