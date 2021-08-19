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

import math
import warnings
from math import gcd
from collections import namedtuple
from typing import Callable, Tuple, Union

import numpy as np
from scipy import special
from scipy.stats import distributions

from ... import tensor as mt
from ...core import ExecutableTuple
from ...typing import TileableType


KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))
Ks_2sampResult = KstestResult


def _compute_prob_inside_method(m, n, g, h):  # pragma: no cover
    """
    Count the proportion of paths that stay strictly inside two diagonal lines.

    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)

    Returns
    -------
    p : float
        The proportion of paths that stay inside the two lines.


    Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    |x/m - y/n| < h / lcm(m, n).
    The paths make steps of size +1 in either positive x or positive y directions.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    """
    # Probability is symmetrical in m, n.  Computation below uses m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    # |nx/g - my/g| < h.
    # Compute matrix A such that:
    #  A(x, 0) = A(0, y) = 1
    #  A(x, y) = A(x, y-1) + A(x-1, y), for x,y>=1, except that
    #  A(x, y) = 0 if |x/m - y/n|>= h
    # Probability is A(m, n)/binom(m+n, n)
    # Optimizations exist for m==n, m==n*p.
    # Only need to preserve a single column of A, and only a sliding window of it.
    # minj keeps track of the slide.
    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # Make a vector long enough to hold maximum window needed.
    lenA = min(2 * maxj + 2, n + 1)
    # This is an integer calculation, but the entries are essentially
    # binomial coefficients, hence grow quickly.
    # Scaling after each column is computed avoids dividing by a
    # large binomial coefficient at the end, but is not sufficient to avoid
    # the large dynamic range which appears during the calculation.
    # Instead we rescale based on the magnitude of the right most term in
    # the column and keep track of an exponent separately and apply
    # it at the end of the calculation.  Similarly when multiplying by
    # the binomial coefficient
    dtype = np.float64
    A = np.zeros(lenA, dtype=dtype)
    # Initialize the first column
    A[minj:maxj] = 1
    expnt = 0
    for i in range(1, m + 1):
        # Generate the next column.
        # First calculate the sliding window
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 0
        # Now fill in the values
        A[0:maxj - minj] = np.cumsum(A[minj - lastminj:maxj - lastminj])
        curlen = maxj - minj
        if lastlen > curlen:
            # Set some carried-over elements to 0
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 0
        # Rescale if the right most value is over 2**900
        val = A[maxj - minj - 1]
        _, valexpt = math.frexp(val)
        if valexpt > 900:
            # Scaling to bring down to about 2**800 appears
            # sufficient for sizes under 10000.
            valexpt -= 800
            A = np.ldexp(A, -valexpt)
            expnt += valexpt

    val = A[maxj - minj - 1]
    # Now divide by the binomial (m+n)!/m!/n!
    for i in range(1, n + 1):
        val = (val * i) / (m + i)
        _, valexpt = math.frexp(val)
        if valexpt < -128:
            val = np.ldexp(val, -valexpt)
            expnt += valexpt
    # Finally scale if needed.
    return np.ldexp(val, expnt)


def _compute_prob_outside_square(n, h):  # pragma: no cover
    """
    Compute the proportion of paths that pass outside the two diagonal lines.

    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n

    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.

    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )  / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


def _count_paths_outside_method(m, n, g, h):  # pragma: no cover
    """
    Count the number of paths that pass outside the specified diagonal.

    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)

    Returns
    -------
    p : float
        The number of paths that go low.
        The calculation may overflow - check for a finite answer.

    Raises
    ------
    FloatingPointError: Raised if the intermediate computation goes outside
    the range of a float.

    Notes
    -----
    Count the integer lattice paths from (0, 0) to (m, n), which at some
    point (x, y) along the path, satisfy:
      m*y <= n*x - h*g
    The paths make steps of size +1 in either positive x or positive y directions.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    """
    # Compute #paths which stay lower than x/m-y/n = h/lcm(m,n)
    # B(x, y) = #{paths from (0,0) to (x,y) without previously crossing the boundary}
    #         = binom(x, y) - #{paths which already reached the boundary}
    # Multiply by the number of path extensions going from (x, y) to (m, n)
    # Sum.

    # Probability is symmetrical in m, n.  Computation below assumes m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Not every x needs to be considered.
    # xj holds the list of x values to be checked.
    # Wherever n*x/m + ng*h crosses an integer
    lxj = n + (mg-h)//mg
    xj = [(h + mg * j + ng-1)//ng for j in range(lxj)]
    # B is an array just holding a few values of B(x,y), the ones needed.
    # B[j] == B(x_j, j)
    if lxj == 0:
        return np.round(special.binom(m + n, n))
    B = np.zeros(lxj)
    B[0] = 1
    # Compute the B(x, y) terms
    # The binomial coefficient is an integer, but special.binom() may return a float.
    # Round it to the nearest integer.
    for j in range(1, lxj):
        Bj = np.round(special.binom(xj[j] + j, j))
        if not np.isfinite(Bj):
            raise FloatingPointError()
        for i in range(j):
            bin = np.round(special.binom(xj[j] - xj[i] + j - i, j-i))  # pylint: disable=redefined-builtin
            Bj -= bin * B[i]
        B[j] = Bj
        if not np.isfinite(Bj):
            raise FloatingPointError()
    # Compute the number of path extensions...
    num_paths = 0
    for j in range(lxj):
        bin = np.round(special.binom((m-xj[j]) + (n - j), n-j))
        term = B[j] * bin
        if not np.isfinite(term):
            raise FloatingPointError()
        num_paths += term
    return np.round(num_paths)


def _attempt_exact_2kssamp(n1, n2, g, d, alternative):  # pragma: no cover
    """Attempts to compute the exact 2sample probability.

    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs

    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        if alternative == 'two-sided':
            if n1 == n2:
                prob = _compute_prob_outside_square(n1, h)
            else:
                prob = 1 - _compute_prob_inside_method(n1, n2, g, h)
        else:
            if n1 == n2:
                # prob = binom(2n, n-h) / binom(2n, n)
                # Evaluating in that form incurs roundoff errors
                # from special.binom. Instead calculate directly
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)  # pylint: disable=redefined-builtin
                if not np.isfinite(bin) or not np.isfinite(num_paths) or num_paths > bin:
                    saw_fp_error = True
                else:
                    prob = num_paths / bin

    except FloatingPointError:
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob


def _calc_prob_2samp(d, n1, n2, alternative, mode):  # pragma: no cover
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N

    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -mt.inf
    original_mode = mode
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int_).max / n2g:
            mode = 'asymp'
            warnings.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning)

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            if original_mode == 'exact':
                warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                              f"Switching to mode={mode}.", RuntimeWarning)

    if mode == 'asymp':
        # The product n1*n2 is large.  Use Smirnov's asymptotic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    return np.clip(prob, 0, 1)


def _compute_dplus(cdfvals, n):
    """Computes D+ as used in the Kolmogorov-Smirnov test.
    Parameters
    ----------
    cdfvals: array_like
      Sorted array of CDF values between 0 and 1
    Returns
    -------
      Maximum distance of the CDF values below Uniform(0, 1)
    """
    return (mt.arange(1.0, n + 1) / n - cdfvals).max()


def _compute_dminus(cdfvals, n):
    """Computes D- as used in the Kolmogorov-Smirnov test.
    Parameters
    ----------
    cdfvals: array_like
      Sorted array of CDF values between 0 and 1
    Returns
    -------
      Maximum distance of the CDF values above Uniform(0, 1)
    """
    return (cdfvals - mt.arange(0.0, n) / n).max()


def ks_1samp(x: Union[np.ndarray, list, TileableType],
             cdf: Callable,
             args: Tuple = (),
             alternative: str = 'two-sided',
             mode: str = 'auto'):
    """
    Performs the one-sample Kolmogorov-Smirnov test for goodness of fit.

    This test compares the underlying distribution F(x) of a sample
    against a given continuous distribution G(x). See Notes for a description
    of the available null and alternative hypotheses.

    Parameters
    ----------
    x : array_like
        a 1-D array of observations of iid random variables.
    cdf : callable
        callable used to calculate the cdf.
    args : tuple, sequence, optional
        Distribution parameters, used with `cdf`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes below.
    mode : {'auto', 'exact', 'approx', 'asymp'}, optional
        Defines the distribution used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : selects one of the other options.
          * 'exact' : uses the exact distribution of test statistic.
          * 'approx' : approximates the two-sided probability with twice
            the one-sided probability
          * 'asymp': uses asymptotic distribution of test statistic

    Returns
    -------
    statistic : float
        KS test statistic, either D, D+ or D- (depending on the value
        of 'alternative')
    pvalue :  float
        One-tailed or two-tailed p-value.

    See Also
    --------
    ks_2samp, kstest

    Notes
    -----
    There are three options for the null and corresponding alternative
    hypothesis that can be selected using the `alternative` parameter.

    - `two-sided`: The null hypothesis is that the two distributions are
      identical, F(x)=G(x) for all x; the alternative is that they are not
      identical.

    - `less`: The null hypothesis is that F(x) >= G(x) for all x; the
      alternative is that F(x) < G(x) for at least one x.

    - `greater`: The null hypothesis is that F(x) <= G(x) for all x; the
      alternative is that F(x) > G(x) for at least one x.

    Note that the alternative hypotheses describe the *CDFs* of the
    underlying distributions, not the observed values. For example,
    suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in
    x1 tend to be less than those in x2.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import mars.tensor as mt
    >>> from mars.tensor.stats import ks_1samp

    >>> np.random.seed(12345678)  #fix random seed to get the same result
    >>> x = mt.linspace(-15, 15, 9, chunk_size=5)
    >>> ks_1samp(x, stats.norm.cdf).execute()
    (0.44435602715924361, 0.038850142705171065)

    >>> ks_1samp(stats.norm.rvs(size=100), stats.norm.cdf).execute()
    KstestResult(statistic=0.165471391799..., pvalue=0.007331283245...)

    *Test against one-sided alternative hypothesis*

    Shift distribution to larger values, so that `` CDF(x) < norm.cdf(x)``:

    >>> x = stats.norm.rvs(loc=0.2, size=100)
    >>> ks_1samp(x, stats.norm.cdf, alternative='less').execute()
    KstestResult(statistic=0.235488541678..., pvalue=1.158315030683...)

    Reject null hypothesis in favor of alternative hypothesis: less

    >>> ks_1samp(x, stats.norm.cdf, alternative='greater').execute()
    KstestResult(statistic=0.010167165616..., pvalue=0.972494973653...)

    Reject null hypothesis in favor of alternative hypothesis: greater

    >>> ks_1samp(x, stats.norm.cdf).execute()
    KstestResult(statistic=0.235488541678..., pvalue=2.316630061366...)

    Don't reject null hypothesis in favor of alternative hypothesis: two-sided

    *Testing t distributed random variables against normal distribution*

    With 100 degrees of freedom the t distribution looks close to the normal
    distribution, and the K-S test does not reject the hypothesis that the
    sample came from the normal distribution:

    >>> ks_1samp(stats.t.rvs(100, size=100), stats.norm.cdf).execute()
    KstestResult(statistic=0.077844250253..., pvalue=0.553155412513...)

    With 3 degrees of freedom the t distribution looks sufficiently different
    from the normal distribution, that we can reject the hypothesis that the
    sample came from the normal distribution at the 10% level:

    >>> ks_1samp(stats.t.rvs(3, size=100), stats.norm.cdf).execute()
    KstestResult(statistic=0.118967105356..., pvalue=0.108627114578...)
    """
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError("Unexpected alternative %s" % alternative)

    x = mt.asarray(x)
    N = x.shape[0]
    x = mt.sort(x)
    cdfvals = x.map_chunk(cdf, args=args, elementwise=True)

    if alternative == 'greater':
        Dplus = _compute_dplus(cdfvals, N)
        return ExecutableTuple(KstestResult(
            Dplus, Dplus.map_chunk(distributions.ksone.sf, args=(N,))))

    if alternative == 'less':
        Dminus = _compute_dminus(cdfvals, N)
        return ExecutableTuple(KstestResult(
            Dminus, Dminus.map_chunk(distributions.ksone.sf, args=(N,))))

    # alternative == 'two-sided':
    Dplus = _compute_dplus(cdfvals, N)
    Dminus = _compute_dminus(cdfvals, N)
    D = mt.stack([Dplus, Dminus]).max()
    if mode == 'auto':  # Always select exact
        mode = 'exact'
    if mode == 'exact':
        prob = D.map_chunk(distributions.kstwo.sf, args=(N,),
                           elementwise=True)
    elif mode == 'asymp':
        prob = (D * np.sqrt(N)).map_chunk(distributions.kstwobign.sf,
                                          elementwise=True)
    else:
        # mode == 'approx'
        prob = 2 * D.map_chunk(distributions.ksone.sf, args=(N,),
                               elementwise=True)
    prob = mt.clip(prob, 0, 1)
    return ExecutableTuple(KstestResult(D, prob))


def ks_2samp(data1: Union[np.ndarray, list, TileableType],
             data2: Union[np.ndarray, list, TileableType],
             alternative: str = 'two-sided',
             mode: str = 'auto'):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.  The alternative hypothesis
    can be either 'two-sided' (default), 'less' or 'greater'.

    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided'
          * 'less': one-sided, see explanation in Notes
          * 'greater': one-sided, see explanation in Notes
    mode : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    statistic : float
        KS statistic.
    pvalue : float
        Two-tailed p-value.

    See Also
    --------
    kstest, ks_1samp, epps_singleton_2samp, anderson_ksamp

    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample KS test, the distribution is
    assumed to be continuous.

    In the one-sided test, the alternative is that the empirical
    cumulative distribution function F(x) of the data1 variable is "less"
    or "greater" than the empirical cumulative distribution function G(x)
    of the data2 variable, ``F(x)<=G(x)``, resp. ``F(x)>=G(x)``.

    If the KS statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.

    If the mode is 'auto', the computation is exact if the sample sizes are
    less than 10000.  For larger sizes, the computation uses the
    Kolmogorov-Smirnov distributions to compute an approximate value.

    The 'two-sided' 'exact' computation computes the complementary probability
    and then subtracts from 1.  As such, the minimum probability it can return
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.

    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import mars.tensor as mt
    >>> from mars.tensor.stats import ks_2samp
    >>> np.random.seed(12345678)  #fix random seed to get the same result
    >>> n1 = 200  # size of first sample
    >>> n2 = 300  # size of second sample

    For a different distribution, we can reject the null hypothesis since the
    pvalue is below 1%:

    >>> rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
    >>> rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
    >>> ks_2samp(rvs1, rvs2).execute()
    KstestResult(statistic=0.20833333333333337, pvalue=5.1292795978041816e-05)

    For a slightly different distribution, we cannot reject the null hypothesis
    at a 10% or lower alpha since the p-value at 0.144 is higher than 10%

    >>> rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
    >>> ks_2samp(rvs1, rvs3).execute()
    KstestResult(statistic=0.10333333333333333, pvalue=0.14691437867433788)

    For an identical distribution, we cannot reject the null hypothesis since
    the p-value is high, 41%:

    >>> rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
    >>> ks_2samp(rvs1, rvs4).execute()
    KstestResult(statistic=0.07999999999999996, pvalue=0.4115432028915931)

    """

    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    data1 = mt.asarray(data1)
    data2 = mt.asarray(data2)
    data1 = mt.sort(data1)
    data2 = mt.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = mt.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = mt.searchsorted(data1, data_all, side='right') / n1
    cdf2 = mt.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2
    minS = mt.clip(-mt.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = mt.max(cddiffs)
    alt2Dvalue = {'less': minS, 'greater': maxS, 'two-sided': mt.maximum(minS, maxS)}
    d = alt2Dvalue[alternative]
    prob = d.map_chunk(_calc_prob_2samp, args=(n1, n2, alternative, mode),
                       elementwise=True, dtype=d.dtype)

    return ExecutableTuple(Ks_2sampResult(d, prob))
