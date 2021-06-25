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

from .power_divergence import power_divergence


def chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    """
    Calculate a one-way chi-square test.

    The chi-square test tests the null hypothesis that the categorical data
    has the given frequencies.

    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.

    Returns
    -------
    chisq : float or ndarray
        The chi-squared test statistic.  The value is a float if `axis` is
        None or `f_obs` and `f_exp` are 1-D.
    p : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `chisq` are scalars.

    See Also
    --------
    scipy.stats.power_divergence

    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.

    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not chi-square, in which case this test
    is not appropriate.

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171022032306/http://vassarstats.net:80/textbook/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.

    >>> import mars.tensor as mt
    >>> from mars.tensor.stats import chisquare
    >>> chisquare([16, 18, 16, 14, 12, 12])
    (2.0, 0.84914503608460956)

    With `f_exp` the expected frequencies can be given.

    >>> chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8]).execute()
    (3.5, 0.62338762774958223)

    When `f_obs` is 2-D, by default the test is applied to each column.

    >>> obs = mt.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> chisquare(obs).execute()
    (array([ 2.        ,  6.66666667]), array([ 0.84914504,  0.24663415]))

    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.

    >>> chisquare(obs, axis=None).execute()
    (23.31034482758621, 0.015975692534127565)
    >>> chisquare(obs.ravel()).execute()
    (23.31034482758621, 0.015975692534127565)

    `ddof` is the change to make to the default degrees of freedom.

    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=1).execute()
    (2.0, 0.73575888234288467)

    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we use ``axis=1``:

    >>> chisquare([16, 18, 16, 14, 12, 12],
    ...           f_exp=[[16, 16, 16, 16, 16, 8], [8, 20, 20, 16, 12, 12]],
    ...           axis=1).execute()
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))

    """
    return power_divergence(f_obs, f_exp=f_exp, ddof=ddof, axis=axis,
                            lambda_="pearson")
