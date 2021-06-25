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

from .cov import cov


def corrcoef(x, y=None, rowvar=True):
    r"""
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.

    Returns
    -------
    R : Tensor
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    Notes
    -----
    Due to floating point rounding the resulting tensor may not be Hermitian,
    the diagonal elements may not be 1, and the elements may not satisfy the
    inequality abs(a) <= 1. The real and imaginary parts are clipped to the
    interval [-1,  1] in an attempt to improve on that situation but is not
    much help in the complex case.

    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    """
    from ..arithmetic import sqrt
    from ..datasource import diag

    c = cov(x, y, rowvar)
    if c.ndim == 0:
        return c / c
    d = diag(c)
    d = d.reshape(d.shape[0], 1)
    sqrt_d = sqrt(d)
    return (c / sqrt_d) / sqrt_d.T
