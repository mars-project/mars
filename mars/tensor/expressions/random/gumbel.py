#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .... import operands
from .core import TensorRandomOperandMixin, handle_array


class TensorGumbel(operands.Gumbel, TensorRandomOperandMixin):
    __slots__ = '_loc', '_scale', '_size'
    _input_fields_ = ['_loc', '_scale']

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorGumbel, self).__init__(_state=state, _size=size, _dtype=dtype,
                                           _gpu=gpu, **kw)

    def __call__(self, loc, scale, chunk_size=None):
        return self.new_tensor([loc, scale], None, raw_chunk_size=chunk_size)


def gumbel(random_state, loc=0.0, scale=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Gumbel distribution.

    Draw samples from a Gumbel distribution with specified location and
    scale.  For more information on the Gumbel distribution, see
    Notes and References below.

    Parameters
    ----------
    loc : float or array_like of floats, optional
        The location of the mode of the distribution. Default is 0.
    scale : float or array_like of floats, optional
        The scale parameter of the distribution. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Gumbel distribution.

    See Also
    --------
    scipy.stats.gumbel_l
    scipy.stats.gumbel_r
    scipy.stats.genextreme
    weibull

    Notes
    -----
    The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
    Value Type I) distribution is one of a class of Generalized Extreme
    Value (GEV) distributions used in modeling extreme value problems.
    The Gumbel is a special case of the Extreme Value Type I distribution
    for maximums from distributions with "exponential-like" tails.

    The probability density for the Gumbel distribution is

    .. math:: p(x) = \frac{e^{-(x - \mu)/ \beta}}{\beta} e^{ -e^{-(x - \mu)/
              \beta}},

    where :math:`\mu` is the mode, a location parameter, and
    :math:`\beta` is the scale parameter.

    The Gumbel (named for German mathematician Emil Julius Gumbel) was used
    very early in the hydrology literature, for modeling the occurrence of
    flood events. It is also used for modeling maximum wind speed and
    rainfall rates.  It is a "fat-tailed" distribution - the probability of
    an event in the tail of the distribution is larger than if one used a
    Gaussian, hence the surprisingly frequent occurrence of 100-year
    floods. Floods were initially modeled as a Gaussian process, which
    underestimated the frequency of extreme events.

    It is one of a class of extreme value distributions, the Generalized
    Extreme Value (GEV) distributions, which also includes the Weibull and
    Frechet.

    The function has a mean of :math:`\mu + 0.57721\beta` and a variance
    of :math:`\frac{\pi^2}{6}\beta^2`.

    References
    ----------
    .. [1] Gumbel, E. J., "Statistics of Extremes,"
           New York: Columbia University Press, 1958.
    .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
           Values from Insurance, Finance, Hydrology and Other Fields,"
           Basel: Birkhauser Verlag, 2001.

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> mu, beta = 0, 0.1 # location and scale
    >>> s = mt.random.gumbel(mu, beta, 1000).execute()

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> count, bins, ignored = plt.hist(s, 30, normed=True)
    >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
    ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
    ...          linewidth=2, color='r')
    >>> plt.show()

    Show how an extreme value distribution can arise from a Gaussian process
    and compare to a Gaussian:

    >>> means = []
    >>> maxima = []
    >>> for i in range(0,1000) :
    ...    a = mt.random.normal(mu, beta, 1000)
    ...    means.append(a.mean().execute())
    ...    maxima.append(a.max().execute())
    >>> count, bins, ignored = plt.hist(maxima, 30, normed=True)
    >>> beta = mt.std(maxima) * mt.sqrt(6) / mt.pi
    >>> mu = mt.mean(maxima) - 0.57721*beta
    >>> plt.plot(bins, ((1/beta)*mt.exp(-(bins - mu)/beta)
    ...          * mt.exp(-mt.exp(-(bins - mu)/beta))).execute(),
    ...          linewidth=2, color='r')
    >>> plt.plot(bins, (1/(beta * mt.sqrt(2 * mt.pi))
    ...          * mt.exp(-(bins - mu)**2 / (2 * beta**2))).execute(),
    ...          linewidth=2, color='g')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().gumbel(
            handle_array(loc), handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorGumbel(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(loc, scale, chunk_size=chunk_size)
