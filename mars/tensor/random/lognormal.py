#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ... import opcodes as OperandDef
from ...serialize import AnyField
from .core import TensorRandomOperandMixin, handle_array, TensorDistribution


class TensorLognormal(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_mean', '_sigma', '_size'
    _input_fields_ = ['_mean', '_sigma']
    _op_type_ = OperandDef.RAND_LOGNORMAL

    _mean = AnyField('mean')
    _sigma = AnyField('sigma')
    _func_name = 'lognormal'

    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_state=state, _size=size, _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, mean, sigma, chunk_size=None):
        return self.new_tensor([mean, sigma], None, raw_chunk_size=chunk_size)


def lognormal(random_state, mean=0.0, sigma=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a log-normal distribution.

    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.  Note that the mean and standard
    deviation are not the values for the distribution itself, but of the
    underlying normal distribution it is derived from.

    Parameters
    ----------
    mean : float or array_like of floats, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float or array_like of floats, optional
        Standard deviation of the underlying normal distribution. Should
        be greater than zero. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``sigma`` are both scalars.
        Otherwise, ``np.broadcast(mean, sigma).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized log-normal distribution.

    See Also
    --------
    scipy.stats.lognorm : probability density function, distribution,
        cumulative density function, etc.

    Notes
    -----
    A variable `x` has a log-normal distribution if `log(x)` is normally
    distributed.  The probability density function for the log-normal
    distribution is:

    .. math:: p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
                     e^{(-\frac{(ln(x)-\mu)^2}{2\sigma^2})}

    where :math:`\mu` is the mean and :math:`\sigma` is the standard
    deviation of the normally distributed logarithm of the variable.
    A log-normal distribution results if a random variable is the *product*
    of a large number of independent, identically-distributed variables in
    the same way that a normal distribution results if the variable is the
    *sum* of a large number of independent, identically-distributed
    variables.

    References
    ----------
    .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
           Distributions across the Sciences: Keys and Clues,"
           BioScience, Vol. 51, No. 5, May, 2001.
           http://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
    .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
           Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> mu, sigma = 3., 1. # mean and standard deviation
    >>> s = mt.random.lognormal(mu, sigma, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(s.execute(), 100, normed=True, align='mid')

    >>> x = mt.linspace(min(bins), max(bins), 10000)
    >>> pdf = (mt.exp(-(mt.log(x) - mu)**2 / (2 * sigma**2))
    ...        / (x * sigma * mt.sqrt(2 * mt.pi)))

    >>> plt.plot(x.execute(), pdf.execute(), linewidth=2, color='r')
    >>> plt.axis('tight')
    >>> plt.show()

    Demonstrate that taking the products of random samples from a uniform
    distribution can be fit well by a log-normal probability density
    function.

    >>> # Generate a thousand samples: each is the product of 100 random
    >>> # values, drawn from a normal distribution.
    >>> b = []
    >>> for i in range(1000):
    ...    a = 10. + mt.random.random(100)
    ...    b.append(mt.product(a).execute())

    >>> b = mt.array(b) / mt.min(b) # scale values to be positive
    >>> count, bins, ignored = plt.hist(b.execute(), 100, normed=True, align='mid')
    >>> sigma = mt.std(mt.log(b))
    >>> mu = mt.mean(mt.log(b))

    >>> x = mt.linspace(min(bins), max(bins), 10000)
    >>> pdf = (mt.exp(-(mt.log(x) - mu)**2 / (2 * sigma**2))
    ...        / (x * sigma * mt.sqrt(2 * mt.pi)))

    >>> plt.plot(x.execute(), pdf.execute(), color='r', linewidth=2)
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().lognormal(
            handle_array(mean), handle_array(sigma), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorLognormal(state=random_state.to_numpy(), size=size, gpu=gpu, dtype=dtype)
    return op(mean, sigma, chunk_size=chunk_size)
