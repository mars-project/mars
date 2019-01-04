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


class TensorLaplace(operands.Laplace, TensorRandomOperandMixin):
    __slots__ = '_loc', '_scale', '_size'
    _input_fields_ = ['_loc', '_scale']

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorLaplace, self).__init__(_state=state, _size=size,
                                            _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, loc, scale, chunk_size=None):
        return self.new_tensor([loc, scale], None, raw_chunk_size=chunk_size)


def laplace(random_state, loc=0.0, scale=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    The Laplace distribution is similar to the Gaussian/normal distribution,
    but is sharper at the peak and has fatter tails. It represents the
    difference between two independent, identically distributed exponential
    random variables.

    Parameters
    ----------
    loc : float or array_like of floats, optional
        The position, :math:`\mu`, of the distribution peak. Default is 0.
    scale : float or array_like of floats, optional
        :math:`\lambda`, the exponential decay. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    chunks : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Laplace distribution.

    Notes
    -----
    It has the probability density function

    .. math:: f(x; \mu, \lambda) = \frac{1}{2\lambda}
                                   \exp\left(-\frac{|x - \mu|}{\lambda}\right).

    The first law of Laplace, from 1774, states that the frequency
    of an error can be expressed as an exponential function of the
    absolute magnitude of the error, which leads to the Laplace
    distribution. For many problems in economics and health
    sciences, this distribution seems to model the data better
    than the standard Gaussian distribution.

    References
    ----------
    .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
           Mathematical Functions with Formulas, Graphs, and Mathematical
           Tables, 9th printing," New York: Dover, 1972.
    .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
           Generalizations, " Birkhauser, 2001.
    .. [3] Weisstein, Eric W. "Laplace Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/LaplaceDistribution.html
    .. [4] Wikipedia, "Laplace distribution",
           http://en.wikipedia.org/wiki/Laplace_distribution

    Examples
    --------
    Draw samples from the distribution

    >>> import mars.tensor as mt

    >>> loc, scale = 0., 1.
    >>> s = mt.random.laplace(loc, scale, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(s.execute(), 30, normed=True)
    >>> x = mt.arange(-8., 8., .01)
    >>> pdf = mt.exp(-abs(x-loc)/scale)/(2.*scale)
    >>> plt.plot(x.execute(), pdf.execute())

    Plot Gaussian for comparison:

    >>> g = (1/(scale * mt.sqrt(2 * np.pi)) *
    ...      mt.exp(-(x - loc)**2 / (2 * scale**2)))
    >>> plt.plot(x.execute(),g.execute())
    """
    if dtype is None:
        dtype = np.random.RandomState().laplace(
            handle_array(loc), handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorLaplace(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(loc, scale, chunk_size=chunk_size)
