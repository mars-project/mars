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


class TensorGamma(operands.Gamma, TensorRandomOperandMixin):
    __slots__ = '_shape', '_scale', '_size'
    _input_fields_ = ['_shape', '_scale']

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorGamma, self).__init__(_state=state, _size=size, _dtype=dtype,
                                          _gpu=gpu, **kw)

    def __call__(self, shape, scale, chunk_size=None):
        return self.new_tensor([shape, scale], None, raw_chunk_size=chunk_size)


def gamma(random_state, shape, scale=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    Parameters
    ----------
    shape : float or array_like of floats
        The shape of the gamma distribution. Should be greater than zero.
    scale : float or array_like of floats, optional
        The scale of the gamma distribution. Should be greater than zero.
        Default is equal to 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(shape, scale).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized gamma distribution.

    See Also
    --------
    scipy.stats.gamma : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

    where :math:`k` is the shape and :math:`\theta` the scale,
    and :math:`\Gamma` is the Gamma function.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.

    References
    ----------
    .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/GammaDistribution.html
    .. [2] Wikipedia, "Gamma distribution",
           http://en.wikipedia.org/wiki/Gamma_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
    >>> s = mt.random.gamma(shape, scale, 1000).execute()

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> import scipy.special as sps
    >>> import numpy as np
    >>> count, bins, ignored = plt.hist(s, 50, normed=True)
    >>> y = bins**(shape-1)*(np.exp(-bins/scale) /
    ...                      (sps.gamma(shape)*scale**shape))
    >>> plt.plot(bins, y, linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().gamma(
            handle_array(shape), handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorGamma(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(shape, scale, chunk_size=chunk_size)
