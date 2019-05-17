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

from .... import opcodes as OperandDef
from ....serialize import AnyField
from .core import TensorRandomOperandMixin, handle_array, TensorDistribution


class TensorStandardGamma(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_shape', '_size'
    _input_fields_ = ['_shape']
    _op_type_ = OperandDef.RAND_STANDARD_GAMMMA

    _shape = AnyField('shape')

    @property
    def shape(self):
        return self._shape

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorStandardGamma, self).__init__(_size=size, _state=state, _dtype=dtype,
                                                  _gpu=gpu, **kw)

    def __call__(self, shape, chunk_size=None):
        return self.new_tensor([shape], None, raw_chunk_size=chunk_size)


def standard_gamma(random_state, shape, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a standard Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated "k") and scale=1.

    Parameters
    ----------
    shape : float or array_like of floats
        Parameter, should be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` is a scalar.  Otherwise,
        ``mt.array(shape).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized standard gamma distribution.

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

    >>> shape, scale = 2., 1. # mean and width
    >>> s = mt.random.standard_gamma(shape, 1000000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> import scipy.special as sps
    >>> count, bins, ignored = plt.hist(s.execute(), 50, normed=True)
    >>> y = bins**(shape-1) * ((mt.exp(-bins/scale))/ \
    ...                       (sps.gamma(shape) * scale**shape))
    >>> plt.plot(bins, y.execute(), linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().standard_gamma(
            handle_array(shape), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorStandardGamma(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(shape, chunk_size=chunk_size)
