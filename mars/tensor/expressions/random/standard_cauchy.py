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
from .core import TensorRandomOperandMixin, TensorDistribution


class TensorStandardCauchy(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_STANDARD_CAUCHY

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorStandardCauchy, self).__init__(_size=size, _state=state, _dtype=dtype,
                                                   _gpu=gpu, **kw)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def standard_cauchy(random_state, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a standard Cauchy distribution with mode = 0.

    Also known as the Lorentz distribution.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    samples : Tensor or scalar
        The drawn samples.

    Notes
    -----
    The probability density function for the full Cauchy distribution is

    .. math:: P(x; x_0, \gamma) = \frac{1}{\pi \gamma \bigl[ 1+
              (\frac{x-x_0}{\gamma})^2 \bigr] }

    and the Standard Cauchy distribution just sets :math:`x_0=0` and
    :math:`\gamma=1`

    The Cauchy distribution arises in the solution to the driven harmonic
    oscillator problem, and also describes spectral line broadening. It
    also describes the distribution of values at which a line tilted at
    a random angle will cut the x axis.

    When studying hypothesis tests that assume normality, seeing how the
    tests perform on data from a Cauchy distribution is a good indicator of
    their sensitivity to a heavy-tailed distribution, since the Cauchy looks
    very much like a Gaussian distribution, but with heavier tails.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
          Distribution",
          http://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
    .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
          Wolfram Web Resource.
          http://mathworld.wolfram.com/CauchyDistribution.html
    .. [3] Wikipedia, "Cauchy distribution"
          http://en.wikipedia.org/wiki/Cauchy_distribution

    Examples
    --------
    Draw samples and plot the distribution:

    >>> import mars.tensor as mt
    >>> import matplotlib.pyplot as plt

    >>> s = mt.random.standard_cauchy(1000000)
    >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
    >>> plt.hist(s.execute(), bins=100)
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().standard_cauchy(size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorStandardCauchy(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
