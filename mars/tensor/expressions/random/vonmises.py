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


class TensorVonmises(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_mu', '_kappa', '_size'
    _input_fields_ = ['_mu', '_kappa']
    _op_type_ = OperandDef.RAND_VONMISES

    _mu = AnyField('mu')
    _kappa = AnyField('kappa')

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorVonmises, self).__init__(_size=size, _state=state, _dtype=dtype,
                                             _gpu=gpu, **kw)

    @property
    def mu(self):
        return self._mu

    @property
    def kappa(self):
        return self._kappa

    def __call__(self, mu, kappa, chunk_size=None):
        return self.new_tensor([mu, kappa], None, raw_chunk_size=chunk_size)


def vonmises(random_state, mu, kappa, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a von Mises distribution.

    Samples are drawn from a von Mises distribution with specified mode
    (mu) and dispersion (kappa), on the interval [-pi, pi].

    The von Mises distribution (also known as the circular normal
    distribution) is a continuous probability distribution on the unit
    circle.  It may be thought of as the circular analogue of the normal
    distribution.

    Parameters
    ----------
    mu : float or array_like of floats
        Mode ("center") of the distribution.
    kappa : float or array_like of floats
        Dispersion of the distribution, has to be >=0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mu`` and ``kappa`` are both scalars.
        Otherwise, ``np.broadcast(mu, kappa).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized von Mises distribution.

    See Also
    --------
    scipy.stats.vonmises : probability density function, distribution, or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the von Mises distribution is

    .. math:: p(x) = \frac{e^{\kappa cos(x-\mu)}}{2\pi I_0(\kappa)},

    where :math:`\mu` is the mode and :math:`\kappa` the dispersion,
    and :math:`I_0(\kappa)` is the modified Bessel function of order 0.

    The von Mises is named for Richard Edler von Mises, who was born in
    Austria-Hungary, in what is now the Ukraine.  He fled to the United
    States in 1939 and became a professor at Harvard.  He worked in
    probability theory, aerodynamics, fluid mechanics, and philosophy of
    science.

    References
    ----------
    .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
           Mathematical Functions with Formulas, Graphs, and Mathematical
           Tables, 9th printing," New York: Dover, 1972.
    .. [2] von Mises, R., "Mathematical Theory of Probability
           and Statistics", New York: Academic Press, 1964.

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> mu, kappa = 0.0, 4.0 # mean and dispersion
    >>> s = mt.random.vonmises(mu, kappa, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import i0
    >>> plt.hist(s.execute(), 50, normed=True)
    >>> x = mt.linspace(-mt.pi, mt.pi, num=51)
    >>> y = mt.exp(kappa*mt.cos(x-mu))/(2*mt.pi*i0(kappa))
    >>> plt.plot(x.execute(), y.execute(), linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().vonmises(
            handle_array(mu), handle_array(kappa), size=(0,)).dtype

    size = random_state._handle_size(size)
    op = TensorVonmises(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(mu, kappa, chunk_size=chunk_size)
