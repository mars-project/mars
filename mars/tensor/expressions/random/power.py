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


class TensorRandomPower(operands.RandomPower, TensorRandomOperandMixin):
    __slots__ = '_a', '_size'
    _input_fields_ = ['_a']

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorRandomPower, self).__init__(_size=size, _state=state, _dtype=dtype,
                                                _gpu=gpu, **kw)

    def __call__(self, a, chunk_size=None):
        return self.new_tensor([a], None, raw_chunk_size=chunk_size)


def power(random_state, a, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Draws samples in [0, 1] from a power distribution with positive
    exponent a - 1.

    Also known as the power function distribution.

    Parameters
    ----------
    a : float or array_like of floats
        Parameter of the distribution. Should be greater than zero.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.  Otherwise,
        ``mt.array(a).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized power distribution.

    Raises
    ------
    ValueError
        If a < 1.

    Notes
    -----
    The probability density function is

    .. math:: P(x; a) = ax^{a-1}, 0 \le x \le 1, a>0.

    The power function distribution is just the inverse of the Pareto
    distribution. It may also be seen as a special case of the Beta
    distribution.

    It is used, for example, in modeling the over-reporting of insurance
    claims.

    References
    ----------
    .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
           in economics and actuarial sciences", Wiley, 2003.
    .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
           Dataplot Reference Manual, Volume 2: Let Subcommands and Library
           Functions", National Institute of Standards and Technology
           Handbook Series, June 2003.
           http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> a = 5. # shape
    >>> samples = 1000
    >>> s = mt.random.power(a, samples)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(s.execute(), bins=30)
    >>> x = mt.linspace(0, 1, 100)
    >>> y = a*x**(a-1.)
    >>> normed_y = samples*mt.diff(bins)[0]*y
    >>> plt.plot(x.execute(), normed_y.execute())
    >>> plt.show()

    Compare the power function distribution to the inverse of the Pareto.

    >>> from scipy import stats
    >>> rvs = mt.random.power(5, 1000000)
    >>> rvsp = mt.random.pareto(5, 1000000)
    >>> xx = mt.linspace(0,1,100)
    >>> powpdf = stats.powerlaw.pdf(xx.execute(),5)

    >>> plt.figure()
    >>> plt.hist(rvs.execute(), bins=50, normed=True)
    >>> plt.plot(xx.execute(),powpdf,'r-')
    >>> plt.title('np.random.power(5)')

    >>> plt.figure()
    >>> plt.hist((1./(1.+rvsp)).execute(), bins=50, normed=True)
    >>> plt.plot(xx.execute(),powpdf,'r-')
    >>> plt.title('inverse of 1 + np.random.pareto(5)')

    >>> plt.figure()
    >>> plt.hist((1./(1.+rvsp)).execute(), bins=50, normed=True)
    >>> plt.plot(xx.execute(),powpdf,'r-')
    >>> plt.title('inverse of stats.pareto(5)')
    """
    if dtype is None:
        dtype = np.random.RandomState().power(
            handle_array(a), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorRandomPower(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(a, chunk_size=chunk_size)
