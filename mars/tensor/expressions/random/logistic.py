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


class TensorLogistic(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_loc', '_scale', '_size'
    _input_fields_ = ['_loc', '_scale']
    _op_type_ = OperandDef.RAND_LOGISTIC

    _loc = AnyField('loc')
    _scale = AnyField('scale')

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorLogistic, self).__init__(_state=state, _size=size,
                                             _dtype=dtype, _gpu=gpu, **kw)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    def __call__(self, loc, scale, chunk_size=None):
        return self.new_tensor([loc, scale], None, raw_chunk_size=chunk_size)


def logistic(random_state, loc=0.0, scale=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a logistic distribution.

    Samples are drawn from a logistic distribution with specified
    parameters, loc (location or mean, also median), and scale (>0).

    Parameters
    ----------
    loc : float or array_like of floats, optional
        Parameter of the distribution. Default is 0.
    scale : float or array_like of floats, optional
        Parameter of the distribution. Should be greater than zero.
        Default is 1.
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
        Drawn samples from the parameterized logistic distribution.

    See Also
    --------
    scipy.stats.logistic : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Logistic distribution is

    .. math:: P(x) = P(x) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2},

    where :math:`\mu` = location and :math:`s` = scale.

    The Logistic distribution is used in Extreme Value problems where it
    can act as a mixture of Gumbel distributions, in Epidemiology, and by
    the World Chess Federation (FIDE) where it is used in the Elo ranking
    system, assuming the performance of each player is a logistically
    distributed random variable.

    References
    ----------
    .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
           Extreme Values, from Insurance, Finance, Hydrology and Other
           Fields," Birkhauser Verlag, Basel, pp 132-133.
    .. [2] Weisstein, Eric W. "Logistic Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/LogisticDistribution.html
    .. [3] Wikipedia, "Logistic-distribution",
           http://en.wikipedia.org/wiki/Logistic_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt
    >>> import matplotlib.pyplot as plt

    >>> loc, scale = 10, 1
    >>> s = mt.random.logistic(loc, scale, 10000)
    >>> count, bins, ignored = plt.hist(s.execute(), bins=50)

    #   plot against distribution

    >>> def logist(x, loc, scale):
    ...     return mt.exp((loc-x)/scale)/(scale*(1+mt.exp((loc-x)/scale))**2)
    >>> plt.plot(bins, logist(bins, loc, scale).execute()*count.max()/\
    ... logist(bins, loc, scale).max().execute())
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().logistic(
            handle_array(loc), handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorLogistic(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(loc, scale, chunk_size=chunk_size)
