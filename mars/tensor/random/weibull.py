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

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import AnyField
from ..utils import gen_random_seeds
from .core import TensorRandomOperandMixin, handle_array, TensorDistribution


class TensorWeibull(TensorDistribution, TensorRandomOperandMixin):
    _input_fields_ = ['_a']
    _op_type_ = OperandDef.RAND_WEIBULL

    _fields_ = '_a', '_size'
    _a = AnyField('a')
    _func_name = 'weibull'

    def __init__(self, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, dtype=dtype, **kw)

    @property
    def a(self):
        return self._a

    def __call__(self, a, chunk_size=None):
        return self.new_tensor([a], None, raw_chunk_size=chunk_size)


def weibull(random_state, a, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Weibull distribution.

    Draw samples from a 1-parameter Weibull distribution with the given
    shape parameter `a`.

    .. math:: X = (-ln(U))^{1/a}

    Here, U is drawn from the uniform distribution over (0,1].

    The more common 2-parameter Weibull, including a scale parameter
    :math:`\lambda` is just :math:`X = \lambda(-ln(U))^{1/a}`.

    Parameters
    ----------
    a : float or array_like of floats
        Shape of the distribution. Should be greater than zero.
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
        Drawn samples from the parameterized Weibull distribution.

    See Also
    --------
    scipy.stats.weibull_max
    scipy.stats.weibull_min
    scipy.stats.genextreme
    gumbel

    Notes
    -----
    The Weibull (or Type III asymptotic extreme value distribution
    for smallest values, SEV Type III, or Rosin-Rammler
    distribution) is one of a class of Generalized Extreme Value
    (GEV) distributions used in modeling extreme value problems.
    This class includes the Gumbel and Frechet distributions.

    The probability density for the Weibull distribution is

    .. math:: p(x) = \frac{a}
                     {\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^a},

    where :math:`a` is the shape and :math:`\lambda` the scale.

    The function has its peak (the mode) at
    :math:`\lambda(\frac{a-1}{a})^{1/a}`.

    When ``a = 1``, the Weibull distribution reduces to the exponential
    distribution.

    References
    ----------
    .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
           1939 "A Statistical Theory Of The Strength Of Materials",
           Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
           Generalstabens Litografiska Anstalts Forlag, Stockholm.
    .. [2] Waloddi Weibull, "A Statistical Distribution Function of
           Wide Applicability", Journal Of Applied Mechanics ASME Paper
           1951.
    .. [3] Wikipedia, "Weibull distribution",
           http://en.wikipedia.org/wiki/Weibull_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> a = 5. # shape
    >>> s = mt.random.weibull(a, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> x = mt.arange(1,100.)/50.
    >>> def weib(x,n,a):
    ...     return (a / n) * (x / n)**(a - 1) * mt.exp(-(x / n)**a)

    >>> count, bins, ignored = plt.hist(mt.random.weibull(5.,1000).execute())
    >>> x = mt.arange(1,100.)/50.
    >>> scale = count.max()/weib(x, 1., 5.).max()
    >>> plt.plot(x.execute(), (weib(x, 1., 5.)*scale).execute())
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().weibull(
            handle_array(a), size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorWeibull(size=size, seed=seed, gpu=gpu, dtype=dtype)
    return op(a, chunk_size=chunk_size)
