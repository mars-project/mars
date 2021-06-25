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


class TensorPoisson(TensorDistribution, TensorRandomOperandMixin):
    _input_fields_ = ['_lam']
    _op_type_ = OperandDef.RAND_POSSION

    _fields_ = '_lam', '_size'
    _lam = AnyField('lam')
    _func_name = 'poisson'

    def __init__(self, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, dtype=dtype, **kw)

    @property
    def lam(self):
        return self._lam

    def __call__(self, lam, chunk_size=None):
        return self.new_tensor([lam], None, raw_chunk_size=chunk_size)


def poisson(random_state, lam=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the binomial distribution
    for large N.

    Parameters
    ----------
    lam : float or array_like of floats
        Expectation of interval, should be >= 0. A sequence of expectation
        intervals must be broadcastable over the requested size.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``lam`` is a scalar. Otherwise,
        ``mt.array(lam).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Poisson distribution.

    Notes
    -----
    The Poisson distribution

    .. math:: f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}

    For events with an expected separation :math:`\lambda` the Poisson
    distribution :math:`f(k; \lambda)` describes the probability of
    :math:`k` events occurring within the observed
    interval :math:`\lambda`.

    Because the output is limited to the range of the C long type, a
    ValueError is raised when `lam` is within 10 sigma of the maximum
    representable value.

    References
    ----------
    .. [1] Weisstein, Eric W. "Poisson Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/PoissonDistribution.html
    .. [2] Wikipedia, "Poisson distribution",
           http://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt
    >>> s = mt.random.poisson(5, 10000)

    Display histogram of the sample:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(s.execute(), 14, normed=True)
    >>> plt.show()

    Draw each 100 values for lambda 100 and 500:

    >>> s = mt.random.poisson(lam=(100., 500.), size=(100, 2))
    """
    if dtype is None:
        dtype = np.random.RandomState().poisson(
            handle_array(lam), size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorPoisson(size=size, seed=seed, gpu=gpu, dtype=dtype)
    return op(lam, chunk_size=chunk_size)
