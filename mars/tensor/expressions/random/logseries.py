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


class TensorLogseries(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_p', '_size'
    _input_fields_ = ['_p']
    _op_type_ = OperandDef.RAND_LOGSERIES

    _p = AnyField('p')

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorLogseries, self).__init__(_state=state, _size=size, _dtype=dtype,
                                              _gpu=gpu, **kw)

    @property
    def p(self):
        return self._p

    def __call__(self, p, chunk_size=None):
        return self.new_tensor([p], None, raw_chunk_size=chunk_size)


def logseries(random_state, p, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a logarithmic series distribution.

    Samples are drawn from a log series distribution with specified
    shape parameter, 0 < ``p`` < 1.

    Parameters
    ----------
    p : float or array_like of floats
        Shape parameter for the distribution.  Must be in the range (0, 1).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``p`` is a scalar.  Otherwise,
        ``np.array(p).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized logarithmic series distribution.

    See Also
    --------
    scipy.stats.logser : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Log Series distribution is

    .. math:: P(k) = \frac{-p^k}{k \ln(1-p)},

    where p = probability.

    The log series distribution is frequently used to represent species
    richness and occurrence, first proposed by Fisher, Corbet, and
    Williams in 1943 [2].  It may also be used to model the numbers of
    occupants seen in cars [3].

    References
    ----------
    .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
           species diversity through the log series distribution of
           occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
           Volume 5, Number 5, September 1999 , pp. 187-195(9).
    .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
           relation between the number of species and the number of
           individuals in a random sample of an animal population.
           Journal of Animal Ecology, 12:42-58.
    .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
           Data Sets, CRC Press, 1994.
    .. [4] Wikipedia, "Logarithmic distribution",
           http://en.wikipedia.org/wiki/Logarithmic_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt
    >>> import matplotlib.pyplot as plt

    >>> a = .6
    >>> s = mt.random.logseries(a, 10000)
    >>> count, bins, ignored = plt.hist(s.execute())

    #   plot against distribution

    >>> def logseries(k, p):
    ...     return -p**k/(k*mt.log(1-p))
    >>> plt.plot(bins, (logseries(bins, a)*count.max()/
    ...          logseries(bins, a).max()).execute(), 'r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().logseries(
            handle_array(p), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorLogseries(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(p, chunk_size=chunk_size)
