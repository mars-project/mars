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


class TensorNegativeBinomial(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_n', '_p', '_size'
    _input_fields_ = ['_n', '_p']
    _op_type_ = OperandDef.RAND_NEGATIVE_BINOMIAL

    _n = AnyField('n')
    _p = AnyField('p')

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorNegativeBinomial, self).__init__(_size=size, _state=state, _dtype=dtype,
                                                     _gpu=gpu, **kw)

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    def __call__(self, n, p, chunk_size=None):
        return self.new_tensor([n, p], None, raw_chunk_size=chunk_size)


def negative_binomial(random_state, n, p, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a negative binomial distribution.

    Samples are drawn from a negative binomial distribution with specified
    parameters, `n` trials and `p` probability of success where `n` is an
    integer > 0 and `p` is in the interval [0, 1].

    Parameters
    ----------
    n : int or array_like of ints
        Parameter of the distribution, > 0. Floats are also accepted,
        but they will be truncated to integers.
    p : float or array_like of floats
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``n`` and ``p`` are both scalars.
        Otherwise, ``np.broadcast(n, p).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized negative binomial distribution,
        where each sample is equal to N, the number of trials it took to
        achieve n - 1 successes, N - (n - 1) failures, and a success on the,
        (N + n)th trial.

    Notes
    -----
    The probability density for the negative binomial distribution is

    .. math:: P(N;n,p) = \binom{N+n-1}{n-1}p^{n}(1-p)^{N},

    where :math:`n-1` is the number of successes, :math:`p` is the
    probability of success, and :math:`N+n-1` is the number of trials.
    The negative binomial distribution gives the probability of n-1
    successes and N failures in N+n-1 trials, and success on the (N+n)th
    trial.

    If one throws a die repeatedly until the third time a "1" appears,
    then the probability distribution of the number of non-"1"s that
    appear before the third "1" is a negative binomial distribution.

    References
    ----------
    .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/NegativeBinomialDistribution.html
    .. [2] Wikipedia, "Negative binomial distribution",
           http://en.wikipedia.org/wiki/Negative_binomial_distribution

    Examples
    --------
    Draw samples from the distribution:

    A real world example. A company drills wild-cat oil
    exploration wells, each with an estimated probability of
    success of 0.1.  What is the probability of having one success
    for each successive well, that is what is the probability of a
    single success after drilling 5 wells, after 6 wells, etc.?

    >>> import mars.tensor as mt

    >>> s = mt.random.negative_binomial(1, 0.1, 100000)
    >>> for i in range(1, 11):
    ...    probability = (mt.sum(s<i) / 100000.).execute()
    ...    print i, "wells drilled, probability of one success =", probability
    """
    if dtype is None:
        dtype = np.random.RandomState().negative_binomial(
            handle_array(n), handle_array(p), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorNegativeBinomial(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(n, p, chunk_size=chunk_size)
