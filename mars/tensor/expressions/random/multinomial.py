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
from ....serialize import ValueType, Int64Field, TupleField
from .core import TensorRandomOperandMixin, TensorDistribution


class TensorMultinomial(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_n', '_pvals', '_size'
    _op_type_ = OperandDef.RAND_MULTINOMIAL

    _n = Int64Field('n')
    _pvals = TupleField('pvals', ValueType.float64)

    def __init__(self, n=None, pvals=None, state=None, size=None,
                 dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorMultinomial, self).__init__(_n=n, _pvals=pvals, _state=state,
                                                _size=size, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def n(self):
        return self._n

    @property
    def pvals(self):
        return self._pvals

    def __call__(self, chunk_size=None):
        if self._size is None:
            shape = (len(self._pvals),)
        else:
            try:
                shape = tuple(self._size) + (len(self._pvals),)
            except TypeError:
                shape = (self._size, len(self._pvals))
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size)


def multinomial(random_state, n, pvals, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalisation of the
    binomial distribution.  Take an experiment with one of ``p``
    possible outcomes.  An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6.  Each sample drawn from the
    distribution represents `n` such experiments.  Its values,
    ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
    outcome was ``i``.

    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        should sum to 1 (however, the last element is always assumed to
        account for the remaining probability, as long as
        ``sum(pvals[:-1]) <= 1)``.
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
    out : Tensor
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Examples
    --------
    Throw a dice 20 times:

    >>> import mars.tensor as mt

    >>> mt.random.multinomial(20, [1/6.]*6, size=1).execute()
    array([[4, 1, 7, 5, 2, 1]])

    It landed 4 times on 1, once on 2, etc.

    Now, throw the dice 20 times, and 20 times again:

    >>> mt.random.multinomial(20, [1/6.]*6, size=2).execute()
    array([[3, 4, 3, 3, 4, 3],
           [2, 4, 3, 4, 0, 7]])

    For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
    we threw 2 times 1, 4 times 2, etc.

    A loaded die is more likely to land on number 6:

    >>> mt.random.multinomial(100, [1/7.]*5 + [2/7.]).execute()
    array([11, 16, 14, 17, 16, 26])

    The probability inputs should be normalized. As an implementation
    detail, the value of the last entry is ignored and assumed to take
    up any leftover probability mass, but this should not be relied on.
    A biased coin which has twice as much weight on one side as on the
    other should be sampled like so:

    >>> mt.random.multinomial(100, [1.0 / 3, 2.0 / 3]).execute()  # RIGHT
    array([38, 62])

    not like:

    >>> mt.random.multinomial(100, [1.0, 2.0]).execute()  # WRONG
    array([100,   0])
    """
    n = int(n)
    pvals = tuple(pvals)
    if dtype is None:
        dtype = np.random.RandomState().multinomial(n, pvals, size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorMultinomial(n=n, pvals=pvals, state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)

