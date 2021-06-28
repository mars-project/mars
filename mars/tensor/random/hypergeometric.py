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


class TensorHypergeometric(TensorDistribution, TensorRandomOperandMixin):
    _input_fields_ = ['_ngood', '_nbad', '_nsample']
    _op_type_ = OperandDef.RAND_HYPERGEOMETRIC

    _fields_ = '_ngood', '_nbad', '_nsample', '_size'
    _ngood = AnyField('ngood')
    _nbad = AnyField('nbad')
    _nsample = AnyField('nsample')
    _func_name = 'hypergeometric'

    def __init__(self, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, dtype=dtype, **kw)

    @property
    def ngood(self):
        return self._ngood

    @property
    def nbad(self):
        return self._nbad

    @property
    def nsample(self):
        return self._nsample

    def __call__(self, ngood, nbad, nsample, chunk_size=None):
        return self.new_tensor([ngood, nbad, nsample], None, raw_chunk_size=chunk_size)


def hypergeometric(random_state, ngood, nbad, nsample, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Hypergeometric distribution.

    Samples are drawn from a hypergeometric distribution with specified
    parameters, ngood (ways to make a good selection), nbad (ways to make
    a bad selection), and nsample = number of items sampled, which is less
    than or equal to the sum ngood + nbad.

    Parameters
    ----------
    ngood : int or array_like of ints
        Number of ways to make a good selection.  Must be nonnegative.
    nbad : int or array_like of ints
        Number of ways to make a bad selection.  Must be nonnegative.
    nsample : int or array_like of ints
        Number of items sampled.  Must be at least 1 and at most
        ``ngood + nbad``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``ngood``, ``nbad``, and ``nsample``
        are all scalars.  Otherwise, ``np.broadcast(ngood, nbad, nsample).size``
        samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized hypergeometric distribution.

    See Also
    --------
    scipy.stats.hypergeom : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Hypergeometric distribution is

    .. math:: P(x) = \frac{\binom{m}{n}\binom{N-m}{n-x}}{\binom{N}{n}},

    where :math:`0 \le x \le m` and :math:`n+m-N \le x \le n`

    for P(x) the probability of x successes, n = ngood, m = nbad, and
    N = number of samples.

    Consider an urn with black and white marbles in it, ngood of them
    black and nbad are white. If you draw nsample balls without
    replacement, then the hypergeometric distribution describes the
    distribution of black balls in the drawn sample.

    Note that this distribution is very similar to the binomial
    distribution, except that in this case, samples are drawn without
    replacement, whereas in the Binomial case samples are drawn with
    replacement (or the sample space is infinite). As the sample space
    becomes large, this distribution approaches the binomial.

    References
    ----------
    .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/HypergeometricDistribution.html
    .. [3] Wikipedia, "Hypergeometric distribution",
           http://en.wikipedia.org/wiki/Hypergeometric_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> ngood, nbad, nsamp = 100, 2, 10
    # number of good, number of bad, and number of samples
    >>> s = mt.random.hypergeometric(ngood, nbad, nsamp, 1000)
    >>> hist(s)
    #   note that it is very unlikely to grab both bad items

    Suppose you have an urn with 15 white and 15 black marbles.
    If you pull 15 marbles at random, how likely is it that
    12 or more of them are one color?

    >>> s = mt.random.hypergeometric(15, 15, 15, 100000)
    >>> (mt.sum(s>=12)/100000. + mt.sum(s<=3)/100000.).execute()
    #   answer = 0.003 ... pretty unlikely!
    """
    if dtype is None:
        dtype = np.random.RandomState().hypergeometric(
            handle_array(ngood), handle_array(nbad), handle_array(nsample), size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorHypergeometric(seed=seed, size=size, gpu=gpu, dtype=dtype)
    return op(ngood, nbad, nsample, chunk_size=chunk_size)
