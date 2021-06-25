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


class TensorGeometric(TensorDistribution, TensorRandomOperandMixin):
    _input_fields_ = ['_p']
    _op_type_ = OperandDef.RAND_GEOMETRIC

    _fields_ = '_p', '_size'
    _p = AnyField('p')
    _func_name = 'geometric'

    @property
    def p(self):
        return self._p

    def __init__(self, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, dtype=dtype, **kw)

    def __call__(self, p, chunk_size=None):
        return self.new_tensor([p], None, raw_chunk_size=chunk_size)


def geometric(random_state, p, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Draw samples from the geometric distribution.

    Bernoulli trials are experiments with one of two outcomes:
    success or failure (an example of such an experiment is flipping
    a coin).  The geometric distribution models the number of trials
    that must be run in order to achieve success.  It is therefore
    supported on the positive integers, ``k = 1, 2, ...``.

    The probability mass function of the geometric distribution is

    .. math:: f(k) = (1 - p)^{k - 1} p

    where `p` is the probability of success of an individual trial.

    Parameters
    ----------
    p : float or array_like of floats
        The probability of success of an individual trial.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``p`` is a scalar.  Otherwise,
        ``mt.array(p).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized geometric distribution.

    Examples
    --------
    Draw ten thousand values from the geometric distribution,
    with the probability of an individual success equal to 0.35:

    >>> import mars.tensor as mt

    >>> z = mt.random.geometric(p=0.35, size=10000)

    How many trials succeeded after a single run?

    >>> ((z == 1).sum() / 10000.).execute()
    0.34889999999999999 #random
    """
    if dtype is None:
        dtype = np.random.RandomState().geometric(
            handle_array(p), size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorGeometric(seed=seed, size=size, gpu=gpu, dtype=dtype)
    return op(p, chunk_size=chunk_size)
