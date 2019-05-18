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


class TensorBeta(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_a', '_b', '_size'
    _input_fields_ = ['_a', '_b']
    _op_type_ = OperandDef.RAND_BETA

    _a = AnyField('a')
    _b = AnyField('b')

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorBeta, self).__init__(_state=state, _size=size,
                                         _dtype=dtype, _gpu=gpu, **kw)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def __call__(self, a, b, chunk_size=None):
        return self.new_tensor([a, b], None, raw_chunk_size=chunk_size)


def beta(random_state, a, b, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Beta distribution.

    The Beta distribution is a special case of the Dirichlet distribution,
    and is related to the Gamma distribution.  It has the probability
    distribution function

    .. math:: f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                                                     (1 - x)^{\beta - 1},

    where the normalisation, B, is the beta function,

    .. math:: B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                                 (1 - t)^{\beta - 1} dt.

    It is often seen in Bayesian inference and order statistics.

    Parameters
    ----------
    a : float or array_like of floats
        Alpha, non-negative.
    b : float or array_like of floats
        Beta, non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.
        Otherwise, ``mt.broadcast(a, b).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized beta distribution.
    """
    if dtype is None:
        dtype = np.random.RandomState().beta(
            handle_array(a), handle_array(b), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorBeta(state=random_state._state, size=size, gpu=gpu, dtype=dtype)
    return op(a, b, chunk_size=chunk_size)
