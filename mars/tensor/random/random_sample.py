#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from .core import TensorRandomOperandMixin, TensorSimpleRandomData


class TensorRandomSample(TensorSimpleRandomData, TensorRandomOperandMixin):
    __slots__ = '_size',
    _op_type_ = OperandDef.RAND_RANDOM_SAMPLE
    _func_name = 'random_sample'

    def __init__(self, state=None, size=None, dtype=None,
                 gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_state=state, _size=size, _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, chunk_size):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def random_sample(random_state, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    Results are from the "continuous uniform" distribution over the
    stated interval.  To sample :math:`Unif[a, b), b > a` multiply
    the output of `random_sample` by `(b-a)` and add `a`::

      (b - a) * random_sample() + a

    Parameters
    ----------
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
    out : float or Tensor of floats
        Array of random floats of shape `size` (unless ``size=None``, in which
        case a single float is returned).

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.random_sample().execute()
    0.47108547995356098
    >>> type(mt.random.random_sample().execute())
    <type 'float'>
    >>> mt.random.random_sample((5,)).execute()
    array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

    Three-by-two array of random numbers from [-5, 0):

    >>> (5 * mt.random.random_sample((3, 2)) - 5).execute()
    array([[-3.99149989, -0.52338984],
           [-2.99091858, -0.79479508],
           [-1.23204345, -1.75224494]])
    """
    if dtype is None:
        dtype = np.dtype('f8')
    size = random_state._handle_size(size)
    op = TensorRandomSample(state=random_state.to_numpy(), size=size, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
