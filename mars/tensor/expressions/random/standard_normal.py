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

from .... import operands
from .core import TensorRandomOperandMixin


class TensorStandardNormal(operands.StandardNormal, TensorRandomOperandMixin):
    __slots__ = '_size',

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorStandardNormal, self).__init__(_size=size, _state=state, _dtype=dtype,
                                                   _gpu=gpu, **kw)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def standard_normal(random_state, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Draw samples from a standard Normal distribution (mean=0, stdev=1).

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
    out : float or Tensor
        Drawn samples.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> s = mt.random.standard_normal(8000)
    >>> s.execute()
    array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
           -0.38672696, -0.4685006 ])                               #random
    >>> s.shape
    (8000,)
    >>> s = mt.random.standard_normal(size=(3, 4, 2))
    >>> s.shape
    (3, 4, 2)
    """
    if dtype is None:
        dtype = np.random.RandomState().standard_normal(size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorStandardNormal(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(chunk_size=chunk_size)
