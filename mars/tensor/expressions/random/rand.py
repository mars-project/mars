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


class TensorRand(operands.Rand, TensorRandomOperandMixin):
    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorRand, self).__init__(_state=state, _size=size,
                                         _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, chunks=None):
        return self.new_tensor(None, None, raw_chunks=chunks)


def rand(random_state, *dn, **kw):
    """
    Random values in a given shape.

    Create a tensor of the given shape and populate it with
    random samples from a uniform distributionc
    over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned tensor, should all be positive.
        If no argument is given a single Python float is returned.

    Returns
    -------
    out : Tensor, shape ``(d0, d1, ..., dn)``
        Random values.

    See Also
    --------
    random

    Notes
    -----
    This is a convenience function. If you want an interface that
    takes a shape-tuple as the first argument, refer to
    mt.random.random_sample .

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.rand(3,2).execute()
    array([[ 0.14022471,  0.96360618],  #random
           [ 0.37601032,  0.25528411],  #random
           [ 0.49313049,  0.94909878]]) #random
    """
    if len(dn) == 1 and isinstance(dn[0], (tuple, list)):
        raise TypeError("'tuple' object cannot be interpreted as an integer")
    if 'dtype' not in kw:
        kw['dtype'] = np.dtype('f8')
    chunks = kw.pop('chunks', None)
    op = TensorRand(state=random_state._state, size=dn, **kw)
    return op(chunks=chunks)
