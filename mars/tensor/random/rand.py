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
from ..utils import gen_random_seeds
from .core import TensorRandomOperandMixin, TensorSimpleRandomData


class TensorRand(TensorSimpleRandomData, TensorRandomOperandMixin):
    _op_type_ = OperandDef.RAND_RAND
    _func_name = 'rand'

    def __init__(self, size=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, dtype=dtype, **kw)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


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

    >>> mt.random.rand(3, 2).execute()
    array([[ 0.14022471,  0.96360618],  #random
           [ 0.37601032,  0.25528411],  #random
           [ 0.49313049,  0.94909878]]) #random
    """
    if len(dn) == 1 and isinstance(dn[0], (tuple, list)):
        raise TypeError("'tuple' object cannot be interpreted as an integer")
    if 'dtype' not in kw:
        kw['dtype'] = np.dtype('f8')
    chunk_size = kw.pop('chunk_size', None)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorRand(seed=seed, size=dn, **kw)

    for key in op.extra_params:
        if not key.startswith('_'):
            raise ValueError(f'rand got unexpected key arguments {key}')

    return op(chunk_size=chunk_size)
