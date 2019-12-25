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


class TensorRandn(TensorSimpleRandomData, TensorRandomOperandMixin):
    _op_type_ = OperandDef.RAND_RANDN
    _func_name = 'randn'

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_state=state, _size=size, _dtype=dtype, _gpu=gpu, **kw)

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def randn(random_state, *dn, **kw):
    r"""
    Return a sample (or samples) from the "standard normal" distribution.

    If positive, int_like or int-convertible arguments are provided,
    `randn` generates an array of shape ``(d0, d1, ..., dn)``, filled
    with random floats sampled from a univariate "normal" (Gaussian)
    distribution of mean 0 and variance 1 (if any of the :math:`d_i` are
    floats, they are first converted to integers by truncation). A single
    float randomly sampled from the distribution is returned if no
    argument is provided.

    This is a convenience function.  If you want an interface that takes a
    tuple as the first argument, use `numpy.random.standard_normal` instead.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned tensor, should be all positive.
        If no argument is given a single Python float is returned.

    Returns
    -------
    Z : Tensor or float
        A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
        the standard normal distribution, or a single such float if
        no parameters were supplied.

    See Also
    --------
    random.standard_normal : Similar, but takes a tuple as its argument.

    Notes
    -----
    For random samples from :math:`N(\mu, \sigma^2)`, use:

    ``sigma * mt.random.randn(...) + mu``

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.randn().execute()
    2.1923875335537315 #random

    Two-by-four tensor of samples from N(3, 6.25):

    >>> (2.5 * mt.random.randn(2, 4) + 3).execute()
    array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
           [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random
    """
    if len(dn) == 1 and isinstance(dn[0], (tuple,  list)):
        raise TypeError("'tuple' object cannot be interpreted as an integer")
    if 'dtype' not in kw:
        kw['dtype'] = np.dtype('f8')
    chunk_size = kw.pop('chunk_size', None)

    op = TensorRandn(state=random_state.to_numpy(), size=dn, **kw)

    for key in op.extra_params:
        if not key.startswith('_'):
            raise ValueError('randn got unexpected key arguments {0}'.format(key))

    return op(chunk_size=chunk_size)
