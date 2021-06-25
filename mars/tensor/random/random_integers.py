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
from ...serialization.serializables import Int64Field
from ..utils import gen_random_seeds
from .core import TensorRandomOperandMixin, TensorSimpleRandomData


class TensorRandomIntegers(TensorSimpleRandomData, TensorRandomOperandMixin):
    _op_type_ = OperandDef.RAND_RANDOM_INTEGERS

    _fields_ = '_low', '_high', '_size'
    _low = Int64Field('low')
    _high = Int64Field('high')
    _func_name = 'random_integers'

    def __init__(self, size=None, low=None, high=None,
                 dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _low=low, _high=high,
                         dtype=dtype, **kw)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def __call__(self, chunk_size=None):
        return self.new_tensor(None, None, raw_chunk_size=chunk_size)


def random_integers(random_state, low, high=None, size=None, chunk_size=None, gpu=None):
    """
    Random integers of type mt.int between `low` and `high`, inclusive.

    Return random integers of type mt.int from the "discrete uniform"
    distribution in the closed interval [`low`, `high`].  If `high` is
    None (the default), then results are from [1, `low`]. The np.int
    type translates to the C long type used by Python 2 for "short"
    integers and its precision is platform dependent.

    This function has been deprecated. Use randint instead.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int, optional
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : int or Tensor of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    random.randint : Similar to `random_integers`, only for the half-open
        interval [`low`, `high`), and 0 is the lowest value if `high` is
        omitted.

    Notes
    -----
    To sample from N evenly spaced floating-point numbers between a and b,
    use::

      a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.random.random_integers(5).execute()
    4
    >>> type(mt.random.random_integers(5).execute())
    <type 'int'>
    >>> mt.random.random_integers(5, size=(3,2)).execute()
    array([[5, 4],
           [3, 3],
           [4, 5]])

    Choose five random numbers from the set of five evenly-spaced
    numbers between 0 and 2.5, inclusive (*i.e.*, from the set
    :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

    >>> (2.5 * (mt.random.random_integers(5, size=(5,)) - 1) / 4.).execute()
    array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ])

    Roll two six sided dice 1000 times and sum the results:

    >>> d1 = mt.random.random_integers(1, 6, 1000)
    >>> d2 = mt.random.random_integers(1, 6, 1000)
    >>> dsums = d1 + d2

    Display results as a histogram:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(dsums.execute(), 11, normed=True)
    >>> plt.show()
    """
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorRandomIntegers(seed=seed, size=size, dtype=np.dtype(int),
                              low=low, high=high, gpu=gpu)
    return op(chunk_size=chunk_size)
