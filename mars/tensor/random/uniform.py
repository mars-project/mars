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


class TensorUniform(TensorDistribution, TensorRandomOperandMixin):
    _input_fields_ = ['_low', '_high']
    _op_type_ = OperandDef.RAND_UNIFORM

    _fields_ = '_low', '_high', '_size'
    _low = AnyField('low')
    _high = AnyField('high')
    _func_name = 'uniform'

    def __init__(self, size=None, state=None, dtype=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _state=state, dtype=dtype, **kw)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def __call__(self, low, high, chunk_size=None):
        return self.new_tensor([low, high], None, raw_chunk_size=chunk_size)


def uniform(random_state, low=0.0, high=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float or array_like of floats, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float or array_like of floats
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``low`` and ``high`` are both scalars.
        Otherwise, ``mt.broadcast(low, high).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    randint : Discrete uniform distribution, yielding integers.
    random_integers : Discrete uniform distribution over the closed
                      interval ``[low, high]``.
    random_sample : Floats uniformly distributed over ``[0, 1)``.
    random : Alias for `random_sample`.
    rand : Convenience function that accepts dimensions as input, e.g.,
           ``rand(2,2)`` would generate a 2-by-2 array of floats,
           uniformly distributed over ``[0, 1)``.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition.

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> s = mt.random.uniform(-1,0,1000)

    All values are within the given interval:

    >>> mt.all(s >= -1).execute()
    True
    >>> mt.all(s < 0).execute()
    True

    Display the histogram of the samples, along with the
    probability density function:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(s.execute(), 15, normed=True)
    >>> plt.plot(bins, mt.ones_like(bins).execute(), linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().uniform(
            handle_array(low), handle_array(high), size=(0,)).dtype
    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorUniform(size=size, seed=seed, gpu=gpu, dtype=dtype)
    return op(low, high, chunk_size=chunk_size)
