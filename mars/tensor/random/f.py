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
from ...serialize import AnyField
from .core import TensorRandomOperandMixin, handle_array, TensorDistribution


class TensorF(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_dfnum', '_dfden', '_size'
    _input_fields_ = ['_dfnum', '_dfden']
    _op_type_ = OperandDef.RAND_F

    _dfnum = AnyField('dfnum')
    _dfden = AnyField('dfden')
    _func_name = 'f'

    def __init__(self, state=None, size=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_state=state, _size=size, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def dfnum(self):
        return self._dfnum

    @property
    def dfden(self):
        return self._dfden

    def __call__(self, dfnum, dfden, chunk_size=None):
        return self.new_tensor([dfnum, dfden], None, raw_chunk_size=chunk_size)


def f(random_state, dfnum, dfden, size=None, chunk_size=None, gpu=None, dtype=None):
    """
    Draw samples from an F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters should be greater than
    zero.

    The random variate of the F distribution (also known as the
    Fisher distribution) is a continuous probability distribution
    that arises in ANOVA tests, and is the ratio of two chi-square
    variates.

    Parameters
    ----------
    dfnum : float or array_like of floats
        Degrees of freedom in numerator, should be > 0.
    dfden : float or array_like of float
        Degrees of freedom in denominator, should be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``dfnum`` and ``dfden`` are both scalars.
        Otherwise, ``np.broadcast(dfnum, dfden).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Fisher distribution.

    See Also
    --------
    scipy.stats.f : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The F statistic is used to compare in-group variances to between-group
    variances. Calculating the distribution depends on the sampling, and
    so it is a function of the respective degrees of freedom in the
    problem.  The variable `dfnum` is the number of samples minus one, the
    between-groups degrees of freedom, while `dfden` is the within-groups
    degrees of freedom, the sum of the number of samples in each group
    minus the number of groups.

    References
    ----------
    .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
           Fifth Edition, 2002.
    .. [2] Wikipedia, "F-distribution",
           http://en.wikipedia.org/wiki/F-distribution

    Examples
    --------
    An example from Glantz[1], pp 47-40:

    Two groups, children of diabetics (25 people) and children from people
    without diabetes (25 controls). Fasting blood glucose was measured,
    case group had a mean value of 86.1, controls had a mean value of
    82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
    data consistent with the null hypothesis that the parents diabetic
    status does not affect their children's blood glucose levels?
    Calculating the F statistic from the data gives a value of 36.01.

    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> dfnum = 1. # between group degrees of freedom
    >>> dfden = 48. # within groups degrees of freedom
    >>> s = mt.random.f(dfnum, dfden, 1000).execute()

    The lower bound for the top 1% of the samples is :

    >>> sorted(s)[-10]
    7.61988120985

    So there is about a 1% chance that the F statistic will exceed 7.62,
    the measured value is 36, so the null hypothesis is rejected at the 1%
    level.
    """
    if dtype is None:
        dtype = np.random.RandomState().f(
            handle_array(dfnum), handle_array(dfden), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorF(state=random_state.to_numpy(), size=size, gpu=gpu, dtype=dtype)
    return op(dfnum, dfden, chunk_size=chunk_size)
