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


class TensorPareto(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_a', '_size'
    _input_fields_ = ['_a']
    _op_type_ = OperandDef.RAND_PARETO

    _a = AnyField('a')
    _func_name = 'pareto'

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _state=state, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def a(self):
        return self._a

    def __call__(self, a, chunk_size=None):
        return self.new_tensor([a], None, raw_chunk_size=chunk_size)


def pareto(random_state, a, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Pareto II or Lomax distribution with
    specified shape.

    The Lomax or Pareto II distribution is a shifted Pareto
    distribution. The classical Pareto distribution can be
    obtained from the Lomax distribution by adding 1 and
    multiplying by the scale parameter ``m`` (see Notes).  The
    smallest value of the Lomax distribution is zero while for the
    classical Pareto distribution it is ``mu``, where the standard
    Pareto distribution has location ``mu = 1``.  Lomax can also
    be considered as a simplified version of the Generalized
    Pareto distribution (available in SciPy), with the scale set
    to one and the location set to zero.

    The Pareto distribution must be greater than zero, and is
    unbounded above.  It is also known as the "80-20 rule".  In
    this distribution, 80 percent of the weights are in the lowest
    20 percent of the range, while the other 20 percent fill the
    remaining 80 percent of the range.

    Parameters
    ----------
    a : float or array_like of floats
        Shape of the distribution. Should be greater than zero.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.  Otherwise,
        ``mt.array(a).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Pareto distribution.

    See Also
    --------
    scipy.stats.lomax : probability density function, distribution or
        cumulative density function, etc.
    scipy.stats.genpareto : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Pareto distribution is

    .. math:: p(x) = \frac{am^a}{x^{a+1}}

    where :math:`a` is the shape and :math:`m` the scale.

    The Pareto distribution, named after the Italian economist
    Vilfredo Pareto, is a power law probability distribution
    useful in many real world problems.  Outside the field of
    economics it is generally referred to as the Bradford
    distribution. Pareto developed the distribution to describe
    the distribution of wealth in an economy.  It has also found
    use in insurance, web page access statistics, oil field sizes,
    and many other problems, including the download frequency for
    projects in Sourceforge [1]_.  It is one of the so-called
    "fat-tailed" distributions.


    References
    ----------
    .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
           Sourceforge projects.
    .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
    .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
           Values, Birkhauser Verlag, Basel, pp 23-30.
    .. [4] Wikipedia, "Pareto distribution",
           http://en.wikipedia.org/wiki/Pareto_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> a, m = 3., 2.  # shape and mode
    >>> s = (mt.random.pareto(a, 1000) + 1) * m

    Display the histogram of the samples, along with the probability
    density function:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, _ = plt.hist(s.execute(), 100, normed=True)
    >>> fit = a*m**a / bins**(a+1)
    >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().pareto(
            handle_array(a), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorPareto(size=size, state=random_state.to_numpy(), gpu=gpu, dtype=dtype)
    return op(a, chunk_size=chunk_size)
