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


class TensorZipf(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_a', '_size'
    _input_fields_ = ['_a']
    _op_type_ = OperandDef.RAND_ZIPF

    _a = AnyField('a')
    _func_name = 'zipf'

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _state=state, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def a(self):
        return self._a

    def __call__(self, a, chunk_size=None):
        return self.new_tensor([a], None, raw_chunk_size=chunk_size)


def zipf(random_state, a, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Zipf distribution.

    Samples are drawn from a Zipf distribution with specified parameter
    `a` > 1.

    The Zipf distribution (also known as the zeta distribution) is a
    continuous probability distribution that satisfies Zipf's law: the
    frequency of an item is inversely proportional to its rank in a
    frequency table.

    Parameters
    ----------
    a : float or array_like of floats
        Distribution parameter. Should be greater than 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar. Otherwise,
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
        Drawn samples from the parameterized Zipf distribution.

    See Also
    --------
    scipy.stats.zipf : probability density function, distribution, or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Zipf distribution is

    .. math:: p(x) = \frac{x^{-a}}{\zeta(a)},

    where :math:`\zeta` is the Riemann Zeta function.

    It is named for the American linguist George Kingsley Zipf, who noted
    that the frequency of any word in a sample of a language is inversely
    proportional to its rank in the frequency table.

    References
    ----------
    .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
           Frequency in Language," Cambridge, MA: Harvard Univ. Press,
           1932.

    Examples
    --------
    Draw samples from the distribution:

    >>> import mars.tensor as mt

    >>> a = 2. # parameter
    >>> s = mt.random.zipf(a, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import special

    Truncate s values at 50 so plot is interesting:

    >>> count, bins, ignored = plt.hist(s[s<50].execute(), 50, normed=True)
    >>> x = mt.arange(1., 50.)
    >>> y = x**(-a) / special.zetac(a)
    >>> plt.plot(x.execute(), (y/mt.max(y)).execute(), linewidth=2, color='r')
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().zipf(
            handle_array(a), size=(0,)).dtype

    size = random_state._handle_size(size)
    op = TensorZipf(size=size, state=random_state.to_numpy(), gpu=gpu, dtype=dtype)
    return op(a, chunk_size=chunk_size)
