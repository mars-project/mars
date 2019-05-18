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

from .... import opcodes as OperandDef
from ....serialize import AnyField
from .core import TensorRandomOperandMixin, handle_array, TensorDistribution


class TensorStandardT(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_df', '_size'
    _input_fields_ = ['_df']
    _op_type_ = OperandDef.RAND_STANDARD_T

    _df = AnyField('df')

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super(TensorStandardT, self).__init__(_size=size, _state=state, _dtype=dtype,
                                              _gpu=gpu, **kw)

    @property
    def df(self):
        return self._df

    def __call__(self, df, chunk_size=None):
        return self.new_tensor([df], None, raw_chunk_size=chunk_size)


def standard_t(random_state, df, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a standard Student's t distribution with `df` degrees
    of freedom.

    A special case of the hyperbolic distribution.  As `df` gets
    large, the result resembles that of the standard normal
    distribution (`standard_normal`).

    Parameters
    ----------
    df : float or array_like of floats
        Degrees of freedom, should be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` is a scalar.  Otherwise,
        ``mt.array(df).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized standard Student's t distribution.

    Notes
    -----
    The probability density function for the t distribution is

    .. math:: P(x, df) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{\pi df}
              \Gamma(\frac{df}{2})}\Bigl( 1+\frac{x^2}{df} \Bigr)^{-(df+1)/2}

    The t test is based on an assumption that the data come from a
    Normal distribution. The t test provides a way to test whether
    the sample mean (that is the mean calculated from the data) is
    a good estimate of the true mean.

    The derivation of the t-distribution was first published in
    1908 by William Gosset while working for the Guinness Brewery
    in Dublin. Due to proprietary issues, he had to publish under
    a pseudonym, and so he used the name Student.

    References
    ----------
    .. [1] Dalgaard, Peter, "Introductory Statistics With R",
           Springer, 2002.
    .. [2] Wikipedia, "Student's t-distribution"
           http://en.wikipedia.org/wiki/Student's_t-distribution

    Examples
    --------
    From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
    women in Kj is:

    >>> import mars.tensor as mt

    >>> intake = mt.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \
    ...                    7515, 8230, 8770])

    Does their energy intake deviate systematically from the recommended
    value of 7725 kJ?

    We have 10 degrees of freedom, so is the sample mean within 95% of the
    recommended value?

    >>> s = mt.random.standard_t(10, size=100000)
    >>> mt.mean(intake).execute()
    6753.636363636364
    >>> intake.std(ddof=1).execute()
    1142.1232221373727

    Calculate the t statistic, setting the ddof parameter to the unbiased
    value so the divisor in the standard deviation will be degrees of
    freedom, N-1.

    >>> t = (mt.mean(intake)-7725)/(intake.std(ddof=1)/mt.sqrt(len(intake)))
    >>> import matplotlib.pyplot as plt
    >>> h = plt.hist(s.execute(), bins=100, normed=True)

    For a one-sided t-test, how far out in the distribution does the t
    statistic appear?

    >>> (mt.sum(s<t) / float(len(s))).execute()
    0.0090699999999999999  #random

    So the p-value is about 0.009, which says the null hypothesis has a
    probability of about 99% of being true.
    """
    if dtype is None:
        dtype = np.random.RandomState().standard_t(
            handle_array(df), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorStandardT(size=size, state=random_state._state, gpu=gpu, dtype=dtype)
    return op(df, chunk_size=chunk_size)
