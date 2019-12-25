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


class TensorWald(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_mean', '_scale', '_size'
    _input_fields_ = ['_mean', '_scale']
    _op_type_ = OperandDef.RAND_WALD

    _mean = AnyField('mean')
    _scale = AnyField('scale')
    _func_name = 'wald'

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _state=state, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._scale

    def __call__(self, mean, scale, chunk_size=None):
        return self.new_tensor([mean, scale], None, raw_chunk_size=chunk_size)


def wald(random_state, mean, scale, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Wald, or inverse Gaussian, distribution.

    As the scale approaches infinity, the distribution becomes more like a
    Gaussian. Some references claim that the Wald is an inverse Gaussian
    with mean equal to 1, but this is by no means universal.

    The inverse Gaussian distribution was first studied in relationship to
    Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
    because there is an inverse relationship between the time to cover a
    unit distance and distance covered in unit time.

    Parameters
    ----------
    mean : float or array_like of floats
        Distribution mean, should be > 0.
    scale : float or array_like of floats
        Scale parameter, should be >= 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(mean, scale).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Wald distribution.

    Notes
    -----
    The probability density function for the Wald distribution is

    .. math:: P(x;mean,scale) = \sqrt{\frac{scale}{2\pi x^3}}e^
                                \frac{-scale(x-mean)^2}{2\cdotp mean^2x}

    As noted above the inverse Gaussian distribution first arise
    from attempts to model Brownian motion. It is also a
    competitor to the Weibull for use in reliability modeling and
    modeling stock returns and interest rate processes.

    References
    ----------
    .. [1] Brighton Webs Ltd., Wald Distribution,
           http://www.brighton-webs.co.uk/distributions/wald.asp
    .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
           Distribution: Theory : Methodology, and Applications", CRC Press,
           1988.
    .. [3] Wikipedia, "Wald distribution"
           http://en.wikipedia.org/wiki/Wald_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram:

    >>> import matplotlib.pyplot as plt
    >>> import mars.tensor as mt
    >>> h = plt.hist(mt.random.wald(3, 2, 100000).execute(), bins=200, normed=True)
    >>> plt.show()
    """
    if dtype is None:
        dtype = np.random.RandomState().wald(
            handle_array(mean), handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorWald(size=size, state=random_state.to_numpy(), gpu=gpu, dtype=dtype)
    return op(mean, scale, chunk_size=chunk_size)
