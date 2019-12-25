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


class TensorRayleigh(TensorDistribution, TensorRandomOperandMixin):
    __slots__ = '_scale', '_size'
    _input_fields_ = ['_scale']
    _op_type_ = OperandDef.RAND_RAYLEIGH

    _scale = AnyField('scale')
    _func_name = 'rayleigh'

    def __init__(self, size=None, state=None, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else dtype
        super().__init__(_size=size, _state=state, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def scale(self):
        return self._scale

    def __call__(self, scale, chunk_size=None):
        return self.new_tensor([scale], None, raw_chunk_size=chunk_size)


def rayleigh(random_state, scale=1.0, size=None, chunk_size=None, gpu=None, dtype=None):
    r"""
    Draw samples from a Rayleigh distribution.

    The :math:`\chi` and Weibull distributions are generalizations of the
    Rayleigh.

    Parameters
    ----------
    scale : float or array_like of floats, optional
        Scale, also equals the mode. Should be >= 0. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``mt.array(scale).size`` samples are drawn.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    dtype : data-type, optional
      Data-type of the returned tensor.

    Returns
    -------
    out : Tensor or scalar
        Drawn samples from the parameterized Rayleigh distribution.

    Notes
    -----
    The probability density function for the Rayleigh distribution is

    .. math:: P(x;scale) = \frac{x}{scale^2}e^{\frac{-x^2}{2 \cdotp scale^2}}

    The Rayleigh distribution would arise, for example, if the East
    and North components of the wind velocity had identical zero-mean
    Gaussian distributions.  Then the wind speed would have a Rayleigh
    distribution.

    References
    ----------
    .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
           http://www.brighton-webs.co.uk/distributions/rayleigh.asp
    .. [2] Wikipedia, "Rayleigh distribution"
           http://en.wikipedia.org/wiki/Rayleigh_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram

    >>> import matplotlib.pyplot as plt
    >>> import mars.tensor as mt

    >>> values = plt.hist(mt.random.rayleigh(3, 100000).execute(), bins=200, normed=True)

    Wave heights tend to follow a Rayleigh distribution. If the mean wave
    height is 1 meter, what fraction of waves are likely to be larger than 3
    meters?

    >>> meanvalue = 1
    >>> modevalue = mt.sqrt(2 / mt.pi) * meanvalue
    >>> s = mt.random.rayleigh(modevalue, 1000000)

    The percentage of waves larger than 3 meters is:

    >>> (100.*mt.sum(s>3)/1000000.).execute()
    0.087300000000000003
    """
    if dtype is None:
        dtype = np.random.RandomState().rayleigh(
            handle_array(scale), size=(0,)).dtype
    size = random_state._handle_size(size)
    op = TensorRayleigh(size=size, state=random_state.to_numpy(), gpu=gpu, dtype=dtype)
    return op(scale, chunk_size=chunk_size)
