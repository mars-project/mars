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
from ..datasource import tensor as astensor
from .core import TensorFFTShiftMixin, TensorFFTShiftBase


class TensorFFTShift(TensorFFTShiftBase, TensorFFTShiftMixin):
    _op_type_ = OperandDef.FFTSHIFT

    def __init__(self, axes=None, **kw):
        super().__init__(_axes=axes, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, x):
        return self.new_tensor([x], x.shape)


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input tensor.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> freqs = mt.fft.fftfreq(10, 0.1)
    >>> freqs.execute()
    array([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    >>> mt.fft.fftshift(freqs).execute()
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = mt.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs.execute()
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> mt.fft.fftshift(freqs, axes=(1,)).execute()
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    x = astensor(x)
    dtype = np.fft.fftshift(np.empty((1,) * max(1, x.ndim), dtype=x.dtype)).dtype
    axes = TensorFFTShift._process_axes(x, axes)
    op = TensorFFTShift(axes=axes, dtype=dtype)
    return op(x)
