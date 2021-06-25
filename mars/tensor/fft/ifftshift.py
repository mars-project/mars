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


class TensorIFFTShift(TensorFFTShiftBase, TensorFFTShiftMixin):
    _op_type_ = OperandDef.IFFTSHIFT

    def __init__(self, axes=None, **kw):
        super().__init__(_axes=axes, **kw)

    @classmethod
    def _is_inverse(cls):
        return True

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, x):
        return self.new_tensor([x], x.shape)


def ifftshift(x, axes=None):
    """
    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : Tensor
        The shifted tensor.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> freqs = mt.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs.execute()
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> mt.fft.ifftshift(mt.fft.fftshift(freqs)).execute()
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    x = astensor(x)
    dtype = np.fft.ifftshift(np.empty((1,) * max(1, x.ndim), dtype=x.dtype)).dtype
    axes = TensorIFFTShift._process_axes(x, axes)
    op = TensorIFFTShift(axes=axes, dtype=dtype)
    return op(x)
