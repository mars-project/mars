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
from ..datasource import tensor as astensor
from .core import TensorFFTMixin, validate_fft, TensorHermitianFFT


class TensorIHFFT(TensorHermitianFFT, TensorFFTMixin):
    _op_type_ = OperandDef.IHFFT

    def __init__(self, n=None, axis=-1, norm=None, dtype=None, **kw):
        super(TensorIHFFT, self).__init__(_n=n, _axis=axis, _norm=norm,
                                          _dtype=dtype, **kw)

    @classmethod
    def _get_shape(cls, op, shape):
        new_shape = list(shape)
        shape = op.n if op.n is not None else shape[op.axis]
        if shape % 2 == 0:
            shape = (shape // 2) + 1
        else:
            shape = (shape + 1) // 2
        new_shape[op.axis] = shape
        return tuple(new_shape)


def ihfft(a, n=None, axis=-1, norm=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    a : array_like
        Input tensor.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped.  If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So here it's `hfft` for
    which you must supply the length of the result if it is to be odd:

    * even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> spectrum = mt.array([ 15, -4, 0, -1, 0, -4])
    >>> mt.fft.ifft(spectrum).execute()
    array([ 1.+0.j,  2.-0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.-0.j])
    >>> mt.fft.ihfft(spectrum).execute()
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j])

    """
    a = astensor(a)
    validate_fft(a, axis=axis, norm=norm)
    op = TensorIHFFT(n=n, axis=axis, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
