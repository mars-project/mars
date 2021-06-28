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
from .core import TensorComplexFFTMixin, validate_fft, TensorStandardFFT


class TensorFFT(TensorStandardFFT, TensorComplexFFTMixin):
    _op_type_ = OperandDef.FFT

    def __init__(self, n=None, axis=-1, norm=None, **kw):
        super().__init__(_n=n, _axis=axis, _norm=norm, **kw)


def fft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [CT].

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        if `axes` is larger than the last axis of `a`.

    See Also
    --------
    mt.fft : for definition of the DFT and conventions used.
    ifft : The inverse of `fft`.
    fft2 : The two-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    rfftn : The *n*-dimensional FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms.  The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

    The DFT is defined, with the conventions used in this implementation, in
    the documentation for the `numpy.fft` module.

    References
    ----------
    .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
            machine calculation of complex Fourier series," *Math. Comput.*
            19: 297-301.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fft.fft(mt.exp(2j * mt.pi * mt.arange(8) / 8)).execute()
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-6.89018570e-16j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+0.00000000e+00j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+1.99159850e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+0.00000000e+00j])

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part, as described in
    the `numpy.fft` documentation:

    >>> import matplotlib.pyplot as plt
    >>> t = mt.arange(256)
    >>> sp = mt.fft.fft(mt.sin(t))
    >>> freq = mt.fft.fftfreq(t.shape[-1])
    >>> plt.plot(freq.execute(), sp.real.execute(), freq.execute(), sp.imag.execute())
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    """
    a = astensor(a)
    validate_fft(a, axis, norm)
    op = TensorFFT(n=n, axis=axis, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
