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
from ..datasource import tensor as astensor
from .core import TensorComplexFFTMixin, validate_fft, TensorStandardFFT


class TensorIFFT(TensorStandardFFT, TensorComplexFFTMixin):
    _op_type_ = OperandDef.IFFT

    def __init__(self, n=None, axis=-1, norm=None, dtype=None, **kw):
        super().__init__(_n=n, _axis=axis, _norm=norm, _dtype=dtype, **kw)


def ifft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.
    For a general description of the algorithm and definitions,
    see `mt.fft`.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``a[0]`` should contain the zero frequency term,
    * ``a[1:n//2]`` should contain the positive-frequency terms,
    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``A[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `numpy.fft` for details.

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT.  If not given, the last
        axis is used.
    norm : {None, "ortho"}, optional
        Normalization mode (see `numpy.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axes` is larger than the last axis of `a`.

    See Also
    --------
    mt.fft : An introduction, with definitions and general explanations.
    fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse
    ifft2 : The two-dimensional inverse FFT.
    ifftn : The n-dimensional inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end.  Even though this is the common
    approach, it might lead to surprising results.  If a different padding is
    desired, it must be performed before calling `ifft`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fft.ifft([0, 4, 0, 0]).execute()
    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j])

    Create and plot a band-limited signal with random phases:

    >>> import matplotlib.pyplot as plt
    >>> t = mt.arange(400)
    >>> n = mt.zeros((400,), dtype=complex)
    >>> n[40:60] = mt.exp(1j*mt.random.uniform(0, 2*mt.pi, (20,)))
    >>> s = mt.fft.ifft(n)
    >>> plt.plot(t.execute(), s.real.execute(), 'b-', t.execute(), s.imag.execute(), 'r--')
    ...
    >>> plt.legend(('real', 'imaginary'))
    ...
    >>> plt.show()

    """
    a = astensor(a)
    validate_fft(a, axis, norm)
    op = TensorIFFT(n=n, axis=axis, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
