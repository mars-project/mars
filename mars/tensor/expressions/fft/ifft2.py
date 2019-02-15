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

from ....operands import fft as fftop
from ..datasource import tensor as astensor
from .core import TensorComplexFFTNMixin, validate_fftn


class TensorIFFT2(fftop.IFFT2, TensorComplexFFTNMixin):
    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorIFFT2, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                          _dtype=dtype, **kw)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an M-dimensional array by means of
    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``
    to within numerical accuracy.  By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e. it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used.  A repeated index in `axes` means the transform over
        that axis is performed multiple times.  A one-element sequence means
        that a one-dimensional FFT is performed.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    mt.fft : Overall view of discrete Fourier transforms, with definitions
         and conventions used.
    fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.
    ifftn : The inverse of the *n*-dimensional FFT.
    fft : The one-dimensional FFT.
    ifft : The one-dimensional inverse FFT.

    Notes
    -----
    `ifft2` is just `ifftn` with a different default for `axes`.

    See `ifftn` for details and a plotting example, and `numpy.fft` for
    definition and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension.  Although this is the common
    approach, it might lead to surprising results.  If another form of zero
    padding is desired, it must be performed before `ifft2` is called.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = 4 * mt.eye(4)
    >>> mt.fft.ifft2(a).execute()
    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])

    """
    if len(axes) != 2:
        raise ValueError("axes length should be 2")
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorIFFT2(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
