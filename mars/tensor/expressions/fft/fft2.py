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
from .core import TensorComplexFFTNMixin, validate_fftn, TensorStandardFFTN


class TensorFFT2(TensorStandardFFTN, TensorComplexFFTNMixin):
    _op_type_ = OperandDef.FFT2

    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorFFT2, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                         _dtype=dtype, **kw)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform

    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
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
    ifft2 : The inverse two-dimensional FFT.
    fft : The one-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For two-dimensional input, swaps first and third quadrants, and second
        and fourth quadrants.

    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.

    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    See `fftn` for details and a plotting example, and `mt.fft` for
    definitions and conventions used.


    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.mgrid[:5, :5][0]
    >>> mt.fft.fft2(a).execute()
    array([[ 50.0 +0.j        ,   0.0 +0.j        ,   0.0 +0.j        ,
              0.0 +0.j        ,   0.0 +0.j        ],
           [-12.5+17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
              0.0 +0.j        ,   0.0 +0.j        ],
           [-12.5 +4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
              0.0 +0.j        ,   0.0 +0.j        ],
           [-12.5 -4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
                0.0 +0.j        ,   0.0 +0.j        ],
           [-12.5-17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
              0.0 +0.j        ,   0.0 +0.j        ]])

    """
    if len(axes) != 2:
        raise ValueError("axes length should be 2")
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorFFT2(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
