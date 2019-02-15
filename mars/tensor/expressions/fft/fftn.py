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


class TensorFFTN(fftop.FFTN, TensorComplexFFTNMixin):
    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorFFTN, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                         _dtype=dtype, **kw)


def fftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform.

    This function computes the *N*-dimensional discrete Fourier Transform over
    any number of axes in an *M*-dimensional tensor by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    mt.fft : Overall view of discrete Fourier transforms, with definitions
        and conventions used.
    ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.
    fft : The one-dimensional FFT, with definitions and conventions used.
    rfftn : The *n*-dimensional FFT of real input.
    fft2 : The two-dimensional FFT.
    fftshift : Shifts zero-frequency terms to centre of tensor

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of all axes, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    See `mt.fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.mgrid[:3, :3, :3][0]
    >>> mt.fft.fftn(a, axes=(1, 2)).execute()
    array([[[  0.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j]],
           [[  9.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j]],
           [[ 18.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j],
            [  0.+0.j,   0.+0.j,   0.+0.j]]])
    >>> mt.fft.fftn(a, (2, 2), axes=(0, 1)).execute()
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    >>> import matplotlib.pyplot as plt
    >>> [X, Y] = mt.meshgrid(2 * mt.pi * mt.arange(200) / 12,
    ...                      2 * mt.pi * mt.arange(200) / 34)
    >>> S = mt.sin(X) + mt.cos(Y) + mt.random.uniform(0, 1, X.shape)
    >>> FS = mt.fft.fftn(S)
    >>> plt.imshow(mt.log(mt.abs(mt.fft.fftshift(FS))**2).execute())
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorFFTN(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
