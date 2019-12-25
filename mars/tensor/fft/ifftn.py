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
from .core import TensorComplexFFTNMixin, validate_fftn, TensorStandardFFTN


class TensorIFFTN(TensorStandardFFTN, TensorComplexFFTNMixin):
    _op_type_ = OperandDef.IFFTN

    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super().__init__(_shape=shape, _axes=axes, _norm=norm, _dtype=dtype, **kw)


def ifftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform over any number of axes in an M-dimensional tensor by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(a)) == a`` to within numerical accuracy.
    For a description of the definitions and conventions used, see `mt.fft`.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e. it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    a : array_like
        Input tensor, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
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
    fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.
    ifft : The one-dimensional inverse FFT.
    ifft2 : The two-dimensional inverse FFT.
    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
        of tensor.

    Notes
    -----
    See `mt.fft` for definitions and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension.  Although this is the common
    approach, it might lead to surprising results.  If another form of zero
    padding is desired, it must be performed before `ifftn` is called.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.eye(4)
    >>> mt.fft.ifftn(mt.fft.fftn(a, axes=(0,)), axes=(1,)).execute()
    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])


    Create and plot an image with band-limited frequency content:

    >>> import matplotlib.pyplot as plt
    >>> n = mt.zeros((200,200), dtype=complex)
    >>> n[60:80, 20:40] = mt.exp(1j*mt.random.uniform(0, 2*mt.pi, (20, 20)))
    >>> im = mt.fft.ifftn(n).real
    >>> plt.imshow(im.execute())
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorIFFTN(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
