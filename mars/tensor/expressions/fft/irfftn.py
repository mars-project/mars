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
from .core import TensorRealIFFTNMixin, validate_fftn, TensorRealFFTN


class TensorIRFFTN(TensorRealFFTN, TensorRealIFFTNMixin):
    _op_type_ = OperandDef.IRFFTN

    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorIRFFTN, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                           _dtype=dtype, **kw)


def irfftn(a, s=None, axes=None, norm=None):
    """
    Compute the inverse of the N-dimensional FFT of real input.

    This function computes the inverse of the N-dimensional discrete
    Fourier Transform for real input over any number of axes in an
    M-dimensional tensor by means of the Fast Fourier Transform (FFT).  In
    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e. as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    a : array_like
        Input tensor.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped.  If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the
        axes specified by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : Tensor
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `a`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given.  In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
        length of the final transformed axis of the input.  To get an odd
        number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    rfftn : The forward n-dimensional FFT of real input,
            of which `ifftn` is the inverse.
    fft : The one-dimensional FFT, with definitions and conventions used.
    irfft : The inverse of the one-dimensional FFT of real input.
    irfft2 : The inverse of the two-dimensional FFT of real input.

    Notes
    -----
    See `fft` for definitions and conventions used.

    See `rfft` for definitions and conventions used for real input.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.zeros((3, 2, 2))
    >>> a[0, 0, 0] = 3 * 2 * 2
    >>> mt.fft.irfftn(a).execute()
    array([[[ 1.,  1.],
            [ 1.,  1.]],
           [[ 1.,  1.],
            [ 1.,  1.]],
           [[ 1.,  1.],
            [ 1.,  1.]]])

    """
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorIRFFTN(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.float_))
    return op(a)
