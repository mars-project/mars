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
from .core import TensorRealFFTNMixin, validate_fftn, TensorRealFFTN


class TensorRFFTN(TensorRealFFTN, TensorRealFFTNMixin):
    _op_type_ = OperandDef.RFFTN

    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorRFFTN, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                          _dtype=dtype, **kw)


def rfftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real tensor by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    a : array_like
        Input tensor, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : complex Tensor
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `a`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `a`.

    See Also
    --------
    irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT
         of real input.
    fft : The one-dimensional FFT, with definitions and conventions used.
    rfft : The one-dimensional FFT of real input.
    fftn : The n-dimensional FFT.
    rfft2 : The two-dimensional FFT of real input.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`.  The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    See `fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.ones((2, 2, 2))
    >>> mt.fft.rfftn(a).execute()
    array([[[ 8.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j]],
           [[ 0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j]]])

    >>> mt.fft.rfftn(a, axes=(2, 0)).execute()
    array([[[ 4.+0.j,  0.+0.j],
            [ 4.+0.j,  0.+0.j]],
           [[ 0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j]]])

    """
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorRFFTN(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
