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
from .core import TensorRealIFFTNMixin, validate_fftn


class TensorIRFFT2(fftop.IRFFT2, TensorRealIFFTNMixin):
    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super(TensorIRFFT2, self).__init__(_shape=shape, _axes=axes, _norm=norm,
                                           _dtype=dtype, **kw)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse FFT of a real array.

    Parameters
    ----------
    a : array_like
        The input tensor
    s : sequence of ints, optional
        Shape of the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : Tensor
        The result of the inverse real 2-D FFT.

    See Also
    --------
    irfftn : Compute the inverse of the N-dimensional FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    """
    if len(axes) != 2:
        raise ValueError("axes length should be 2")
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorIRFFT2(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.float_))
    return op(a)
