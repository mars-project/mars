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
from .core import TensorRealFFTNMixin, validate_fftn, TensorRealFFTN


class TensorRFFT2(TensorRealFFTN, TensorRealFFTNMixin):
    _op_type_ = OperandDef.RFFT2

    def __init__(self, shape=None, axes=None, norm=None, dtype=None, **kw):
        super().__init__(_shape=shape, _axes=axes, _norm=norm, _dtype=dtype, **kw)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional FFT of a real tensor.

    Parameters
    ----------
    a : array_like
        Input tensor, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {None, "ortho"}, optional
        Normalization mode (see `mt.fft`). Default is None.

    Returns
    -------
    out : Tensor
        The result of the real 2-D FFT.

    See Also
    --------
    rfftn : Compute the N-dimensional discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    """
    if len(axes) != 2:
        raise ValueError("axes length should be 2")
    a = astensor(a)
    axes = validate_fftn(a, s=s, axes=axes, norm=norm)
    op = TensorRFFT2(shape=s, axes=axes, norm=norm, dtype=np.dtype(np.complex_))
    return op(a)
