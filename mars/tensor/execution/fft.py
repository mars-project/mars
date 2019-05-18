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
try:
    import scipy.fftpack as scifft
except ImportError:  # pragma: no cover
    scifft = None

from ..expressions import fft as fftop
from .array import get_array_module


def _get_fft_func(op, xp):
    fun_name = type(op).__name__.lower()[6:]  # all op starts with tensor
    if type(op) in (fftop.TensorFFT, fftop.TensorIFFT, fftop.TensorFFT2, fftop.TensorIFFT2,
                    fftop.TensorFFTN, fftop.TensorIFFTN):
        if xp is np and scifft and op.norm is None:
            def f(*args, **kwargs):
                kwargs.pop('norm', None)
                if 's' in kwargs:
                    kwargs['shape'] = kwargs.pop('s', None)
                return getattr(scifft, fun_name)(*args, **kwargs)
            return f
        else:
            return getattr(xp.fft, fun_name)
    else:
        return getattr(xp.fft, fun_name)


def _fft(ctx, chunk):
    a = ctx[chunk.inputs[0].key]
    xp = get_array_module(a)
    fun = _get_fft_func(chunk.op, xp)
    res = fun(a, n=chunk.op.n, axis=chunk.op.axis, norm=chunk.op.norm)
    if res.dtype != chunk.op.dtype:
        res = res.astype(chunk.op.dtype)
    ctx[chunk.key] = res


def _fftn(ctx, chunk):
    a = ctx[chunk.inputs[0].key]
    xp = get_array_module(a)
    fun = _get_fft_func(chunk.op, xp)
    res = fun(a, s=chunk.op.shape, axes=chunk.op.axes, norm=chunk.op.norm)
    if res.dtype != chunk.op.dtype:
        res = res.astype(chunk.op.dtype)
    ctx[chunk.key] = res


def _fftfreq_chunk(ctx, chunk):
    n, d = chunk.op.n, chunk.op.d
    x = ctx[chunk.inputs[0].key].copy()
    x[x >= (n + 1) // 2] -= n
    x /= n * d
    ctx[chunk.key] = x


def register_fft_handler():
    from ..expressions.fft import core as fftop
    from ..expressions.fft.fftfreq import TensorFFTFreqChunk
    from ...executor import register

    register(fftop.TensorStandardFFT, _fft)
    register(fftop.TensorStandardFFTN, _fftn)
    register(fftop.TensorRealFFT, _fft)
    register(fftop.TensorRealFFTN, _fftn)
    register(fftop.TensorHermitianFFT, _fft)
    register(TensorFFTFreqChunk, _fftfreq_chunk)
