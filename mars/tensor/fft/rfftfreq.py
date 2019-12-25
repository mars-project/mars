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
from ...serialize import Int32Field, Float64Field
from ..datasource import arange
from ..operands import TensorOperand, TensorOperandMixin
from ..core import TensorOrder


class TensorRFFTFreq(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.RFFTFREQ

    _n = Int32Field('n')
    _d = Float64Field('d')

    def __init__(self, n=None, d=None, dtype=None, gpu=False, **kw):
        super().__init__(_n=n, _d=d, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d

    def __call__(self, chunk_size=None):
        shape = (self.n // 2 + 1,)
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size,
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        t = arange(tensor.shape[0], dtype=op.dtype, gpu=op.gpu,
                   chunk_size=tensor.extra_params.raw_chunk_size)._inplace_tile()
        t = t / (op.n * op.d)
        t._inplace_tile()

        new_op = op.copy()
        return new_op.new_tensors(None, tensor.shape, order=tensor.order,
                                  chunks=t.chunks, nsplits=t.nsplits, **tensor.extra_params)


def rfftfreq(n, d=1.0, gpu=False, chunk_size=None):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float tensor `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    f : Tensor
        Tensor of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> signal = mt.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = mt.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = mt.fft.fftfreq(n, d=1./sample_rate)
    >>> freq.execute()
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = mt.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq.execute()
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    n, d = int(n), float(d)
    op = TensorRFFTFreq(n=n, d=d, dtype=np.dtype(float), gpu=gpu)
    return op(chunk_size=chunk_size)
