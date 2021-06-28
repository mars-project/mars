#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
from ...core import NotSupportTile, recursive_tile
from ...serialization.serializables import Int32Field, Float64Field, KeyField
from ..operands import TensorOperand, TensorHasInput, TensorOperandMixin
from ..datasource import arange
from ..core import TensorOrder


class TensorFFTFreq(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.FFTFREQ

    _n = Int32Field('n')
    _d = Float64Field('d')

    def __init__(self, n=None, d=None, **kw):
        super().__init__(_n=n, _d=d, **kw)

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d

    def __call__(self, chunk_size=None):
        shape = (self.n,)
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size,
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        in_tensor = yield from recursive_tile(
            arange(op.n, gpu=op.gpu, dtype=op.dtype,
                   chunks=tensor.extra_params.raw_chunk_size))

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = TensorFFTFreqChunk(n=op.n, d=op.d, dtype=op.dtype)
            out_chunk = chunk_op.new_chunk([c], shape=c.shape,
                                           index=c.index, order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=in_tensor.nsplits,
                                  **tensor.extra_params)


class TensorFFTFreqChunk(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.FFTFREQ_CHUNK

    _input = KeyField('input')
    _n = Int32Field('n')
    _d = Float64Field('d')

    def __init__(self, n=None, d=None, dtype=None, **kw):
        super().__init__(_n=n, _d=d, _dtype=dtype, **kw)

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('FFTFreqChunk is a chunk operand which does not support tile')

    @classmethod
    def execute(cls, ctx, op):
        n, d = op.n, op.d
        x = ctx[op.inputs[0].key].copy()
        x[x >= (n + 1) // 2] -= n
        x /= n * d
        ctx[op.outputs[0].key] = x


def fftfreq(n, d=1.0, gpu=False, chunk_size=None):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float tensor `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

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
        Array of length `n` containing the sample frequencies.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> signal = mt.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = mt.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = mt.fft.fftfreq(n, d=timestep)
    >>> freq.execute()
    array([ 0.  ,  1.25,  2.5 ,  3.75, -5.  , -3.75, -2.5 , -1.25])

    """
    n, d = int(n), float(d)
    op = TensorFFTFreq(n=n, d=d, dtype=np.dtype(float), gpu=gpu)
    return op(chunk_size)
