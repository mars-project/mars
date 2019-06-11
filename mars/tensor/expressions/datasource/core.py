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

import itertools

import numpy as np

from ....compat import izip
from ....config import options
from ..utils import normalize_shape, decide_chunk_sizes
from ..core import TensorOperand, TensorOperandMixin


class TensorDataSource(TensorOperand, TensorOperandMixin):
    """
    Tensor data source base class, provide universal tile logic,
    subclass can overwrite tile method.
    """

    __slots__ = ()

    def to_chunk_op(self, *args):
        chunk_shape, _, chunk_size = args
        chunk_op = self.copy().reset_key()
        chunk_op.extra_params = {'size': chunk_shape}  # to make op key different
        return chunk_op

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        chunk_size = tensor.extra_params.raw_chunk_size or options.tensor.chunk_size
        chunk_size = decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_shape, chunk_idx in izip(itertools.product(*chunk_size),
                                           itertools.product(*chunk_size_idxes)):
            chunk_op = op.to_chunk_op(chunk_shape, chunk_idx, chunk_size)
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, chunks=out_chunks, nsplits=chunk_size)


class TensorNoInput(TensorDataSource):
    """
    Tensor operand with no inputs.
    """

    def check_inputs(self, inputs):
        # no inputs
        if inputs and len(inputs) > 0:
            raise ValueError("Tensor data source has no inputs")

    def _new_chunks(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super(TensorNoInput, self)._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        shape = kw.get('shape', None)
        self.extra_params['shape'] = shape  # set shape to make the operand key different
        return super(TensorNoInput, self)._new_tileables(inputs, kws=kws, **kw)

    def __call__(self, shape, chunk_size=None):
        shape = normalize_shape(shape)
        return self.new_tensor(None, shape, raw_chunk_size=chunk_size)


class TensorHasInput(TensorDataSource):
    """
    Tensor operand with a single input.
    """

    @property
    def input(self):
        return self._input

    def check_inputs(self, inputs):
        # no inputs
        if len(inputs) != 1:
            raise ValueError("Tensor can only have 1 input")

    def _set_inputs(self, inputs):
        super(TensorHasInput, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.input.chunks:
            out_chunk = op.copy().reset_key().new_chunk([c], shape=c.shape, index=c.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=out_chunks,
                                  nsplits=op.input.nsplits)

    def __call__(self, a):
        return self.new_tensor([a], a.shape)


class TensorLike(TensorHasInput):
    def _set_inputs(self, inputs):
        super(TensorLike, self)._set_inputs(inputs)
        if self.dtype is None:
            self._dtype = self.input.dtype
        if self.gpu is None:
            self._gpu = self.input.op.gpu

        # FIXME: remove when cupy supports other dtypes
        if self._gpu and self._dtype not in (np.float32, np.float64):
            raise NotImplementedError('Sparse tensor on GPU only supports float32 and float64')
