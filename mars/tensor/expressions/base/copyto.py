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

from ....operands import CopyTo
from ....compat import lrange
from ..utils import unify_chunks, broadcast_shape
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor
from .broadcast_to import broadcast_to


class TensorCopyTo(CopyTo, TensorOperandMixin):
    def __init__(self, casting=None, dtype=None, gpu=None, sparse=None, **kw):
        super(TensorCopyTo, self).__init__(_casting=casting, _dtype=dtype,
                                           _gpu=gpu, _sparse=sparse, **kw)

    def check_inputs(self, inputs):
        if not 2 <= len(inputs) <= 3:
            raise ValueError("inputs' length must be 2 or 3")

    def _set_inputs(self, inputs):
        super(TensorCopyTo, self)._set_inputs(inputs)

        self._src = self._inputs[0]
        self._dst = self._inputs[1]
        if len(self._inputs) > 2:
            self._where = self._inputs[2]

    @staticmethod
    def _extract_inputs(inputs):
        if len(inputs) == 2:
            (src, dst), where = inputs, None
        else:
            src, dst, where = inputs
            if where is True:
                where = None
            else:
                where = astensor(where)

        return src, dst, where

    def __call__(self, *inputs):
        from ..core import Tensor

        src, dst, where = self._extract_inputs(inputs)

        if not isinstance(dst, Tensor):
            raise TypeError('dst has to be a Tensor')

        self._dtype = dst.dtype
        self._gpu = dst.op.gpu
        self._sparse = dst.issparse()

        if not np.can_cast(src.dtype, dst.dtype, casting=self.casting):
            raise TypeError('Cannot cast array from {0!r} to {1!r} '
                            'according to the rule {2!s}'.format(
                src.dtype, dst.dtype, self.casting))

        try:
            broadcast_to(src, dst.shape)
        except ValueError:
            raise ValueError('could not broadcast input array '
                             'from shape {0!r} into shape {1!r}'.format(src.shape, dst.shape))
        if where:
            try:
                broadcast_to(where, dst.shape)
            except ValueError:
                raise ValueError('could not broadcast where mask '
                                 'from shape {0!r} into shape {1!r}'.format(src.shape, dst.shape))

        inps = [src, dst]
        if where is not None:
            inps.append(where)
        ret = self.new_tensor(inps, dst.shape)
        dst.data = ret.data

    @classmethod
    def tile(cls, op):
        inputs = unify_chunks(*[(input, lrange(input.ndim)[::-1]) for input in op.inputs])

        chunk_shapes = [t.chunk_shape if hasattr(t, 'chunk_shape') else t
                        for t in inputs]
        out_chunk_shape = broadcast_shape(*chunk_shapes)

        out_chunks = []
        nsplits = [[None] * shape for shape in out_chunk_shape]
        get_index = lambda idx, t: tuple(0 if t.nsplits[i] == (1,) else ix
                                         for i, ix in enumerate(idx))
        for out_idx in itertools.product(*(map(range, out_chunk_shape))):
            in_chunks = [t.cix[get_index(out_idx[-t.ndim:], t)] if t.ndim != 0 else t.chunks[0]
                         for t in inputs]
            out_chunk = op.copy().reset_key().new_chunk(in_chunks, in_chunks[1].shape, index=out_idx)
            out_chunks.append(out_chunk)
            for i, idx, s in zip(itertools.count(0), out_idx, out_chunk.shape):
                nsplits[i][idx] = s

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, chunks=out_chunks, nsplits=nsplits)


def copyto(dst, src, casting='same_kind', where=True):
    """
    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    Parameters
    ----------
    dst : Tensor
        The tensor into which values are copied.
    src : array_like
        The tensor from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean tensor which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.
    """
    op = TensorCopyTo(casting=casting)
    return op(src, dst, where)
