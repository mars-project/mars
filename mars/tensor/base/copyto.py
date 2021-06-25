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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import KeyField, StringField
from ...utils import has_unknown_shape
from ..utils import unify_chunks, broadcast_shape
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from .broadcast_to import broadcast_to
from ..array_utils import as_same_device, device


class TensorCopyTo(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.COPYTO

    _src = KeyField('src')
    _dst = KeyField('dest')
    _casting = StringField('casting')
    _where = KeyField('where')

    def __init__(self, casting=None, **kw):
        super().__init__(_casting=casting, **kw)

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def casting(self):
        return self._casting

    @property
    def where(self):
        return self._where

    def check_inputs(self, inputs):
        if not 2 <= len(inputs) <= 3:
            raise ValueError("inputs' length must be 2 or 3")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)

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

        self.dtype = dst.dtype
        self.gpu = dst.op.gpu
        self.sparse = dst.issparse()

        if not np.can_cast(src.dtype, dst.dtype, casting=self.casting):
            raise TypeError(f'Cannot cast array from {src.dtype!r} to {dst.dtype!r} '
                            f'according to the rule {self.casting!s}')

        try:
            broadcast_to(src, dst.shape)
        except ValueError:
            raise ValueError('could not broadcast input array '
                             f'from shape {src.shape!r} into shape {dst.shape!r}')
        if where:
            try:
                broadcast_to(where, dst.shape)
            except ValueError:
                raise ValueError('could not broadcast where mask '
                                 f'from shape {src.shape!r} into shape {dst.shape!r}')

        inps = [src, dst]
        if where is not None:
            inps.append(where)
        ret = self.new_tensor(inps, dst.shape, order=dst.order)
        dst.data = ret.data

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield
        inputs = yield from unify_chunks(
            *[(input, list(range(input.ndim))[::-1]) for input in op.inputs])
        output = op.outputs[0]

        chunk_shapes = [t.chunk_shape if hasattr(t, 'chunk_shape') else t
                        for t in inputs]
        out_chunk_shape = broadcast_shape(*chunk_shapes)

        out_chunks = []
        nsplits = [[np.nan] * shape for shape in out_chunk_shape]
        get_index = lambda idx, t: tuple(0 if t.nsplits[i] == (1,) else ix
                                         for i, ix in enumerate(idx))
        for out_idx in itertools.product(*(map(range, out_chunk_shape))):
            in_chunks = [t.cix[get_index(out_idx[-t.ndim:], t)] if t.ndim != 0 else t.chunks[0]
                         for t in inputs]
            out_chunk = op.copy().reset_key().new_chunk(
                in_chunks, shape=in_chunks[1].shape, order=output.order, index=out_idx)
            out_chunks.append(out_chunk)
            for i, idx, s in zip(itertools.count(0), out_idx, out_chunk.shape):
                nsplits[i][idx] = s

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, output.shape, order=output.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            dst = inputs[1].copy()
            src = inputs[0]
            where = inputs[2] if len(inputs) > 2 else True

            xp.copyto(dst, src, casting=op.casting, where=where)
            ctx[op.outputs[0].key] = dst


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
