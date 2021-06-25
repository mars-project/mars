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
from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...serialization.serializables import KeyField, AnyField, Int32Field
from ...utils import has_unknown_shape
from ..core import Tensor, TENSOR_TYPE, TENSOR_CHUNK_TYPE, TensorOrder
from ..utils import broadcast_shape, unify_chunks
from ..operands import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from .ravel import ravel


class TensorRepeat(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.REPEAT

    _input = KeyField('input')
    _repeats = AnyField('repeats')
    _axis = Int32Field('axis')

    def __init__(self, axis=None, dtype=None, sparse=False, **kw):
        super().__init__(_axis=axis, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def repeats(self):
        return self._repeats

    @property
    def axis(self):
        return self._axis

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]
        if len(inputs) > 1:
            self._repeats = self._inputs[1]

    def __call__(self, a, repeats):
        axis = self._axis
        a = astensor(a)
        if axis is None:
            a = ravel(a)

        ax = axis or 0

        if not isinstance(repeats, Integral):
            if not isinstance(repeats, Tensor):
                repeats = np.asarray(repeats)
                if repeats.size == 1:
                    repeats = int(repeats[0])
                    size = repeats * a.shape[axis or 0]
                elif a.shape[ax] == 1:
                    size = repeats = int(repeats.sum())
                else:
                    size = int(repeats.sum())
            else:
                size = np.nan
            if not isinstance(repeats, Integral):
                if repeats.ndim != 1:
                    raise ValueError('repeats should be 1-d tensor')
                broadcast_shape(repeats.shape, a.shape[ax: ax + 1])
        else:
            size = a.shape[axis or 0] * repeats

        shape = a.shape[:ax] + (size,) + a.shape[ax + 1:]
        self.dtype = a.dtype
        self.sparse = a.issparse()

        inputs = [a]
        if isinstance(repeats, Tensor):
            inputs.append(repeats)
        else:
            self._repeats = repeats

        return self.new_tensor(inputs, shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        a = op.input
        repeats = op.repeats
        axis = op.axis
        ax = axis or 0
        out = op.outputs[0]

        if has_unknown_shape(*op.inputs):
            yield

        if isinstance(repeats, TENSOR_TYPE):
            a, repeats = yield from unify_chunks(a, (repeats, (ax,)))

        nsplit = a.nsplits[axis or 0]

        if isinstance(repeats, Integral):
            new_nsplit = []
            for split in nsplit:
                s = max(split // repeats, 1)
                c = split // s
                new_nsplit.extend([s] * c)
                if split % s != 0:
                    new_nsplit.append(split % s)

            a = yield from recursive_tile(
                a.rechunk({ax: new_nsplit}))

        out_chunks = []
        ax_cum_count = np.cumsum((0,) + a.nsplits[ax])
        is_repeats_ndarray = isinstance(repeats, np.ndarray)
        for out_idx in itertools.product(*[range(len(s)) for s in a.nsplits]):
            in_chunk = a.cix[out_idx]
            ax_idx = out_idx[ax]
            if is_repeats_ndarray:
                start = ax_cum_count[ax_idx]
                stop = ax_cum_count[ax_idx + 1]
                rp = repeats[start: stop]
                size = int(rp.sum())
            elif not isinstance(repeats, Integral):
                rp = repeats.cix[ax_idx, ]
                size = np.nan
            else:
                rp = repeats
                size = in_chunk.shape[ax] * rp

            chunk_inputs = [in_chunk]
            if isinstance(rp, TENSOR_CHUNK_TYPE):
                chunk_inputs.append(rp)

            chunk_shape = in_chunk.shape[:ax] + (size,) + in_chunk.shape[ax + 1:]
            chunk_op = op.copy().reset_key()
            if len(chunk_inputs) < 2:
                # repeats is not chunk
                chunk_op._repeats = rp
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk_shape,
                                           index=out_idx, order=out.order)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        a = inputs[0]
        if len(inputs) > 1:
            repeats = inputs[1]
        else:
            repeats = op.repeats

        with device(device_id):
            ctx[op.outputs[0].key] = xp.repeat(a, repeats=repeats, axis=op.axis)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    repeats : int or tensor of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input tensor, and return a flat output tensor.

    Returns
    -------
    repeated_tensor : Tensor
        Output array which has the same shape as `a`, except along
        the given axis.

    See Also
    --------
    tile : Tile a tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.repeat(3, 4).execute()
    array([3, 3, 3, 3])
    >>> x = mt.array([[1,2],[3,4]])
    >>> mt.repeat(x, 2).execute()
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> mt.repeat(x, 3, axis=1).execute()
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> mt.repeat(x, [1, 2], axis=0).execute()
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    op = TensorRepeat(axis=axis)
    return op(a, repeats)
