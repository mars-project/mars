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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import Int32Field
from ..utils import unify_chunks, check_out_param
from ..array_utils import as_same_device, device
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..core import Tensor, TensorOrder


class TensorStack(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.STACK

    _axis = Int32Field('axis')

    def __init__(self, axis=None, dtype=None, sparse=False, **kw):
        super().__init__(_axis=axis, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def axis(self):
        return self._axis

    def __call__(self, tensors, out=None):
        if out is not None and not isinstance(out, Tensor):
            raise TypeError('`out` must be a Tensor, got {0} instead'.format(type(out)))

        shape = tensors[0].shape[:self._axis] + (len(tensors),) + tensors[0].shape[self._axis:]
        tensor_order = TensorOrder.C_ORDER if out is None else out.order
        t = self.new_tensor(tensors, shape, order=tensor_order)

        if out is None:
            return t

        if out.shape != t.shape:
            raise ValueError('Output tensor has wrong dimensionality')
        check_out_param(out, t, 'same_kind')
        out.data = t.data
        return out

    @classmethod
    def tile(cls, op):
        from ..indexing.slice import TensorSlice

        inputs = unify_chunks(*op.inputs)
        output = op.outputs[0]
        axis = op.axis

        output_nsplits = inputs[0].nsplits[:axis] + ((1,) * len(inputs),) + \
            inputs[0].nsplits[axis:]
        output_idxes = itertools.product(*[range(len(nsplit)) for nsplit in output_nsplits])

        out_chunks = []
        for idx in output_idxes:
            input_idx = idx[:axis] + idx[axis + 1:]
            i = idx[axis]
            input_chunk = inputs[i].cix[input_idx]
            slices = [slice(None)] * axis + [np.newaxis] + [slice(None)] * (len(input_idx) - axis)
            shape = input_chunk.shape[:axis] + (1,) + input_chunk.shape[axis:]
            chunk_op = TensorSlice(slices=slices, dtype=op.dtype, sparse=op.sparse)
            out_chunk = chunk_op.new_chunk([input_chunk], shape=shape, index=idx, order=output.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, output.shape,
                                  chunks=out_chunks, nsplits=output_nsplits)

    @classmethod
    def execute(cls, ctx, op):
        raw_inputs = [ctx[c.key] for c in op.inputs]
        is_input_tuple = isinstance(raw_inputs[0], tuple)
        input_tuple_len = len(raw_inputs[0]) if is_input_tuple else 1

        if is_input_tuple:
            # situation that stack is used during tiling, not created by user
            inputs = list(itertools.chain.from_iterable(raw_inputs))
        else:
            inputs = raw_inputs
        # move all the data to the same device
        inputs, device_id, xp = as_same_device(
            inputs, device=op.device, ret_extra=True)
        if is_input_tuple:
            inputs = [inputs[i * input_tuple_len: (i + 1) * input_tuple_len]
                      for i in range(len(raw_inputs))]
        else:
            inputs = [[inp] for inp in inputs]

        axis = op.axis
        out = op.outputs[0]
        with device(device_id):
            rets = []
            for i in range(input_tuple_len):
                ret = xp.stack([inp[i] for inp in inputs], axis=axis)
                # make sure order is identical to out's order
                ret = ret.astype(ret.dtype, order=out.order.value, copy=False)
                rets.append(ret)
            ctx[out.key] = rets if is_input_tuple else rets[0]


def stack(tensors, axis=0, out=None):
    """
    Join a sequence of tensors along a new axis.

    The `axis` parameter specifies the index of the new axis in the dimensions
    of the result. For example, if ``axis=0`` it will be the first dimension
    and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    tensors : sequence of array_like
        Each tensor must have the same shape.
    axis : int, optional
        The axis in the result tensor along which the input tensors are stacked.
    out : Tensor, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.

    Returns
    -------
    stacked : Tensor
        The stacked tensor has one more dimension than the input tensors.

    See Also
    --------
    concatenate : Join a sequence of tensors along an existing axis.
    split : Split tensor into a list of multiple sub-tensors of equal size.
    block : Assemble tensors from blocks.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> arrays = [mt.random.randn(3, 4) for _ in range(10)]
    >>> mt.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> mt.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> mt.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = mt.array([1, 2, 3])
    >>> b = mt.array([2, 3, 4])
    >>> mt.stack((a, b)).execute()
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> mt.stack((a, b), axis=-1).execute()
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    tensors = [astensor(t) for t in tensors]

    if len(set(t.shape for t in tensors)) != 1:
        raise ValueError('all input tensors must have the same shape')

    ndim = len(tensors[0].shape)
    raw_axis = axis
    if axis < 0:
        axis = ndim + axis + 1
    if axis > ndim or axis < 0:
        raise np.AxisError('axis {0} is out of bounds for tensor '
                           'of dimension {1}'.format(raw_axis, ndim))

    dtype = np.result_type(*[t.dtype for t in tensors])
    sparse = all(t.issparse() for t in tensors)

    op = TensorStack(axis=axis, dtype=dtype, sparse=sparse)
    return op(tensors, out=out)
