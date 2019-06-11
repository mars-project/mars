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

from .... import opcodes as OperandDef
from ....serialize import Int32Field
from ..utils import unify_chunks
from ..core import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor


class TensorStack(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.STACK

    _axis = Int32Field('axis')

    def __init__(self, axis=None, dtype=None, sparse=False, **kw):
        super(TensorStack, self).__init__(_axis=axis, _dtype=dtype,
                                          _sparse=sparse, **kw)

    @property
    def axis(self):
        return self._axis

    def __call__(self, tensors):
        shape = tensors[0].shape[:self._axis] + (len(tensors),) + tensors[0].shape[self._axis:]
        return self.new_tensor(tensors, shape)

    @classmethod
    def tile(cls, op):
        from ..indexing.slice import TensorSlice

        inputs = unify_chunks(*op.inputs)
        axis = op.axis

        output_nsplits = inputs[0].nsplits[:axis] + ((1,) * len(inputs),) + \
                         inputs[0].nsplits[axis:]
        output_idxes = itertools.product(*[range(len(nsplit)) for nsplit in output_nsplits])

        out_chunks = []
        for idx in output_idxes:
            input_idx = idx[:axis] + idx[axis+1:]
            i = idx[axis]
            input_chunk = inputs[i].cix[input_idx]
            slices = [slice(None)] * axis + [np.newaxis] + [slice(None)] * (len(input_idx) - axis)
            shape = input_chunk.shape[:axis] + (1,) + input_chunk.shape[axis:]
            chunk_op = TensorSlice(slices=slices, dtype=op.dtype, sparse=op.sparse)
            out_chunk = chunk_op.new_chunk([input_chunk], shape=shape, index=idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=output_nsplits)


def stack(tensors, axis=0):
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
    return op(tensors)
