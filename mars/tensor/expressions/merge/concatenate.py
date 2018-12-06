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

from ....operands import Concatenate
from ..utils import validate_axis, unify_chunks
from ..datasource import tensor as astensor
from ..core import TensorOperandMixin


class TensorConcatenate(Concatenate, TensorOperandMixin):
    def __init__(self, axis=None, dtype=None, sparse=False, **kw):
        super(TensorConcatenate, self).__init__(_axis=axis, _dtype=dtype,
                                                _sparse=sparse, **kw)

    def __call__(self, tensors):
        if len(set(t.ndim for t in tensors)) != 1:
            raise ValueError('all the input tensors must have same number of dimensions')

        axis = self._axis
        shapes = [t.shape[:axis] + t.shape[axis+1:] for t in tensors]
        if len(set(shapes)) != 1:
            raise ValueError('all the input tensor dimensions '
                             'except for the concatenation axis must match exactly')

        shape = [0 if i == axis else tensors[0].shape[i] for i in range(tensors[0].ndim)]
        shape[axis] = sum(t.shape[axis] for t in tensors)

        if any(np.isnan(s) for i, s in enumerate(shape) if i != axis):
            raise ValueError('cannot concatenate tensor with unknown shape')

        return self.new_tensor(tensors, shape=tuple(shape))

    @classmethod
    def tile(cls, op):
        from ..indexing.slice import TensorSlice

        inputs = op.inputs
        axis = op.axis

        c = itertools.count(inputs[0].ndim)
        tensor_axes = [(t, tuple(i if i != axis else next(c) for i in range(t.ndim)))
                       for t in inputs]
        inputs = unify_chunks(*tensor_axes)

        out_chunk_shape = [0 if i == axis else inputs[0].chunk_shape[i]
                           for i in range(inputs[0].ndim)]
        out_chunk_shape[axis] = sum(t.chunk_shape[axis] for t in inputs)
        out_nsplits = [None if i == axis else inputs[0].nsplits[i]
                       for i in range(inputs[0].ndim)]
        out_nsplits[axis] = tuple(itertools.chain(*[t.nsplits[axis] for t in inputs]))

        out_chunks = []
        axis_cum_chunk_shape = np.cumsum([t.chunk_shape[axis] for t in inputs])
        for out_idx in itertools.product(*[range(s) for s in out_chunk_shape]):
            axis_index = np.searchsorted(axis_cum_chunk_shape, out_idx[axis], side='right')
            t = inputs[axis_index]
            axis_inner_index = out_idx[axis] - \
                (0 if axis_index < 1 else axis_cum_chunk_shape[axis_index - 1])
            idx = out_idx[:axis] + (axis_inner_index,) + out_idx[axis+1:]
            in_chunk = t.cix[idx]
            if idx == out_idx:
                # if index is the same, just use the input chunk
                out_chunks.append(in_chunk)
            else:
                chunk_op = TensorSlice(slices=[slice(None) for _ in range(in_chunk.ndim)],
                                       dtype=in_chunk.dtype, sparse=in_chunk.op.sparse)
                out_chunk = chunk_op.new_chunk([in_chunk], in_chunk.shape, index=out_idx)

                out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  nsplits=out_nsplits, chunks=out_chunks)


def concatenate(tensors, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The tensors must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the tensors will be joined.  Default is 0.

    Returns
    -------
    res : Tensor
        The concatenated tensor.

    See Also
    --------
    array_split : Split a tensor into multiple sub-arrays of equal or
                  near-equal size.
    split : Split tensor into a list of multiple sub-tensors of equal size.
    hsplit : Split tensor into multiple sub-tensors horizontally (column wise)
    vsplit : Split tensor into multiple sub-tensors vertically (row wise)
    dsplit : Split tensor into multiple sub-tensors along the 3rd axis (depth).
    stack : Stack a sequence of tensors along a new axis.
    hstack : Stack tensors in sequence horizontally (column wise)
    vstack : Stack tensors in sequence vertically (row wise)
    dstack : Stack tensors in sequence depth wise (along third dimension)

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4]])
    >>> b = mt.array([[5, 6]])
    >>> mt.concatenate((a, b), axis=0).execute()
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> mt.concatenate((a, b.T), axis=1).execute()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    tensors = [astensor(t) for t in tensors]

    axis = validate_axis(tensors[0].ndim, axis)
    dtype = np.result_type(*(t.dtype for t in tensors))
    sparse = all(t.issparse() for t in tensors)

    op = TensorConcatenate(axis=axis, dtype=dtype, sparse=sparse)
    return op(tensors)
