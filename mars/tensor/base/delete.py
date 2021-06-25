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

from collections import defaultdict

import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import Int32Field, Int64Field, AnyField, KeyField
from ...utils import has_unknown_shape
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs, validate_axis, slice_split, calc_object_length


class TensorDelete(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.DELETE

    _index_obj = AnyField('index_obj')
    _axis = Int32Field('axis')
    _input = KeyField('input')

    # for chunk
    _offset_on_axis = Int64Field('offset_on_axis')

    def __init__(self, index_obj=None, axis=None, offset_on_axis=None, **kw):
        super().__init__(_index_obj=index_obj, _axis=axis,
                         _offset_on_axis=offset_on_axis, **kw)

    @property
    def index_obj(self):
        return self._index_obj

    @property
    def axis(self):
        return self._axis

    @property
    def offset_on_axis(self):
        return self._offset_on_axis

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if len(self._inputs) > 1:
            self._index_obj = self._inputs[1]

    @classmethod
    def tile(cls, op: 'TensorDelete'):
        inp = op.input
        index_obj = op.index_obj
        axis = op.axis
        if axis is None:
            inp = yield from recursive_tile(inp.flatten())
            axis = 0
        if has_unknown_shape(inp):
            yield

        if isinstance(index_obj, int):
            index_obj = [index_obj]

        if isinstance(index_obj, ENTITY_TYPE):
            index_obj = yield from recursive_tile(
                index_obj.rechunk(index_obj.shape))
            offsets = np.cumsum([0] + list(inp.nsplits[axis]))
            out_chunks = []
            for c in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op._index_obj = index_obj.chunks[0]
                chunk_op._offset_on_axis = int(offsets[c.index[axis]])
                shape = tuple(np.nan if j == axis else s
                              for j, s in enumerate(c.shape))
                out_chunks.append(chunk_op.new_chunk([c, index_obj.chunks[0]],
                                                     shape=shape,
                                                     index=c.index))
            nsplits_on_axis = (np.nan,) * len(inp.nsplits[axis])
        else:
            nsplits_on_axis = [None for _ in inp.nsplits[axis]]
            out_chunks = []
            # index_obj is list, tuple, slice or array like
            if isinstance(index_obj, slice):
                slc_splits = slice_split(index_obj, inp.nsplits[axis])
                for c in inp.chunks:
                    if c.index[axis] in slc_splits:
                        chunk_op = op.copy().reset_key()
                        chunk_slc = slc_splits[c.index[axis]]
                        shape = tuple(s - calc_object_length(chunk_slc, s) if j == axis else s
                                      for j, s in enumerate(c.shape))
                        chunk_op._index_obj = chunk_slc
                        out_chunks.append(
                            chunk_op.new_chunk([c], shape=shape, index=c.index))
                        nsplits_on_axis[c.index[axis]] = shape[axis]
                    else:
                        out_chunks.append(c)
                        nsplits_on_axis[c.index[axis]] = c.shape[axis]
            else:
                index_obj = np.array(index_obj)
                cum_splits = np.cumsum([0] + list(inp.nsplits[axis]))
                chunk_indexes = defaultdict(list)
                for int_idx in index_obj:
                    in_idx = cum_splits.searchsorted(int_idx, side='right') - 1
                    chunk_indexes[in_idx].append(int_idx - cum_splits[in_idx])

                for c in inp.chunks:
                    idx_on_axis = c.index[axis]
                    if idx_on_axis in chunk_indexes:
                        chunk_op = op.copy().reset_key()
                        chunk_op._index_obj = chunk_indexes[idx_on_axis]
                        shape = tuple(s - len(chunk_indexes[idx_on_axis])
                                      if j == axis else s for j, s in enumerate(c.shape))
                        out_chunks.append(
                            chunk_op.new_chunk([c], shape=shape, index=c.index))
                        nsplits_on_axis[c.index[axis]] = shape[axis]
                    else:
                        out_chunks.append(c)
                        nsplits_on_axis[c.index[axis]] = c.shape[axis]

        nsplits = tuple(s if i != axis else tuple(nsplits_on_axis)
                        for i, s in enumerate(inp.nsplits))
        out = op.outputs[0]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        inp = ctx[op.input.key]
        index_obj = ctx[op.index_obj.key] if hasattr(op.index_obj, 'key') else op.index_obj
        if op.offset_on_axis is None:
            ctx[op.outputs[0].key] = np.delete(inp, index_obj, axis=op.axis)
        else:
            index_obj = np.array(index_obj)
            part_index = [idx - op.offset_on_axis for idx in index_obj if (
                    (idx >= op.offset_on_axis) and idx < (op.offset_on_axis + inp.shape[op.axis or 0]))]

            ctx[op.outputs[0].key] = np.delete(
                inp, part_index, axis=op.axis)

    def __call__(self, arr, obj, shape):
        return self.new_tensor(filter_inputs([arr, obj]),
                               shape=shape, order=arr.order)


def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int or array of ints
        Indicate indices of sub-arrays to remove along the specified axis.
    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : mars.tensor
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> arr = mt.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr.execute()
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> mt.delete(arr, 1, 0).execute()
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])
    >>> mt.delete(arr, np.s_[::2], 1).execute()
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> mt.delete(arr, [1,3,5], None).execute()
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])
    """
    arr = astensor(arr)
    arr = astensor(arr)
    if getattr(obj, 'ndim', 0) > 1:  # pragma: no cover
        raise ValueError('index array argument obj to insert must be '
                         'one dimensional or scalar')

    if axis is None:
        # if axis is None, array will be flatten
        arr_size = arr.size
        idx_length = calc_object_length(obj, size=arr_size)
        shape = (arr_size - idx_length,)
    else:
        validate_axis(arr.ndim, axis)
        idx_length = calc_object_length(obj, size=arr.shape[axis])
        shape = tuple(s - idx_length if i == axis else s
                      for i, s in enumerate(arr.shape))

    op = TensorDelete(index_obj=obj, axis=axis, dtype=arr.dtype)
    return op(arr, obj, shape)
