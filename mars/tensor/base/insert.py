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
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import Int32Field, TupleField, AnyField, KeyField
from ...utils import has_unknown_shape
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs, validate_axis, calc_object_length


class TensorInsert(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INSERT

    _index_obj = AnyField('index_obj')
    _values = AnyField('values')
    _axis = Int32Field('axis')
    _input = KeyField('input')

    # for chunk
    _range_on_axis = TupleField('range_on_axis')

    def __init__(self, index_obj=None, values=None, axis=None,
                 range_on_axis=None, **kw):
        super().__init__(_index_obj=index_obj, _values=values,
                         _axis=axis, _range_on_axis=range_on_axis, **kw)

    @property
    def index_obj(self):
        return self._index_obj

    @property
    def values(self):
        return self._values

    @property
    def axis(self):
        return self._axis

    @property
    def range_on_axis(self):
        return self._range_on_axis

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        if isinstance(self._index_obj, ENTITY_TYPE):
            self._index_obj = next(inputs_iter)
        if isinstance(self._values, ENTITY_TYPE):
            self._values = next(inputs_iter)

    @classmethod
    def tile(cls, op: 'TensorInsert'):
        inp = op.inputs[0]
        axis = op.axis
        if axis is None:
            inp = yield from recursive_tile(inp.flatten())
            axis = 0
        else:
            new_splits = [s if i == axis else sum(s)
                          for i, s in enumerate(inp.nsplits)]
            inp = yield from recursive_tile(inp.rechunk(new_splits))

        if has_unknown_shape(inp):
            yield

        index_obj = op.index_obj
        values = op.values
        if isinstance(values, ENTITY_TYPE):
            # if values is Mars type, we rechunk it into one chunk and
            # all insert chunks depend on it
            values = yield from recursive_tile(values.rechunk(values.shape))

        nsplits_on_axis = []
        if isinstance(index_obj, int):
            splits = inp.nsplits[axis]
            cum_splits = np.cumsum([0] + list(splits))
            # add 1 for last split
            cum_splits[-1] = cum_splits[-1] + 1
            in_idx = cum_splits.searchsorted(index_obj, side='right') - 1
            out_chunks = []
            for chunk in inp.chunks:
                if chunk.index[axis] == in_idx:
                    chunk_op = op.copy().reset_key()
                    chunk_op._index_obj = index_obj - cum_splits[in_idx]
                    if isinstance(values, ENTITY_TYPE):
                        chunk_values = values.chunks[0]
                    else:
                        chunk_values = values
                    inputs = filter_inputs([chunk, chunk_values])
                    shape = tuple(s + calc_object_length(index_obj) if i == axis else s
                                  for i, s in enumerate(chunk.shape))
                    out_chunks.append(chunk_op.new_chunk(inputs, shape=shape,
                                                         index=chunk.index))
                    nsplits_on_axis.append(shape[axis])
                else:
                    out_chunks.append(chunk)
                    nsplits_on_axis.append(chunk.shape[axis])
        elif isinstance(index_obj, ENTITY_TYPE):
            index_obj = yield from recursive_tile(index_obj.rechunk(index_obj.shape))
            offset = 0
            out_chunks = []
            for chunk in inp.chunks:
                chunk_op = op.copy().reset_key()
                chunk_op._index_obj = index_obj.chunks[0]
                if isinstance(values, ENTITY_TYPE):
                    chunk_values = values.chunks[0]
                else:
                    chunk_values = values
                chunk_op._values = chunk_values
                if chunk.index[axis] + 1 == len(inp.nsplits[axis]):
                    # the last chunk on axis
                    chunk_op._range_on_axis = (offset, offset + chunk.shape[axis] + 1)
                else:
                    chunk_op._range_on_axis = (offset, offset + chunk.shape[axis])
                shape = tuple(np.nan if j == axis else s
                              for j, s in enumerate(chunk.shape))
                inputs = filter_inputs([chunk, index_obj.chunks[0],
                                        chunk_values])
                out_chunks.append(chunk_op.new_chunk(inputs, shape=shape,
                                                     index=chunk.index))
                offset += chunk.shape[axis]
                nsplits_on_axis.append(np.nan)
        else:
            # index object is slice or sequence of ints
            if isinstance(index_obj, slice):
                index_obj = range(index_obj.start or 0, index_obj.stop,
                                  index_obj.step or 1)
            splits = inp.nsplits[axis]
            cum_splits = np.cumsum([0] + list(splits))
            # add 1 for last split
            cum_splits[-1] = cum_splits[-1] + 1
            chunk_idx_params = [[[], []] for _ in splits]
            for i, int_idx in enumerate(index_obj):
                in_idx = cum_splits.searchsorted(int_idx, side='right') - 1
                chunk_idx_params[in_idx][0].append(int_idx - cum_splits[in_idx])
                chunk_idx_params[in_idx][1].append(i)

            out_chunks = []
            offset = 0
            for chunk in inp.chunks:
                idx_on_axis = chunk.index[axis]
                if len(chunk_idx_params[idx_on_axis][0]) > 0:
                    chunk_op = op.copy().reset_key()
                    chunk_index_obj = chunk_idx_params[idx_on_axis][0]
                    shape = tuple(s + len(chunk_index_obj) if j == axis else s
                                  for j, s in enumerate(chunk.shape))
                    if isinstance(values, int):
                        chunk_op._index_obj = chunk_index_obj
                        out_chunks.append(chunk_op.new_chunk([chunk], shape=shape,
                                                             index=chunk.index))
                    elif isinstance(values, ENTITY_TYPE):
                        chunk_op._values = values.chunks[0]
                        if chunk.index[axis] + 1 == len(inp.nsplits[axis]):
                            chunk_op._range_on_axis = (offset, offset + chunk.shape[axis] + 1)
                        else:
                            chunk_op._range_on_axis = (offset, offset + chunk.shape[axis])
                        out_chunks.append(chunk_op.new_chunk([chunk, values.chunks[0]],
                                                             shape=shape,
                                                             index=chunk.index))
                        offset += chunk.shape[axis]
                    else:
                        chunk_op._index_obj = chunk_index_obj
                        values = np.asarray(values)
                        to_shape = [calc_object_length(index_obj, chunk.shape[axis])] + \
                                   [s for j, s in enumerate(inp.shape) if j != axis]
                        if all(j == k for j, k in zip(to_shape, values.shape)):
                            chunk_values = np.asarray(values)[chunk_idx_params[idx_on_axis][1]]
                            chunk_op._values = chunk_values
                            out_chunks.append(chunk_op.new_chunk([chunk], shape=shape,
                                                                 index=chunk.index))
                        else:
                            out_chunks.append(chunk_op.new_chunk([chunk],
                                                                 shape=shape,
                                                                 index=chunk.index))

                    nsplits_on_axis.append(shape[axis])
                else:
                    out_chunks.append(chunk)
                    nsplits_on_axis.append(chunk.shape[axis])

        nsplits = tuple(s if i != axis else tuple(nsplits_on_axis)
                        for i, s in enumerate(inp.nsplits))
        out = op.outputs[0]
        new_op = op.copy()
        return new_op.new_tensors(op.inputs, shape=out.shape, order=out.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op: 'TensorInsert'):
        inp = ctx[op.input.key]
        index_obj = ctx[op.index_obj.key] if hasattr(op.index_obj, 'key') else op.index_obj
        values = ctx[op.values.key] if hasattr(op.values, 'key') else op.values
        if op.range_on_axis is None:
            ctx[op.outputs[0].key] = np.insert(inp, index_obj, values, axis=op.axis)
        else:
            if isinstance(index_obj, slice):
                index_obj = np.arange(index_obj.step or 0,
                                      index_obj.stop,
                                      index_obj.step or 1)
            else:
                index_obj = np.array(index_obj)
            values = np.asarray(values)

            part_index = [i for i, idx in enumerate(index_obj) if (
                    (idx >= op.range_on_axis[0]) and idx < op.range_on_axis[1])]
            if (values.ndim > 0) and \
                    len(index_obj) == len(values) and \
                    (values[0].ndim > 0 or inp.ndim == 1):
                ctx[op.outputs[0].key] = np.insert(
                    inp, index_obj[part_index] - op.range_on_axis[0],
                    values[part_index], axis=op.axis)
            else:
                ctx[op.outputs[0].key] = np.insert(
                    inp, index_obj[part_index] - op.range_on_axis[0],
                    values, axis=op.axis)

    def __call__(self, arr, obj, values, shape):
        return self.new_tensor(filter_inputs([arr, obj, values]),
                               shape=shape, order=arr.order)


def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array like
        Input array.
    obj : int, slice or sequence of ints
        Object that defines the index or indices before which `values` is
        inserted.
    values : array_like
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[...,obj,...] = values``
        is legal.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.
    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.
    See Also
    --------
    append : Append elements at the end of an array.
    concatenate : Join a sequence of arrays along an existing axis.
    delete : Delete elements from an array.
    Notes
    -----
    Note that for higher dimensional inserts `obj=0` behaves very different
    from `obj=[0]` just like `arr[:,0,:] = values` is different from
    `arr[:,[0],:] = values`.
    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = mt.array([[1, 1], [2, 2], [3, 3]])
    >>> a.execute()
    array([[1, 1],
           [2, 2],
           [3, 3]])
    >>> mt.insert(a, 1, 5).execute()
    array([1, 5, 1, ..., 2, 3, 3])
    >>> mt.insert(a, 1, 5, axis=1).execute()
    array([[1, 5, 1],
           [2, 5, 2],
           [3, 5, 3]])
    Difference between sequence and scalars:
    >>> mt.insert(a, [1], [[1],[2],[3]], axis=1).execute()
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> b = a.flatten()
    >>> b.execute()
    array([1, 1, 2, 2, 3, 3])
    >>> mt.insert(b, [2, 2], [5, 6]).execute()
    array([1, 1, 5, ..., 2, 3, 3])
    >>> mt.insert(b, slice(2, 4), [5, 6]).execute()
    array([1, 1, 5, ..., 2, 3, 3])
    >>> mt.insert(b, [2, 2], [7.13, False]).execute() # type casting
    array([1, 1, 7, ..., 2, 3, 3])
    >>> x = mt.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> mt.insert(x, idx, 999, axis=1).execute()
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])
    """
    arr = astensor(arr)
    if getattr(obj, 'ndim', 0) > 1:  # pragma: no cover
        raise ValueError('index array argument obj to insert must be '
                         'one dimensional or scalar')

    if axis is None:
        # if axis is None, array will be flatten
        arr_size = arr.size
        idx_length = calc_object_length(obj, size=arr_size)
        shape = (arr_size + idx_length,)
    else:
        validate_axis(arr.ndim, axis)
        idx_length = calc_object_length(obj, size=arr.shape[axis])
        shape = tuple(s + idx_length if i == axis else s
                      for i, s in enumerate(arr.shape))

    op = TensorInsert(index_obj=obj, values=values, axis=axis, dtype=arr.dtype)
    return op(arr, obj, values, shape)
