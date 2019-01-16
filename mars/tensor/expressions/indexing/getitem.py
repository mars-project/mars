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

from numbers import Integral
import operator
import itertools
import contextlib

import numpy as np

from ....operands import Index
from ....core import BaseWithKey, Entity
from ....compat import reduce, irange, izip
from ...core import TENSOR_TYPE
from ..utils import unify_chunks, split_index_into_chunks, is_asc_sorted, \
    slice_split, calc_sliced_size
from ..core import TensorOperandMixin
from .core import process_index, get_index_and_shape


class TensorIndex(Index, TensorOperandMixin):
    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorIndex, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    def calc_rough_shape(self, *inputs_shape):
        input_shape = inputs_shape[0]
        shape = self.outputs[0].shape
        new_indexes = [index for index in self._indexes if index is not None]

        idx = 0
        new_shape = []
        for index in new_indexes:
            if isinstance(index, (BaseWithKey, Entity)) and index.dtype == np.bool_:
                new_shape.append(input_shape[idx: (idx + index.ndim)])
                idx += index.ndim - 1
            idx += 1

        rough_shape = []
        idx = 0
        for s in shape:
            if np.isnan(s):
                rough_shape.extend(new_shape[idx])
                idx += 1
            else:
                rough_shape.append(s)
        return tuple(rough_shape)

    def calc_shape(self, *inputs_shape):
        return tuple(get_index_and_shape(inputs_shape[0], self._indexes)[1])

    @contextlib.contextmanager
    def _handle_params(self, inputs, indexes):
        """
        Index operator is special, it has additional parameter `indexes` which may also be tensor type,
        normally, this indexes is provided when called by `tile` or `TensorIndex.__call__`, however, calls
        in `GraphActor.get_executable_operand_dag` only provide inputs, in such situation, we need get `indexes`
        from operand itself and replace tensor-liked indexes by new one in `inputs`.
        """
        if indexes is not None:
            self._indexes = indexes
            indexes_inputs = [ind for ind in indexes if isinstance(ind, (BaseWithKey, Entity))]
            inputs = inputs + indexes_inputs
        yield inputs

        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (BaseWithKey, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes

    def new_tensors(self, inputs, shape, **kw):
        indexes = kw.pop('indexes', None)
        with self._handle_params(inputs, indexes) as mix_inputs:
            return super(TensorIndex, self).new_tensors(mix_inputs, shape, **kw)

    def new_chunks(self, inputs, shape, **kw):
        indexes = kw.pop('indexes', None)
        with self._handle_params(inputs, indexes) as mix_inputs:
            return super(TensorIndex, self).new_chunks(mix_inputs, shape, **kw)

    def __call__(self, a, index, shape):
        return self.new_tensor([a], shape, indexes=index)

    @classmethod
    def tile(cls, op):
        from ..merge.concatenate import TensorConcatenate

        in_tensor = op.input
        tensor = op.outputs[0]

        indexes = list(op.indexes)
        axis = 0
        output_axis = 0
        output_chunk_shape = []
        to_concat_axis_index = []
        for i, index in enumerate(indexes):
            if isinstance(index, TENSOR_TYPE) and index.dtype == np.bool_:
                # bool indexing
                # do unify chunk here
                in_tensor, index = unify_chunks(
                    in_tensor, (index, tuple(axis+ii for ii in range(index.ndim))))

                output_chunk_shape.append(reduce(operator.mul, index.chunk_shape))
                indexes[i] = index.chunks
                axis += index.ndim
                output_axis += 1
            elif isinstance(index, np.ndarray):
                # fancy indexing
                if index.ndim > 1:
                    # currently we only support tensor from numpy.ndarray
                    raise NotImplementedError
                try:
                    np_index = np.sort(index)
                    splits = split_index_into_chunks(in_tensor.nsplits[axis], np_index)
                except IndexError as e:
                    idx, size = e.idx, e.size
                    raise IndexError('index {0} is out of bounds for axis {1} with size {2}'.format(
                        idx, axis, size
                    ))

                non_empty_idx_splits = [(j, s) for j, s in enumerate(splits) if s.size > 0]
                non_empty_splits, _ = tuple(zip(*non_empty_idx_splits))
                try:
                    indexes[i] = non_empty_idx_splits
                except:
                    raise
                if not is_asc_sorted(index):
                    pos_index = np.searchsorted(np_index, index)
                    to_concat_axis_index.append((output_axis, pos_index))

                axis += 1
                output_axis += 1
                output_chunk_shape.append(len(non_empty_splits))
            elif isinstance(index, (slice, Integral)):
                indexes[i] = sorted(slice_split(index, in_tensor.nsplits[axis]).items(),
                                    key=operator.itemgetter(0))
                if isinstance(index, slice) and index.step is not None and index.step < 0:
                    indexes[i] = list(reversed(indexes[i]))
                if isinstance(index, slice):
                    output_chunk_shape.append(len(indexes[i]))
                axis += 1
                if isinstance(index, slice):
                    output_axis += 1
            elif isinstance(index, TENSOR_TYPE):
                raise NotImplementedError('Mars currently does not support fancy index from tensor')
            else:
                assert index is None
                output_chunk_shape.append(1)
                output_axis += 1

        out_chunks = []
        for output_idx in itertools.product(*(irange(s) for s in output_chunk_shape)):
            chunk_idx = []
            chunk_index = []  # chunk[index]
            chunk_shape = []
            axis = 0
            output_axis = 0
            for raw_index, index in izip(op.indexes, indexes):
                if isinstance(raw_index, TENSOR_TYPE) and raw_index.dtype == np.bool_:
                    indexed = index[output_idx[output_axis]]
                    chunk_index.append(indexed)
                    chunk_idx.extend(indexed.index)
                    chunk_shape.append(np.nan)
                    axis += raw_index.ndim
                    output_axis += 1
                elif isinstance(raw_index, np.ndarray):
                    input_index, indexed = index[output_idx[output_axis]]
                    chunk_index.append(indexed)
                    chunk_idx.append(input_index)
                    chunk_shape.append(len(indexed))
                    axis += 1
                    output_axis += 1
                elif isinstance(raw_index, slice):
                    sliceobj = index[output_idx[output_axis]][1]
                    chunk_index.append(sliceobj)
                    ix = index[output_idx[output_axis]][0]
                    chunk_idx.append(ix)
                    chunk_shape.append(calc_sliced_size(in_tensor.nsplits[axis][ix], sliceobj))
                    axis += 1
                    output_axis += 1
                elif isinstance(raw_index, Integral):
                    input_index, sliceobj = index[0]
                    chunk_index.append(sliceobj)
                    chunk_idx.append(input_index)
                    axis += 1
                else:
                    chunk_index.append(index)
                    chunk_shape.append(1)
                    output_axis += 1

            chunk_input = in_tensor.cix[tuple(chunk_idx)]
            chunk_op = op.copy().reset_key()
            chunk = chunk_op.new_chunk([chunk_input], tuple(chunk_shape), indexes=chunk_index, index=output_idx)
            out_chunks.append(chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy().reset_key()
        tensor = new_op.new_tensor([op.input], tensor.shape, indexes=op.indexes, chunks=out_chunks, nsplits=nsplits)

        if len(to_concat_axis_index) > 1:
            raise NotImplementedError

        if to_concat_axis_index:
            axis, output_index = to_concat_axis_index[0]
            indexobj = [slice(None)] * axis + [output_index] + [slice(None)] * (tensor.ndim - axis - 1)

            output_shape = list(tensor.shape)
            output_shape[axis] = len(output_index)
            output_nsplits = list(nsplits)
            output_nsplits[axis] = (output_shape[axis],)
            output_chunks = []
            for idx in itertools.product(*(range(len(it)) for it in (nsplits[:axis]+nsplits[axis+1:]))):
                new_idx = idx[:axis] + (0,) + idx[axis:]
                chunk_idxes = (idx[:axis] + (i,) + idx[axis:]
                               for i in range(len(nsplits[axis])))
                chunks = [tensor.cix[chunk_idx] for chunk_idx in chunk_idxes]
                s = list(chunks[0].shape)
                s[axis] = len(output_index)
                concat_chunk_op = TensorConcatenate(
                    axis=axis, dtype=chunks[0].dtype, sparse=chunks[0].op.sparse)
                concat_chunk = concat_chunk_op.new_chunk(chunks, tuple(s), index=new_idx)
                out_chunk_op = TensorIndex(dtype=concat_chunk.dtype, sparse=concat_chunk.op.sparse)
                out_chunk = out_chunk_op.new_chunk([concat_chunk], tuple(s), indexes=indexobj, index=new_idx)
                output_chunks.append(out_chunk)

            new_op = tensor.op.copy()
            tensor = new_op.new_tensor([op.input], tuple(output_shape), indexes=op.indexes,
                                       chunks=output_chunks, nsplits=output_nsplits)

        return [tensor]


def _getitem(a, item):
    if isinstance(item, (list, tuple)) and \
            all(isinstance(it, slice) and it == slice(None) for it in item):
        # nothing to do
        return a

    # TODO(jisheng): field access, e.g. t['a'], t[['a', 'b']]

    index = process_index(a, item)
    index, shape = get_index_and_shape(a.shape, index)
    op = TensorIndex(dtype=a.dtype, sparse=a.issparse())
    return op(a, index, tuple(shape))
