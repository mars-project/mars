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

from .... import opcodes as OperandDef
from ....serialize import KeyField, ListField, Int32Field
from ....core import Base, Entity
from ....compat import irange, OrderedDict
from ...core import TENSOR_TYPE
from ..utils import unify_chunks, slice_split, split_indexes_into_chunks, \
    broadcast_shape, calc_sliced_size
from ..core import TensorHasInput, TensorOperandMixin, \
    TensorShuffleMap, TensorShuffleReduce
from .core import process_index, get_index_and_shape


FANCY_INDEX_TYPES = TENSOR_TYPE + (np.ndarray,)


class TensorIndex(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEX

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorIndex, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @property
    def indexes(self):
        return self._indexes

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
            indexes_inputs = [ind for ind in indexes if isinstance(ind, (Base, Entity))]
            inputs = inputs + indexes_inputs
        yield inputs

        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes

    def _new_tileables(self, inputs, kws=None, **kw):
        indexes = kw.pop('indexes', None)
        with self._handle_params(inputs, indexes) as mix_inputs:
            return super(TensorIndex, self)._new_tileables(mix_inputs, kws=kws, **kw)

    def _new_chunks(self, inputs, kws=None, **kw):
        indexes = kw.pop('indexes', None)
        with self._handle_params(inputs, indexes) as mix_inputs:
            return super(TensorIndex, self)._new_chunks(mix_inputs, kws=kws, **kw)

    def __call__(self, a, index, shape):
        return self.new_tensor([a], shape, indexes=index)

    @staticmethod
    def _fancy_index_distribute(input_tensor, fancy_indexes, axes):
        # fancy_indexes will be all tensors or all ndarrays
        if isinstance(fancy_indexes[0], np.ndarray):
            concat_fancy_index = np.asarray([fi.flatten() for fi in fancy_indexes])
            # first split the fancy indexes into lists which size is identical to
            # chunk size of input_tensor on the specified axes
            nsplits = [input_tensor.nsplits[axis] for axis in axes]
            index_on_splits = split_indexes_into_chunks(nsplits, concat_fancy_index)

            # remember some split index could be empty
            return index_on_splits

        # assert isinstance(fancy_index, TENSOR_TYPE)
        #
        # axis_chunk_shape = input_tensor.chunk_shape[axis]
        # # generate shuffle map
        # shuffle_map_chunks = []
        # for chunk in fancy_index.chunks:
        #     shuffle_map_op = FancyIndexingDistributeMap(dest_chunk_size=axis_chunk_shape, dtype=chunk.dtype)
        #     shuffle_map_chunk = shuffle_map_op.new_chunk(
        #         [chunk], shape=(np.nan, fancy_index.ndim + 1), index=chunk.index)
        #     shuffle_map_chunks.append(shuffle_map_chunk)
        # # do shuffle here
        # proxy_chunk = TensorShuffleProxy(dtype=fancy_index.dtype, tensor_keys=[fancy_index.key]) \
        #     .new_chunk(shuffle_map_chunks, shape=())
        # shuffle_out_chunks = []
        # for j in range(axis_chunk_shape):
        #     shuffle_key = str(j)
        #     shuffle_reduce_op = FancyIndexingDistributeReduce(dtype=proxy_chunk.dtype, _shuffle_key=shuffle_key)
        #     shuffle_reduce_chunk = shuffle_reduce_op.new_chunk(
        #         [proxy_chunk], shape=(np.nan, fancy_index.ndim + 1))
        #     shuffle_out_chunks.append(shuffle_reduce_chunk)
        #
        # return shuffle_out_chunks

    @classmethod
    def _process_all_fancy_indexes(cls, fancy_indexes):
        from ..base import broadcast_to

        shape = broadcast_shape(*[fancy_index.shape for fancy_index in fancy_indexes])
        # fancy indexes should be all tensors or ndarrays
        processed_fancy_indexes = []
        if isinstance(fancy_indexes[0], np.ndarray):
            for fancy_index in fancy_indexes:
                assert isinstance(fancy_index, np.ndarray)
                processed_fancy_indexes.append(np.broadcast_to(fancy_index, shape))
        else:
            broadcast_fancy_indexes = [broadcast_to(fancy_index, shape).single_tiles()
                                       for fancy_index in fancy_indexes]
            broadcast_fancy_indexes = unify_chunks(broadcast_fancy_indexes)
            processed_fancy_indexes.extend(broadcast_fancy_indexes)

        return processed_fancy_indexes

    @classmethod
    def tile(cls, op):
        from ..merge import TensorConcatenate

        in_tensor = op.input
        out_tensor = op.outputs[0]

        in_axis = 0
        out_axis = 0
        out_chunk_shape = []
        fancy_index_in_axis_to_raw = OrderedDict()
        fancy_index_out_in_axes = dict()
        in_axis_to_processed_index = dict()
        for j, index_obj in enumerate(op.indexes):
            if isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_:
                # bool indexing
                # unify chunks first
                index_obj_axes = \
                    (index_obj, tuple(in_axis + i_dim for i_dim in range(index_obj.ndim)))
                in_tensor, index_obj = unify_chunks(in_tensor, index_obj_axes)
                in_axis_to_processed_index[in_axis] = index_obj
                in_axis += index_obj.ndim
                out_axis += 1
                out_chunk_shape.append(len(index_obj.chunks))
            elif isinstance(index_obj, FANCY_INDEX_TYPES):
                # fancy indexing
                # because we need to unify all fancy index's chunks first, so here we don't do process
                fancy_index_in_axis_to_raw[in_axis] = index_obj
                fancy_index_out_in_axes[out_axis] = in_axis
                out_chunk_shape.append(in_tensor.chunk_shape[in_axis])
                in_axis += 1
                out_axis += 1
            elif isinstance(index_obj, slice):
                reverse = (index_obj.step or 0) < 0
                in_axis_to_processed_index[in_axis] = \
                    sorted(slice_split(index_obj, in_tensor.nsplits[in_axis]).items(),
                           key=operator.itemgetter(0), reverse=reverse)
                out_chunk_shape.append(len(in_axis_to_processed_index[in_axis]))
                in_axis += 1
                out_axis += 1
            elif isinstance(index_obj, Integral):
                in_axis_to_processed_index[in_axis] = \
                    list(slice_split(index_obj, in_tensor.nsplits[in_axis]).items())[0]
                in_axis += 1
            else:
                assert index_obj is None
                out_chunk_shape.append(1)
                out_axis += 1

        fancy_index_start_axis = None
        fancy_indexes = None
        select_pos = None
        fancy_indexes_asc_sorted = False
        if fancy_index_in_axis_to_raw:
            fancy_index_start_axis = next(iter(fancy_index_in_axis_to_raw))
            fancy_indexes = cls._process_all_fancy_indexes(list(fancy_index_in_axis_to_raw.values()))
            distributed_fancy_index, select_pos, fancy_indexes_asc_sorted = \
                cls._fancy_index_distribute(in_tensor, fancy_indexes, list(fancy_index_in_axis_to_raw))
            if select_pos.shape != fancy_indexes[0].shape:
                select_pos = select_pos.reshape(fancy_indexes[0].shape)
            in_axis_to_processed_index[fancy_index_start_axis] = distributed_fancy_index

        out_chunks = []
        idx_to_acc = {}
        for out_idx in itertools.product(*(irange(s) for s in out_chunk_shape)):
            in_chunk_idx = []
            chunk_index_obj = []
            chunk_shape = []
            real_out_idx = []
            in_axis = 0
            out_axis = 0
            for index_obj in op.indexes:
                processed_index_obj = in_axis_to_processed_index.get(in_axis)
                if isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_:
                    index_chunk = processed_index_obj.chunks[out_idx[out_axis]]
                    chunk_index_obj.append(index_chunk)
                    in_chunk_idx.extend(index_chunk.index)
                    chunk_shape.append(np.nan)
                    real_out_idx.append(out_idx[out_axis])
                    in_axis += index_obj.ndim
                    out_axis += 1
                elif isinstance(index_obj, FANCY_INDEX_TYPES):
                    # chunk_index_obj and chunk_shape will be generated together for all fancy indexes
                    idx = out_idx[out_axis]
                    in_chunk_idx.append(idx)
                    in_axis += 1
                    out_axis += 1
                elif isinstance(index_obj, slice):
                    idx, sliceobj = processed_index_obj[out_idx[out_axis]]
                    chunk_index_obj.append(sliceobj)
                    in_chunk_idx.append(idx)
                    chunk_shape.append(calc_sliced_size(in_tensor.nsplits[in_axis][idx], sliceobj))
                    real_out_idx.append(out_idx[out_axis])
                    in_axis += 1
                    out_axis += 1
                elif isinstance(index_obj, Integral):
                    idx, sliceobj = processed_index_obj
                    chunk_index_obj.append(sliceobj)
                    in_chunk_idx.append(idx)
                    in_axis += 1
                else:
                    chunk_index_obj.append(None)
                    chunk_shape.append(1)
                    real_out_idx.append(out_idx[out_axis])
                    out_axis += 1

            in_chunk = in_tensor.cix[tuple(in_chunk_idx)]
            if fancy_index_start_axis is not None:
                # calculate fancy index's chunk_index_obj and chunk_shape
                fancy_index_in_idxes = tuple(in_chunk_idx[axis] for axis in fancy_index_in_axis_to_raw)
                splits = in_axis_to_processed_index[fancy_index_start_axis][fancy_index_in_idxes]
                if len(splits[0]) == 0:
                    # this input chunk is not effected by fancy indexing, just skip
                    continue
                for j, fancy_index_axis in enumerate(fancy_index_in_axis_to_raw):
                    chunk_index_obj.insert(fancy_index_axis, splits[j])
                chunk_shape.insert(fancy_index_start_axis, len(splits[0]))

                idx_key = tuple(real_out_idx)
                if idx_key not in idx_to_acc:
                    idx_to_acc[idx_key] = itertools.count(0)
                real_out_idx.insert(fancy_index_start_axis, next(idx_to_acc[idx_key]))

            chunk_op = op.copy().reset_key()
            pos_select_chunk = chunk_op.new_chunk([in_chunk], shape=tuple(chunk_shape),
                                           indexes=chunk_index_obj, index=tuple(real_out_idx))
            out_chunks.append(pos_select_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        chunk_shape = tuple(len(ns) for ns in nsplits)
        index_to_out_chunks = {c.index: c for c in out_chunks}

        if fancy_index_start_axis is not None:
            # handle fancy indexing again
            if isinstance(fancy_indexes[0], np.ndarray) and \
                    (not fancy_indexes_asc_sorted or fancy_indexes[0].ndim > 1):
                # concat the fancy index effected chunk together
                concat_axis = fancy_index_start_axis
                old_chunk_shape = chunk_shape
                chunk_shape = chunk_shape[:concat_axis] + (1,) + chunk_shape[concat_axis + 1:]
                out_chunks = []
                for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
                    to_concat_chunks_idxes = [out_idx[:concat_axis] + (j,) + out_idx[concat_axis + 1:]
                                              for j in range(old_chunk_shape[concat_axis])]
                    to_concat_chunks = [index_to_out_chunks[idx] for idx in to_concat_chunks_idxes]
                    concat_chunk_shape = list(to_concat_chunks[0].shape)
                    concat_chunk_shape[concat_axis] = sum(c.shape[concat_axis] for c in to_concat_chunks)
                    concat_op = TensorConcatenate(axis=concat_axis, dtype=to_concat_chunks[0].dtype,
                                                  sparse=to_concat_chunks[0].issparse())
                    concat_chunk = concat_op.new_chunk(to_concat_chunks, shape=tuple(concat_chunk_shape),
                                                       index=out_idx)
                    out_chunk_op = TensorIndex(dtype=concat_chunk.dtype, sparse=concat_chunk.issparse())
                    out_index_obj = [slice(None)] * concat_axis + [select_pos] + \
                                    [slice(None)] * (len(nsplits) - concat_axis - 1)
                    pos_select_chunk = out_chunk_op.new_chunk([concat_chunk], shape=concat_chunk.shape,
                                                              indexes=out_index_obj, index=out_idx)
                    # if op.indexes[fancy_index_start_axis].ndim == 1:
                    out_chunks.append(pos_select_chunk)
                nsplits[concat_axis] = (sum(nsplits[concat_axis]),)

        new_op = op.copy()
        new_tensor = new_op.new_tensor(op.inputs, out_tensor.shape, indexes=op.indexes,
                                       chunks=out_chunks, nsplits=nsplits)
        return [new_tensor]


class FancyIndexingDistributeMap(TensorShuffleMap):
    _op_type_ = OperandDef.FANCY_INDEX_DISTRIBUTE_MAP

    _dest_chunk_size = Int32Field('dest_chunk_size')

    def __init__(self, dest_chunk_size=None, dtype=None, sparse=None, **kw):
        super(FancyIndexingDistributeMap, self).__init__(
            _dest_chunk_size=dest_chunk_size, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def dest_chunk_size(self):
        return self._dest_chunk_size


class FancyIndexingDistributeReduce(TensorShuffleReduce):
    _op_type_ = OperandDef.FANCY_INDEX_DISTRIBUTE_REDUCE

    _input = KeyField('input')

    def __init__(self, dtype=None, sparse=None, **kw):
        super(FancyIndexingDistributeReduce, self).__init__(
            _dtype=dtype, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(FancyIndexingDistributeReduce, self)._set_inputs(inputs)
        self._input = inputs[0]

    @property
    def input(self):
        return self._input


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
