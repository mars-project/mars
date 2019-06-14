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

import numpy as np

from .... import opcodes as OperandDef
from ....serialize import ValueType, KeyField, ListField, TupleField, Int32Field
from ....core import Base, Entity
from ....compat import irange, OrderedDict
from ...core import TENSOR_TYPE
from ..utils import unify_chunks, slice_split, split_indexes_into_chunks, \
    calc_pos, broadcast_shape, calc_sliced_size, recursive_tile, filter_inputs
from ..core import TensorHasInput, TensorOperandMixin, \
    TensorShuffleMap, TensorShuffleReduce, TensorShuffleProxy
from .core import process_index, calc_shape


FANCY_INDEX_TYPES = TENSOR_TYPE + (np.ndarray,)


class TensorIndex(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEX

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, dtype=None, sparse=False, indexes=None, **kw):
        super(TensorIndex, self).__init__(_dtype=dtype, _sparse=sparse, _indexes=indexes, **kw)

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super(TensorIndex, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes

    def __call__(self, a, index, shape):
        self._indexes = index
        return self.new_tensor(filter_inputs([a] + list(index)), shape)

    @staticmethod
    def _fancy_index_distribute(input_tensor, fancy_indexes, axes):
        from ..merge import stack

        # fancy_indexes will be all tensors or all ndarrays
        if isinstance(fancy_indexes[0], np.ndarray):
            concat_fancy_index = np.asarray([fi.flatten() for fi in fancy_indexes])
            # first split the fancy indexes into lists which size is identical to
            # chunk size of input_tensor on the specified axes
            nsplits = [input_tensor.nsplits[axis] for axis in axes]
            # some split index could be empty
            return split_indexes_into_chunks(nsplits, concat_fancy_index)

        assert isinstance(fancy_indexes[0], TENSOR_TYPE)

        # stack fancy indexes into one
        concat_fancy_index = recursive_tile(stack(fancy_indexes))
        concat_fancy_index = concat_fancy_index.rechunk({0: len(fancy_indexes)}).single_tiles()

        # generate shuffle map
        shuffle_map_chunks = []
        for chunk in concat_fancy_index.chunks:
            shuffle_map_op = FancyIndexingDistributeMap(
                dest_nsplits=input_tensor.nsplits, axes=axes, dtype=chunk.dtype)
            shuffle_map_chunk = shuffle_map_op.new_chunk([chunk], shape=(np.nan,), index=chunk.index)
            shuffle_map_chunks.append(shuffle_map_chunk)
        # do shuffle here
        proxy_chunk = TensorShuffleProxy(dtype=fancy_indexes[0].dtype, tensor_keys=[fancy_indexes[0].key]) \
            .new_chunk(shuffle_map_chunks, shape=())
        idx_to_fancy_indexes_chunks = OrderedDict()
        idx_to_pos_chunks = OrderedDict()
        for idxes in itertools.product(*(range(input_tensor.chunk_shape[axis]) for axis in axes)):
            shuffle_key = ','.join(str(idx) for idx in idxes)
            shuffle_reduce_op = FancyIndexingDistributeReduce(
                axes=axes, dtype=proxy_chunk.dtype, _shuffle_key=shuffle_key)
            kws = []
            for ax in axes:
                kw = {
                    'axis': ax,
                    'shape': (np.nan,),
                    'index': idxes
                }
                kws.append(kw)
            kws.append({'pos': True, 'shape': (np.nan,), 'index': idxes})
            shuffle_reduce_chunks = shuffle_reduce_op.new_chunks([proxy_chunk], kws=kws)
            idx_to_fancy_indexes_chunks[idxes] = shuffle_reduce_chunks[:-1]
            idx_to_pos_chunks[idxes] = shuffle_reduce_chunks[-1]

        return idx_to_fancy_indexes_chunks, idx_to_pos_chunks, False

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
            broadcast_fancy_indexes = unify_chunks(*broadcast_fancy_indexes)
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
        fancy_index_out_in_axes = OrderedDict()
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

        fancy_index_start_in_axis = None
        fancy_index_start_out_axis = None
        fancy_indexes = None
        poses = None
        fancy_indexes_asc_sorted = False
        if fancy_index_in_axis_to_raw:
            fancy_index_start_in_axis = next(iter(fancy_index_in_axis_to_raw))
            fancy_index_start_out_axis = next(iter(fancy_index_out_in_axes))
            fancy_indexes = cls._process_all_fancy_indexes(list(fancy_index_in_axis_to_raw.values()))
            distributed_fancy_index, poses, fancy_indexes_asc_sorted = \
                cls._fancy_index_distribute(in_tensor, fancy_indexes, list(fancy_index_in_axis_to_raw))
            in_axis_to_processed_index[fancy_index_start_in_axis] = distributed_fancy_index

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
            if fancy_index_start_in_axis is not None:
                # calculate fancy index's chunk_index_obj and chunk_shape
                fancy_index_in_idxes = tuple(in_chunk_idx[axis] for axis in fancy_index_in_axis_to_raw)
                splits = in_axis_to_processed_index[fancy_index_start_in_axis][fancy_index_in_idxes]
                if splits[0].shape[0] == 0:
                    # this input chunk is not effected by fancy indexing, just skip
                    continue
                for j, fancy_index_axis in enumerate(fancy_index_in_axis_to_raw):
                    chunk_index_obj.insert(fancy_index_axis, splits[j])
                chunk_shape.insert(fancy_index_start_out_axis, splits[0].shape[0])

                idx_key = tuple(real_out_idx)
                if idx_key not in idx_to_acc:
                    idx_to_acc[idx_key] = itertools.count(0)
                real_out_idx.insert(fancy_index_start_out_axis, next(idx_to_acc[idx_key]))

            chunk_op = op.copy().reset_key()
            chunk_op._indexes = chunk_index_obj
            out_chunk = chunk_op.new_chunk(filter_inputs([in_chunk] + chunk_index_obj),
                                           shape=tuple(chunk_shape), index=tuple(real_out_idx))
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks
                         if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        chunk_shape = tuple(len(ns) for ns in nsplits)
        index_to_out_chunks = {c.index: c for c in out_chunks}

        if fancy_index_start_in_axis is not None:
            # handle fancy indexing again
            concat_axis = fancy_index_start_out_axis
            if isinstance(fancy_indexes[0], np.ndarray) and \
                    (not fancy_indexes_asc_sorted or fancy_indexes[0].ndim > 1):
                # concat the fancy index effected chunk together
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
                    select_pos = calc_pos(fancy_indexes[0].shape, poses)
                    out_index_obj = [slice(None)] * concat_axis + [select_pos] + \
                                    [slice(None)] * (len(nsplits) - concat_axis - 1)
                    out_chunk_op = TensorIndex(dtype=concat_chunk.dtype, sparse=concat_chunk.issparse(),
                                               indexes=out_index_obj)
                    pos_select_shape = concat_chunk.shape[:concat_axis] + fancy_indexes[0].shape + \
                        concat_chunk.shape[concat_axis + 1:]
                    pos_select_idx = out_idx[:concat_axis] + (0,) * fancy_indexes[0].ndim + \
                        out_idx[concat_axis + 1:]
                    pos_select_chunk = out_chunk_op.new_chunk([concat_chunk], shape=pos_select_shape,
                                                              index=pos_select_idx)
                    out_chunks.append(pos_select_chunk)
                nsplits = nsplits[:concat_axis] + [(s,) for s in fancy_indexes[0].shape] + \
                    nsplits[concat_axis + 1:]
            elif isinstance(fancy_indexes[0], TENSOR_TYPE):
                concat_idx_to_map_chunks = dict()
                for c in out_chunks:
                    pos_idx = np.unravel_index(c.index[concat_axis],
                                               tuple(in_tensor.chunk_shape[ax]
                                                     for ax in fancy_index_in_axis_to_raw))
                    pos_chunk = poses[pos_idx]
                    concat_map_op = FancyIndexingConcatMap(fancy_index_axis=fancy_index_start_out_axis,
                                                           sparse=c.issparse(), dtype=c.dtype)
                    concat_map_chunk_shape = c.shape[:concat_axis] + (np.nan,) + c.shape[concat_axis + 1:]
                    concat_map_chunk = concat_map_op.new_chunk([c, pos_chunk], shape=concat_map_chunk_shape,
                                                               index=c.index)
                    concat_idx_to_map_chunks[concat_map_chunk.index] = concat_map_chunk
                out_chunks = []
                no_shuffle_chunk_shape = chunk_shape[:concat_axis] + chunk_shape[concat_axis + 1:]
                for idx in itertools.product(*(range(s) for s in no_shuffle_chunk_shape)):
                    to_shuffle_chunks = []
                    for f_idx in range(chunk_shape[concat_axis]):
                        concat_idx = idx[:concat_axis] + (f_idx,) + idx[concat_axis:]
                        to_shuffle_chunks.append(concat_idx_to_map_chunks[concat_idx])
                    proxy_op = TensorShuffleProxy(dtype=to_shuffle_chunks[0].dtype,
                                                  no_shuffle_idx=idx)
                    proxy_chunk = proxy_op.new_chunk(to_shuffle_chunks, shape=())
                    acc = itertools.count(0)
                    for reduce_idx in itertools.product(*(range(s) for s in fancy_indexes[0].chunk_shape)):
                        fancy_index_chunk = fancy_indexes[0].cix[reduce_idx]
                        concat_reduce_op = FancyIndexingConcatReduce(fancy_index_axis=fancy_index_start_out_axis,
                                                                     fancy_index_shape=fancy_index_chunk.shape,
                                                                     dtype=proxy_chunk.dtype,
                                                                     sparse=to_shuffle_chunks[0].issparse(),
                                                                     _shuffle_key=str(next(acc)))
                        reduce_chunk_shape = no_shuffle_chunk_shape[:concat_axis] + \
                            fancy_index_chunk.shape + no_shuffle_chunk_shape[concat_axis:]
                        reduce_chunk_idx = idx[:concat_axis] + fancy_index_chunk.index + idx[concat_axis:]
                        concat_reduce_chunk = concat_reduce_op.new_chunk([proxy_chunk], shape=reduce_chunk_shape,
                                                                         index=reduce_chunk_idx)
                        out_chunks.append(concat_reduce_chunk)
                nsplits = nsplits[:concat_axis] + list(fancy_indexes[0].nsplits) + nsplits[concat_axis + 1:]

        new_op = op.copy()
        new_tensor = new_op.new_tensor(op.inputs, out_tensor.shape,
                                       chunks=out_chunks, nsplits=nsplits)
        return [new_tensor]


class FancyIndexingDistributeMap(TensorShuffleMap, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_DISTRIBUTE_MAP

    _dest_nsplits = TupleField('dest_nsplits', ValueType.tuple(ValueType.uint64))
    _axes = TupleField('axes', ValueType.int32)

    def __init__(self, dest_nsplits=None, axes=None, dtype=None, sparse=None, **kw):
        super(FancyIndexingDistributeMap, self).__init__(
            _dest_nsplits=dest_nsplits, _axes=axes, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def dest_nsplits(self):
        return self._dest_nsplits

    @property
    def axes(self):
        return self._axes


class FancyIndexingDistributeReduce(TensorShuffleReduce, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_DISTRIBUTE_REDUCE

    _input = KeyField('input')
    _axes = TupleField('axes', ValueType.int32)

    def __init__(self, axes=None, dtype=None, sparse=None, **kw):
        super(FancyIndexingDistributeReduce, self).__init__(
            _axes=axes, _dtype=dtype, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super(FancyIndexingDistributeReduce, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    @property
    def output_limit(self):
        # return fancy indexes on each axis as well as original position
        return len(self._axes) + 1

    @property
    def axes(self):
        return self._axes

    @property
    def input(self):
        return self._input


class FancyIndexingConcatMap(TensorShuffleMap, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_CONCAT_MAP

    _fancy_index_axis = Int32Field('fancy_index_axis')

    def __init__(self, fancy_index_axis=None, dtype=None, sparse=None, **kw):
        super(FancyIndexingConcatMap, self).__init__(
            _fancy_index_axis=fancy_index_axis, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def input(self):
        return self._input

    @property
    def fancy_index_axis(self):
        return self._fancy_index_axis


class FancyIndexingConcatReduce(TensorShuffleReduce, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_CONCAT_REDUCE

    _fancy_index_axis = Int32Field('fancy_index_axis')
    _fancy_index_shape = TupleField('fancy_index_shape', ValueType.int64)

    def __init__(self, fancy_index_axis=None, fancy_index_shape=None, dtype=None, sparse=None, **kw):
        super(FancyIndexingConcatReduce, self).__init__(
            _fancy_index_axis=fancy_index_axis, _fancy_index_shape=fancy_index_shape,
            _dtype=dtype, _sparse=sparse, **kw)

    @property
    def fancy_index_axis(self):
        return self._fancy_index_axis

    @property
    def fancy_index_shape(self):
        return self._fancy_index_shape


def _getitem(a, item):
    if isinstance(item, (list, tuple)) and \
            all(isinstance(it, slice) and it == slice(None) for it in item):
        # nothing to do
        return a

    # TODO(jisheng): field access, e.g. t['a'], t[['a', 'b']]

    index = process_index(a, item)
    shape = calc_shape(a.shape, index)
    op = TensorIndex(dtype=a.dtype, sparse=a.issparse(), indexes=index)
    return op(a, index, tuple(shape))
