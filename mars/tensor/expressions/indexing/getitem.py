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
from ....compat import OrderedDict, Enum, reduce
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

    @classmethod
    def tile(cls, op):
        tile_handler = TensorIndexTilesHandler(op)
        return tile_handler()


class IndexInfo(object):
    def __init__(self, raw_index_obj, index_obj, index_type, in_axis, out_axis):
        self.raw_index_obj = raw_index_obj
        self.index_obj = index_obj
        self.index_type = index_type
        self.in_axis = in_axis
        self.out_axis = out_axis


class FancyIndexInfo(object):
    def __init__(self):
        self.chunk_index_to_pos = None
        self.fancy_index_asc_sorted = None
        self.fancy_index_in_axes = None
        self.chunk_unified_fancy_indexes = None
        self.fancy_index_chunk_idx_to_out_idx = dict()
        self.fancy_index_all_ndarray = False

    def __bool__(self):
        return self.chunk_index_to_pos is not None

    __nonzero__ = __bool__


class IndexType(Enum):
    bool_index = 0
    fancy_index = 1
    slice = 2
    integral = 3
    new_axis = 4


class TensorIndexTilesHandler(object):
    def __init__(self, op):
        self._op = op
        self._in_tensor = self._op.input
        self._index_infos = []
        self._fancy_index_infos = []
        self._fancy_index_info = FancyIndexInfo()
        self._out_chunks = []
        self._nsplits = None
        self._chunk_shape = None

    @classmethod
    def _is_bool_index(cls, index_obj):
        return isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_

    @classmethod
    def _is_fancy_index(cls, index_obj):
        return isinstance(index_obj, FANCY_INDEX_TYPES) and index_obj.dtype != np.bool_

    def _extract_indexes_info(self):
        in_axis = out_axis = 0
        fancy_index_out_axis = None
        for raw_index_obj in self._op.indexes:
            if self._is_bool_index(raw_index_obj):
                # bool indexing
                # unify chunk first
                index_obj_axes = (raw_index_obj,
                                  tuple(in_axis + i_dim for i_dim in range(raw_index_obj.ndim)))
                in_tensor, index_obj = unify_chunks(self._in_tensor, index_obj_axes)
                self._in_tensor = in_tensor
                self._index_infos.append(IndexInfo(raw_index_obj, index_obj,
                                                   IndexType.bool_index, in_axis, out_axis))
                in_axis += index_obj.ndim
                out_axis += 1
            elif self._is_fancy_index(raw_index_obj):
                # fancy indexing
                # because we need to unify all fancy indexes' chunks together later
                # so here we don't do process any of them here
                first_fancy_index = False
                if fancy_index_out_axis is None:
                    first_fancy_index = True
                    fancy_index_out_axis = out_axis
                index_info = IndexInfo(raw_index_obj, None, IndexType.fancy_index,
                                       in_axis, fancy_index_out_axis)
                self._index_infos.append(index_info)
                self._fancy_index_infos.append(index_info)
                in_axis += 1
                if first_fancy_index:
                    out_axis += 1
            elif isinstance(raw_index_obj, slice):
                reverse = (raw_index_obj.step or 0) < 0
                idx_to_slices = sorted(slice_split(raw_index_obj, self._in_tensor.nsplits[in_axis]).items(),
                                       key=operator.itemgetter(0), reverse=reverse)
                index_obj = OrderedDict()
                for j, idx_to_slice in enumerate(idx_to_slices):
                    idx, s = idx_to_slice
                    index_obj[idx] = (j, s)
                self._index_infos.append(IndexInfo(raw_index_obj, index_obj, IndexType.slice,
                                                   in_axis, out_axis))
                in_axis += 1
                out_axis += 1
            elif isinstance(raw_index_obj, Integral):
                index_obj = slice_split(raw_index_obj, self._in_tensor.nsplits[in_axis])
                self._index_infos.append(IndexInfo(raw_index_obj, index_obj, IndexType.integral,
                                                   in_axis, out_axis))
                in_axis += 1
            else:
                # new axis
                assert raw_index_obj is None
                self._index_infos.append(IndexInfo(raw_index_obj, raw_index_obj, IndexType.new_axis,
                                                   in_axis, out_axis))
                out_axis += 1

    def _preprocess_fancy_indexes(self):
        from ..base import broadcast_to

        if len(self._fancy_index_infos) == 0:
            return

        fancy_indexes = [info.raw_index_obj for info in self._fancy_index_infos]

        shape = broadcast_shape(*[fancy_index.shape for fancy_index in fancy_indexes])
        # fancy indexes should be all tensors or ndarrays
        if isinstance(fancy_indexes[0], np.ndarray):
            self._fancy_index_info.chunk_unified_fancy_indexes = \
                [np.broadcast_to(fancy_index, shape) for fancy_index in fancy_indexes]
        else:
            broadcast_fancy_indexes = [broadcast_to(fancy_index, shape).single_tiles()
                                       for fancy_index in fancy_indexes]
            broadcast_fancy_indexes = unify_chunks(*broadcast_fancy_indexes)
            self._fancy_index_info.chunk_unified_fancy_indexes = broadcast_fancy_indexes

    def _extract_ndarray_fancy_index_info(self, fancy_indexes):
        # concat fancy indexes together
        concat_fancy_index = np.asarray([fi.flatten() for fi in fancy_indexes])
        # first split the fancy indexes into lists which size is identical to
        # chunk size of input_tensor on the specified axes
        nsplits = [self._in_tensor.nsplits[info.in_axis] for info in self._fancy_index_infos]
        chunk_index_to_fancy_indexes_chunks, chunk_index_to_pos, fancy_index_asc_sorted = \
            split_indexes_into_chunks(nsplits, concat_fancy_index)
        for index_info in self._fancy_index_infos:
            index_info.index_obj = chunk_index_to_fancy_indexes_chunks
        self._fancy_index_info.chunk_index_to_pos = chunk_index_to_pos
        self._fancy_index_info.fancy_index_asc_sorted = fancy_index_asc_sorted

    def _extract_tensor_fancy_index_info(self, fancy_indexes):
        from ..merge import stack

        axes = tuple(info.in_axis for info in self._fancy_index_infos)

        # stack fancy indexes into one
        concat_fancy_index = recursive_tile(stack(fancy_indexes))
        concat_fancy_index = concat_fancy_index.rechunk({0: len(fancy_indexes)}).single_tiles()

        # generate shuffle map, for concatenated fancy index,
        # calculated a counterpart index chunk for each chunk of input tensor
        shuffle_map_chunks = []
        for chunk in concat_fancy_index.chunks:
            shuffle_map_op = FancyIndexingDistributeMap(
                dest_nsplits=self._in_tensor.nsplits, axes=axes, dtype=chunk.dtype)
            shuffle_map_chunk = shuffle_map_op.new_chunk([chunk], shape=(np.nan,), index=chunk.index)
            shuffle_map_chunks.append(shuffle_map_chunk)
        # shuffle proxy
        proxy_chunk = TensorShuffleProxy(dtype=fancy_indexes[0].dtype, tensor_keys=[fancy_indexes[0].key]) \
            .new_chunk(shuffle_map_chunks, shape=())
        chunk_index_to_fancy_indexes_chunks = OrderedDict()
        chunk_index_to_pos = OrderedDict()
        for idx in itertools.product(*(range(self._in_tensor.chunk_shape[axis]) for axis in axes)):
            shuffle_key = ','.join(str(i) for i in idx)
            shuffle_reduce_op = FancyIndexingDistributeReduce(axes=axes, dtype=proxy_chunk.dtype,
                                                              _shuffle_key=shuffle_key)
            # chunks of fancy indexes on each axis
            kws = [{'axis': ax, 'shape': (np.nan,), 'index': idx} for ax in axes]
            kws.append({'pos': True, 'shape': (np.nan,), 'index': idx})
            shuffle_reduce_chunks = shuffle_reduce_op.new_chunks([proxy_chunk], kws=kws)
            chunk_index_to_fancy_indexes_chunks[idx] = shuffle_reduce_chunks[:-1]
            chunk_index_to_pos[idx] = shuffle_reduce_chunks[-1]

        for index_info in self._fancy_index_infos:
            index_info.index_obj = chunk_index_to_fancy_indexes_chunks
        self._fancy_index_info.chunk_index_to_pos = chunk_index_to_pos
        self._fancy_index_info.fancy_index_asc_sorted = False

    def _process_fancy_indexes(self):
        if len(self._fancy_index_infos) == 0:
            return

        fancy_index_infos = self._fancy_index_infos
        fancy_indexes = self._fancy_index_info.chunk_unified_fancy_indexes
        if isinstance(fancy_indexes[0], np.ndarray):
            self._extract_ndarray_fancy_index_info(
                fancy_indexes)
            self._fancy_index_info.fancy_index_all_ndarray = True
        else:
            self._extract_tensor_fancy_index_info(fancy_indexes)
        self._fancy_index_info.fancy_index_in_axes = \
            OrderedDict([(info.in_axis, i) for i, info in enumerate(fancy_index_infos)])
        out_idx = itertools.count(0)
        for chunk_idx, fancy_index_chunks in fancy_index_infos[0].index_obj.items():
            if fancy_index_chunks[0].shape[0] != 0:
                self._fancy_index_info.fancy_index_chunk_idx_to_out_idx[chunk_idx] = next(out_idx)

    def _process_in_tensor(self):
        for chunk in self._in_tensor.chunks:
            chunk_index = []  # chunk.index
            chunk_shape = []
            chunk_index_objs = []
            ignore = False
            for index_info in self._index_infos:
                if index_info.index_type == IndexType.bool_index:
                    chunk_shape.append(np.nan)
                    in_axis = index_info.in_axis
                    n_axes = index_info.index_obj.ndim
                    chunk_index_obj_idx = chunk.index[in_axis: in_axis + n_axes]
                    chunk_index_obj = index_info.index_obj.cix[chunk_index_obj_idx]
                    chunk_index_objs.append(chunk_index_obj)
                    cs = self._in_tensor.chunk_shape
                    out_chunk_idx = sum(idx * reduce(operator.mul, cs[i+1:], 1) for i, idx
                                        in zip(itertools.count(0), chunk_index_obj_idx))
                    chunk_index.append(out_chunk_idx)
                elif index_info.index_type == IndexType.fancy_index:
                    fancy_in_axis_to_idx = self._fancy_index_info.fancy_index_in_axes
                    i_fancy_index = fancy_in_axis_to_idx[index_info.in_axis]
                    in_chunk_idx = chunk.index
                    chunk_index_obj_idx = tuple(in_chunk_idx[ax] for ax in fancy_in_axis_to_idx)
                    chunk_index_obj = index_info.index_obj[chunk_index_obj_idx][i_fancy_index]
                    if chunk_index_obj.shape[0] == 0:
                        ignore = True
                        break
                    chunk_index_objs.append(chunk_index_obj)
                    if i_fancy_index == 0:
                        chunk_index.append(
                            self._fancy_index_info.fancy_index_chunk_idx_to_out_idx[chunk_index_obj_idx])
                        chunk_shape.append(chunk_index_obj.shape[0])
                elif index_info.index_type == IndexType.slice:
                    in_axis = index_info.in_axis
                    out_chunk_idx, chunk_index_obj = \
                        index_info.index_obj.get(chunk.index[in_axis], (None, None))
                    if chunk_index_obj is None:
                        ignore = True
                        break
                    chunk_index_objs.append(chunk_index_obj)
                    chunk_index.append(out_chunk_idx)
                    chunk_shape.append(calc_sliced_size(chunk.shape[in_axis], chunk_index_obj))
                elif index_info.index_type == IndexType.integral:
                    chunk_index_obj = index_info.index_obj.get(chunk.index[index_info.in_axis])
                    if chunk_index_obj is None:
                        ignore = True
                        break
                    chunk_index_objs.append(chunk_index_obj)
                else:
                    chunk_index_objs.append(None)
                    chunk_index.append(0)
                    chunk_shape.append(1)
            if ignore:
                continue
            chunk_op = self._op.copy().reset_key()
            chunk_op._indexes = chunk_index_objs
            out_chunk = chunk_op.new_chunk(filter_inputs([chunk] + chunk_index_objs),
                                           shape=tuple(chunk_shape), index=tuple(chunk_index))
            self._out_chunks.append(out_chunk)

        self._out_chunks = sorted(self._out_chunks, key=operator.attrgetter('index'))
        self._nsplits = [tuple(c.shape[i] for c in self._out_chunks
                               if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                         for i in range(len(self._out_chunks[0].shape))]
        self._chunk_shape = tuple(len(ns) for ns in self._nsplits)

    def _postprocess_ndarray_fancy_index(self, fancy_index_infos):
        from ..merge import TensorConcatenate

        if self._fancy_index_info.fancy_index_asc_sorted and \
                self._fancy_index_info.chunk_unified_fancy_indexes[0].ndim == 1:
            return

        concat_axis = fancy_index_infos[0].out_axis
        chunk_shape = self._chunk_shape
        index_to_out_chunks = {c.index: c for c in self._out_chunks}
        fancy_indexes = self._fancy_index_info.chunk_unified_fancy_indexes
        self._chunk_shape = self._chunk_shape[:concat_axis] + (1,) + self._chunk_shape[concat_axis + 1:]

        out_chunks = []
        for out_idx in itertools.product(*(range(s) for s in self._chunk_shape)):
            to_concat_chunks_idxes = [out_idx[:concat_axis] + (j,) + out_idx[concat_axis + 1:]
                                      for j in range(chunk_shape[concat_axis])]
            to_concat_chunks = [index_to_out_chunks[idx] for idx in to_concat_chunks_idxes]
            concat_chunk_shape = list(to_concat_chunks[0].shape)
            concat_chunk_shape[concat_axis] = sum(c.shape[concat_axis] for c in to_concat_chunks)
            concat_op = TensorConcatenate(axis=concat_axis, dtype=to_concat_chunks[0].dtype,
                                          sparse=to_concat_chunks[0].issparse())
            concat_chunk = concat_op.new_chunk(to_concat_chunks, shape=tuple(concat_chunk_shape),
                                               index=out_idx)
            select_pos = calc_pos(fancy_indexes[0].shape, self._fancy_index_info.chunk_index_to_pos)
            out_index_obj = [slice(None)] * concat_axis + [select_pos] + \
                            [slice(None)] * (len(self._nsplits) - concat_axis - 1)
            out_chunk_op = TensorIndex(dtype=concat_chunk.dtype, sparse=concat_chunk.issparse(),
                                       indexes=out_index_obj)
            pos_select_shape = concat_chunk.shape[:concat_axis] + fancy_indexes[0].shape + \
                concat_chunk.shape[concat_axis + 1:]
            pos_select_idx = out_idx[:concat_axis] + (0,) * fancy_indexes[0].ndim + \
                out_idx[concat_axis + 1:]
            pos_select_chunk = out_chunk_op.new_chunk([concat_chunk], shape=pos_select_shape,
                                                      index=pos_select_idx)
            out_chunks.append(pos_select_chunk)

        self._out_chunks = out_chunks
        self._nsplits = self._nsplits[:concat_axis] + [(s,) for s in fancy_indexes[0].shape] + \
            self._nsplits[concat_axis + 1:]

    def _postprocess_tensor_fancy_index(self, fancy_index_infos):
        concat_axis = fancy_index_infos[0].out_axis
        chunk_shape = self._chunk_shape
        fancy_indexes = self._fancy_index_info.chunk_unified_fancy_indexes
        concat_idx_to_map_chunks = dict()
        for c in self._out_chunks:
            pos_idx = np.unravel_index(c.index[concat_axis],
                                       tuple(self._in_tensor.chunk_shape[ax]
                                             for ax in self._fancy_index_info.fancy_index_in_axes))
            pos_chunk = self._fancy_index_info.chunk_index_to_pos[pos_idx]
            concat_map_op = FancyIndexingConcatMap(fancy_index_axis=concat_axis,
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
                concat_reduce_op = FancyIndexingConcatReduce(fancy_index_axis=concat_axis,
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

        self._out_chunks = out_chunks
        self._nsplits = self._nsplits[:concat_axis] + list(fancy_indexes[0].nsplits) + \
            self._nsplits[concat_axis + 1:]

    def _postprocess_fancy_index(self):
        if not self._fancy_index_info:
            return

        fancy_index_infos = [info for info in self._index_infos
                             if info.index_type == IndexType.fancy_index]
        if self._fancy_index_info.fancy_index_all_ndarray:
            self._postprocess_ndarray_fancy_index(fancy_index_infos)
        else:
            self._postprocess_tensor_fancy_index(fancy_index_infos)

    def __call__(self):
        self._extract_indexes_info()
        self._preprocess_fancy_indexes()
        self._process_fancy_indexes()
        self._process_in_tensor()
        self._postprocess_fancy_index()

        new_op = self._op.copy()
        new_tensor = new_op.new_tensor(self._op.inputs, self._op.outputs[0].shape,
                                       chunks=self._out_chunks, nsplits=self._nsplits)
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
