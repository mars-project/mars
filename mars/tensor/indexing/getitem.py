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

from ... import opcodes as OperandDef
from ...serialize import ValueType, KeyField, ListField, TupleField, Int32Field
from ...core import Base, Entity
from ...compat import OrderedDict, Enum, reduce
from ..core import TENSOR_TYPE, TensorOrder
from ..utils import unify_chunks, slice_split, split_indexes_into_chunks, \
    calc_pos, broadcast_shape, calc_sliced_size, recursive_tile, filter_inputs
from ..operands import TensorHasInput, TensorOperandMixin, \
    TensorShuffleMap, TensorShuffleReduce, TensorShuffleProxy
from ...utils import get_shuffle_input_keys_idxes
from ..array_utils import get_array_module
from .core import process_index, calc_shape

FANCY_INDEX_TYPES = TENSOR_TYPE + (np.ndarray,)


class TensorIndex(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEX

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, dtype=None, sparse=False, indexes=None, create_view=False, **kw):
        super(TensorIndex, self).__init__(_dtype=dtype, _sparse=sparse, _indexes=indexes,
                                          _create_view=create_view, **kw)

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super(TensorIndex, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes

    def on_output_modify(self, new_output):
        from .setitem import TensorIndexSetValue

        if self._create_view:
            a = self.input
            op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(),
                                     indexes=self._indexes, value=new_output)
            return op(a, self._indexes, new_output)

    def on_input_modify(self, new_input):
        if self._create_view:
            new_op = self.copy().reset_key()
            new_inputs = [new_input] + self.inputs[1:]
            return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    def __call__(self, a, index, shape, order):
        self._indexes = index
        return self.new_tensor(filter_inputs([a] + list(index)), shape, order=order)

    @classmethod
    def tile(cls, op):
        tile_handler = TensorIndexTilesHandler(op)
        return tile_handler()

    @classmethod
    def execute(cls, ctx, op):
        indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                        for index in op.indexes)
        input_ = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = input_[indexes].astype(
            input_.dtype, order=op.outputs[0].order.value, copy=False)

    @classmethod
    def estimate_size(cls, ctx, op):
        from mars.core import Base, Entity
        chunk = op.outputs[0]
        shape = chunk.shape
        new_indexes = [index for index in op._indexes if index is not None]

        new_shape = []
        first_fancy_index = False
        for index in new_indexes:
            if isinstance(index, (Base, Entity)):
                if index.dtype != np.bool_:
                    if not first_fancy_index:
                        first_fancy_index = True
                    else:
                        continue
                new_shape.append(ctx[index.key][0] // index.dtype.itemsize)

        rough_shape = []
        idx = 0
        for s in shape:
            if np.isnan(s):
                rough_shape.append(new_shape[idx])
                idx += 1
            else:
                rough_shape.append(s)
        result = int(np.prod(rough_shape) * chunk.dtype.itemsize)
        ctx[chunk.key] = (result, result)


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


def _is_bool_index(index_obj):
    return isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_


def _is_fancy_index(index_obj):
    return isinstance(index_obj, FANCY_INDEX_TYPES) and index_obj.dtype != np.bool_


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

    def _extract_indexes_info(self):
        in_axis = out_axis = 0
        fancy_index_out_axis = None
        for raw_index_obj in self._op.indexes:
            if _is_bool_index(raw_index_obj):
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
            elif _is_fancy_index(raw_index_obj):
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

        if len(self._in_tensor.chunks) == 1 and \
                all(len(idx.chunks) == 1 for idx in fancy_indexes):
            # only 1 chunk for input tensor and fancy indexing tensors,
            # thus no need to do shuffle
            chunks = [info.raw_index_obj.chunks[0] for info in self._fancy_index_infos]
            for fancy_index_info in self._fancy_index_infos:
                fancy_index_info.index_obj = {(0,) * len(axes): chunks}
            return

        # stack fancy indexes into one
        concat_fancy_index = recursive_tile(stack(fancy_indexes))
        concat_fancy_index = concat_fancy_index.rechunk({0: len(fancy_indexes)}).single_tiles()

        # generate shuffle map, for concatenated fancy index,
        # calculated a counterpart index chunk for each chunk of input tensor
        shuffle_map_chunks = []
        for chunk in concat_fancy_index.chunks:
            shuffle_map_op = FancyIndexingDistributeMap(
                dest_nsplits=self._in_tensor.nsplits, axes=axes, dtype=chunk.dtype)
            shuffle_map_chunk = shuffle_map_op.new_chunk([chunk], shape=(np.nan,),
                                                         index=chunk.index, order=TensorOrder.C_ORDER)
            shuffle_map_chunks.append(shuffle_map_chunk)
        # shuffle proxy
        proxy_chunk = TensorShuffleProxy(dtype=fancy_indexes[0].dtype, tensor_keys=[fancy_indexes[0].key]) \
            .new_chunk(shuffle_map_chunks, shape=(), order=TensorOrder.C_ORDER)
        chunk_index_to_fancy_indexes_chunks = OrderedDict()
        chunk_index_to_pos = OrderedDict()
        for idx in itertools.product(*(range(self._in_tensor.chunk_shape[axis]) for axis in axes)):
            shuffle_key = ','.join(str(i) for i in idx)
            shuffle_reduce_op = FancyIndexingDistributeReduce(axes=axes, dtype=proxy_chunk.dtype,
                                                              _shuffle_key=shuffle_key)
            # chunks of fancy indexes on each axis
            kws = [{'axis': ax, 'shape': (np.nan,), 'index': idx, 'order': self._op.outputs[0].order}
                   for ax in axes]
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
                    out_chunk_idx = sum(idx * reduce(operator.mul, cs[i + 1:], 1) for i, idx
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
                                           shape=tuple(chunk_shape), index=tuple(chunk_index),
                                           order=self._op.outputs[0].order)
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
                                               index=out_idx, order=TensorOrder.C_ORDER)
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
                                                      index=pos_select_idx, order=TensorOrder.C_ORDER)
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
                                                       index=c.index, order=TensorOrder.C_ORDER)
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
            proxy_chunk = proxy_op.new_chunk(to_shuffle_chunks, shape=(), order=TensorOrder.C_ORDER)
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
                                                                 index=reduce_chunk_idx,
                                                                 order=TensorOrder.C_ORDER)
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
                                       order=self._op.outputs[0].order,
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

    @classmethod
    def execute(cls, ctx, op):
        nsplits = op.dest_nsplits
        axes = op.axes
        fancy_index_nsplits = [nsplits[ax] for ax in axes]
        indexes = ctx[op.inputs[0].key]
        flatten_indexes = indexes.reshape(indexes.shape[0], -1)
        idx_to_fancy_indexes, idx_to_poses = \
            split_indexes_into_chunks(fancy_index_nsplits, flatten_indexes, False)
        for idx in idx_to_fancy_indexes:
            group_key = ','.join(str(i) for i in idx)
            ctx[(op.outputs[0].key, group_key)] = (idx_to_fancy_indexes[idx], idx_to_poses[idx])

    @classmethod
    def estimate_size(cls, ctx, op):
        fancy_index_size = len(op.axes)
        inp_size = ctx[op.inputs[0].key][0]
        factor = 1 / float(fancy_index_size) + fancy_index_size  # 1/#fancy_index is the poses
        ctx[op.outputs[0].key] = (inp_size * factor,) * 2


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

    @classmethod
    def execute(cls, ctx, op):
        in_chunk = op.inputs[0]

        input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)

        fancy_indexes = []
        poses = []
        shuffle_key = op.shuffle_key
        xp = None
        for input_key in input_keys:
            key = (input_key, shuffle_key)
            fancy_index, pos = ctx[key]
            if xp is None:
                xp = get_array_module(fancy_index)
            if fancy_index.size == 0:
                fancy_index = fancy_index.reshape(len(op.axes), 0)
            fancy_indexes.append(fancy_index)
            poses.append(pos)

        fancy_index = np.hstack(fancy_indexes)
        pos = np.hstack(poses)

        assert len(op.outputs) - 1 == len(fancy_index)
        for out_chunk, axis_fancy_index in zip(op.outputs[:-1], fancy_index):
            ctx[out_chunk.key] = axis_fancy_index
        ctx[op.outputs[-1].key] = np.asarray([len(p) for p in poses]), pos

    @classmethod
    def estimate_size(cls, ctx, op):
        sum_size = 0
        for shuffle_input in op.inputs[0].inputs or ():
            sum_size += ctx[shuffle_input.key]
        for out_chunk in op.outputs:
            ctx[out_chunk.key] = sum_size, sum_size


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

    @classmethod
    def execute(cls, ctx, op):
        indexed_array = ctx[op.inputs[0].key]
        sizes, pos = ctx[op.inputs[1].key]
        acc_sizes = np.cumsum(sizes)
        fancy_index_axis = op.fancy_index_axis

        for i in range(len(sizes)):
            start = 0 if i == 0 else acc_sizes[i - 1]
            end = acc_sizes[i]
            select = (slice(None),) * fancy_index_axis + (slice(start, end),)
            ctx[(op.outputs[0].key, str(i))] = (indexed_array[select], pos[start: end])

    @classmethod
    def estimate_size(cls, ctx, op):
        input_size = ctx[op.inputs[0].key][0]
        pos_size = ctx[op.inputs[0].key][0]
        ctx[op.outputs[0].key] = input_size + pos_size, input_size + pos_size * 2


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

    @classmethod
    def execute(cls, ctx, op):
        in_chunk = op.inputs[0]
        input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)
        fancy_index_axis = op.fancy_index_axis
        fancy_index_shape = op.fancy_index_shape

        indexed_arrays = []
        poses = []
        shuffle_key = op.shuffle_key
        for input_key in input_keys:
            index_array, pos = ctx[(input_key, shuffle_key)]
            indexed_arrays.append(index_array)
            poses.append(pos)

        concat_array = np.concatenate(indexed_arrays, axis=fancy_index_axis)
        concat_pos = np.hstack(poses)
        select_pos = calc_pos(fancy_index_shape, concat_pos)
        select = (slice(None),) * fancy_index_axis + (select_pos,)
        ctx[op.outputs[0].key] = concat_array[select]

    @classmethod
    def estimate_size(cls, ctx, op):
        chunk = op.outputs[0]
        input_sizes = [ctx[c.key][0] for c in op.inputs[0].inputs or ()]
        ctx[chunk.key] = chunk.nbytes, chunk.nbytes + sum(input_sizes)


def _is_create_view(index):
    # is view if all of index is slice, int or newaxis
    return all(isinstance(ind, (slice, Integral)) or ind is None for ind in index)


def _calc_order(a, index):
    if a.order == TensorOrder.C_ORDER:
        return TensorOrder.C_ORDER

    in_axis = 0
    for ind in index:
        if _is_bool_index(ind):
            in_axis += ind.ndim
            return TensorOrder.C_ORDER
        elif _is_fancy_index(ind):
            in_axis += 1
            return TensorOrder.C_ORDER
        elif ind is None:
            continue
        elif isinstance(ind, slice):
            shape = a.shape[in_axis]
            slc = ind.indices(shape)
            if slc[0] == 0 and slc[1] == shape and slc[2] == 1:
                continue
            else:
                return TensorOrder.C_ORDER
        else:
            assert isinstance(ind, Integral)
            in_axis += 1
            return TensorOrder.C_ORDER

    return TensorOrder.F_ORDER


def _getitem(a, item):
    if isinstance(item, (list, tuple)) and \
            all(isinstance(it, slice) and it == slice(None) for it in item):
        # nothing to do
        return a

    # TODO(jisheng): field access, e.g. t['a'], t[['a', 'b']]

    index = process_index(a.ndim, item)
    shape = calc_shape(a.shape, index)
    tensor_order = _calc_order(a, index)
    op = TensorIndex(dtype=a.dtype, sparse=a.issparse(), indexes=index,
                     create_view=_is_create_view(index))
    return op(a, index, tuple(shape), order=tensor_order)
