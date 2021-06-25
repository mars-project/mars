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

import inspect
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from enum import Enum
from operator import itemgetter
from typing import Tuple, List, Union
from numbers import Integral

import numpy as np

from ...core import Tileable, recursive_tile
from ...core.operand import OperandStage
from ...utils import calc_nsplits, has_unknown_shape
from ..core import TENSOR_TYPE, Chunk, TensorOrder
from ..operands import TensorShuffleProxy
from ..utils import slice_split, calc_sliced_size, broadcast_shape, unify_chunks, \
    split_indexes_into_chunks, filter_inputs, calc_pos


class IndexType(Enum):
    new_axis = 0
    slice = 1
    label_slice = 2  # e.g. 'a': 'd' used for pandas etc
    integer = 3
    label = 4  # e.g. 'a' used for pandas etc
    bool_index = 5
    fancy_index = 6
    label_fancy_index = 7  # e.g. ['a', 'b', 'c'] for pandas etc


class IndexInfo:
    def __init__(self,
                 index_type: IndexType,
                 input_axis: int,
                 output_axis: int,
                 raw_index,
                 handler):
        self.index_type = index_type
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.raw_index = raw_index
        self.handler = handler


class FancyIndexInfo(IndexInfo):
    def __init__(self,
                 index_type: IndexType,
                 input_axis: int,
                 output_axis: int,
                 raw_index,
                 handler):
        super().__init__(index_type, input_axis, output_axis,
                         raw_index, handler)

        # extra info for fancy index
        # shape broadcast index
        self.shape_unified_index = None
        # split info
        #   - chunk_index_to_fancy_index_arrays
        #   - chunk_index_to_raw_positions
        #   - is_fancy_index_asc_sorted
        self.split_info = None


ChunkIndexAxisInfo = namedtuple(
    'chunk_index_axis_info',
    ['output_axis_index', 'processed_index', 'output_shape'])


class ChunkIndexInfo:
    def __init__(self):
        self.indexes = []
        self.output_chunk_index = []
        self.output_chunk_shape = []

    def set(self, info: ChunkIndexAxisInfo):
        output_axis_index = info.output_axis_index
        if output_axis_index is not None:
            self.output_chunk_index.append(output_axis_index)
        self.indexes.append(info.processed_index)
        output_shape = info.output_shape
        if output_shape is not None:
            if not isinstance(output_shape, tuple):
                self.output_chunk_shape.append(output_shape)
            else:
                self.output_chunk_shape.extend(output_shape)


class IndexHandlerContext(ABC):
    def __init__(self, op):
        self.parsed_infos = []
        self.input_axis = 0
        self.output_axis = 0

        # store index_type -> positions
        # for a quick search on indexes of a specified index type
        self._index_type_to_positions = dict()

        # store chunk index -> ChunkIndexInfo
        # for the IndexHandler to process
        self.chunk_index_to_info = OrderedDict()
        self.op = op
        self.tileable = op.input
        self.set_tileable(self.tileable)

        # chunks and nsplits, used for store intermediate result
        self.processed_chunks = None
        self.out_chunks = None
        self.out_nsplits = None

    def append(self, index_info: IndexInfo):
        position = len(self.parsed_infos)
        if index_info.index_type not in self._index_type_to_positions:
            self._index_type_to_positions[index_info.index_type] = []
        self._index_type_to_positions[index_info.index_type].append(position)
        self.parsed_infos.append(index_info)

    def get_positions(self, index_type: IndexType) -> List[int]:
        return self._index_type_to_positions.get(index_type, [])

    def get_indexes(self, index_type: IndexType):
        return [self.parsed_infos[i] for i in self.get_positions(index_type)]

    def set_tileable(self, tileable: Tileable):
        for chunk in tileable.chunks:
            self.chunk_index_to_info[chunk.index] = ChunkIndexInfo()

    @abstractmethod
    def concat_chunks(self,
                      chunks: List[Chunk],
                      axis: Union[Tuple, int]) -> Chunk:
        pass

    @abstractmethod
    def create_chunk(self,
                     chunk_index: Tuple[int],
                     chunk_index_info: ChunkIndexInfo) -> Chunk:
        pass

    def create_tileable(self) -> Tileable:
        out = self.op.outputs[0]
        params = out.params
        params['chunks'] = self.out_chunks
        params['nsplits'] = self.out_nsplits
        if 'shape' in params and any(np.isnan(s) for s in params['shape']):
            params['shape'] = tuple(sum(ns) for ns in self.out_nsplits)
        new_op = out.op.copy()
        return new_op.new_tileable(out.inputs, kws=[params])


class TensorIndexHandlerContext(IndexHandlerContext):
    def concat_chunks(self,
                      chunks: List[Chunk],
                      axis: Union[Tuple[int], int]) -> Chunk:
        from ..merge import TensorConcatenate

        assert isinstance(axis, int), \
            'axis to concat could only be int for tensor'

        shape = list(chunks[0].shape)
        shape[axis] = sum(c.shape[axis] for c in chunks)
        chunk_index = list(chunks[0].index)
        chunk_index[axis] = 0

        op = TensorConcatenate(axis=axis, dtype=chunks[0].dtype,
                               sparse=chunks[0].issparse())
        return op.new_chunk(chunks, shape=tuple(shape),
                            index=tuple(chunk_index),
                            order=TensorOrder.C_ORDER)

    def create_chunk(self,
                     chunk_index: Tuple[int],
                     chunk_index_info: ChunkIndexInfo) -> Chunk:
        chunk_op = self.op.copy().reset_key()
        chunk_op._indexes = indexes = chunk_index_info.indexes
        chunk_input = self.tileable.chunks[0] if self.tileable.ndim == 0 else \
            self.tileable.cix[chunk_index]
        chunk_inputs = filter_inputs([chunk_input] + indexes)
        return chunk_op.new_chunk(chunk_inputs,
                                  shape=tuple(chunk_index_info.output_chunk_shape),
                                  index=tuple(chunk_index_info.output_chunk_index),
                                  order=self.op.outputs[0].order)


_type_to_instance = {}


class IndexHandler(ABC):
    @classmethod
    def get_instance(cls):
        if cls not in _type_to_instance:
            _type_to_instance[cls] = cls()
        return _type_to_instance[cls]

    @abstractmethod
    def accept(cls, raw_index):
        pass

    @abstractmethod
    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        pass

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        pass

    @abstractmethod
    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        pass

    def postprocess(self,
                    index_info: IndexInfo,
                    context: IndexHandlerContext) -> None:
        pass

    @classmethod
    def set_chunk_index_info(cls,
                             context: IndexHandlerContext,
                             index_info: IndexInfo,
                             chunk_index: Tuple[int],
                             chunk_index_info: ChunkIndexInfo,
                             output_axis_index: int,
                             index,
                             output_shape: int):
        _ = context, index_info, chunk_index
        chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                                processed_index=index,
                                                output_shape=output_shape))


class NewaxisIndexHandler(IndexHandler):
    def accept(self, raw_index):
        return raw_index is np.newaxis

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = IndexInfo(IndexType.new_axis,
                         context.input_axis,
                         context.output_axis,
                         raw_index,
                         self)
        context.output_axis += 1
        context.append(info)
        return info

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        for chunk_index_info in context.chunk_index_to_info.values():
            # index on axis and index object
            chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=0,
                                                    processed_index=None,
                                                    output_shape=1))


class SliceIndexHandler(IndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, slice)

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = IndexInfo(IndexType.slice,
                         context.input_axis,
                         context.output_axis,
                         raw_index,
                         self)
        context.input_axis += 1
        context.output_axis += 1
        context.append(info)
        return info

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        # make sure input tileable has known chunk shapes
        if has_unknown_shape(context.tileable):
            yield []

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        # slice.step < 0
        is_reversed = (index_info.raw_index.step or 0) < 0

        # e.g. slice_split(slice(3, 10), [2, 2, 7, 5])
        # return {1: slice(1, 2, 1), 2: slice(0, 6, 1)}
        effected_i_to_slice = slice_split(
            index_info.raw_index, tileable.nsplits[index_info.input_axis])
        output_axis_index_range = range(len(effected_i_to_slice)) if not is_reversed else \
            range(len(effected_i_to_slice) - 1, -1, -1)
        other_index_to_iter = dict()

        index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in index_to_info.items():
            i = chunk_index[input_axis]
            other_index = chunk_index[:input_axis] + chunk_index[input_axis + 1:]
            size = tileable.nsplits[input_axis][i]
            if i not in effected_i_to_slice:
                # delete it, the input chunk could be ignored
                del context.chunk_index_to_info[chunk_index]
            else:
                slc = effected_i_to_slice[i]
                output_shape = calc_sliced_size(size, slc)
                if other_index not in other_index_to_iter:
                    other_index_to_iter[other_index] = iter(output_axis_index_range)
                output_axis_index = next(other_index_to_iter[other_index])
                self.set_chunk_index_info(context, index_info, chunk_index, chunk_index_info,
                                          output_axis_index, slc, output_shape)


class IntegralIndexHandler(IndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, Integral)

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = IndexInfo(IndexType.integer,
                         context.input_axis,
                         context.output_axis,
                         raw_index,
                         self)
        context.input_axis += 1
        context.append(info)
        return info

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        if has_unknown_shape(context.tileable):
            yield []

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis

        # e.g. slice_split(6, [2, 2, 7, 5])
        # return {2: 2}
        effected_i_to_slice = slice_split(
            index_info.raw_index, tileable.nsplits[index_info.input_axis])

        index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in index_to_info.items():
            i = chunk_index[input_axis]
            if i not in effected_i_to_slice:
                # delete it, the input chunk could be ignored
                del context.chunk_index_to_info[chunk_index]
            else:
                slc = effected_i_to_slice[i]
                chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=None,
                                                        processed_index=slc,
                                                        output_shape=None))


class _BoolIndexHandler(IndexHandler):
    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = IndexInfo(IndexType.bool_index,
                         context.input_axis,
                         context.output_axis,
                         raw_index,
                         self)
        context.input_axis += raw_index.ndim
        context.output_axis += 1
        context.append(info)
        return info

    @classmethod
    def _is_first_bool_index(self,
                             context: IndexHandlerContext,
                             index_info: IndexInfo) -> bool:
        bool_index_infos = [info for info in context.parsed_infos
                            if info.index_type == IndexType.bool_index]
        return bool_index_infos[0] is index_info


class NDArrayBoolIndexHandler(_BoolIndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, np.ndarray) and \
               raw_index.dtype == np.bool_

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        if has_unknown_shape(context.tileable):
            yield []

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        is_first_bool_index = self._is_first_bool_index(context, index_info)

        axes = list(range(input_axis, input_axis + index_info.raw_index.ndim))
        cum_sizes = []
        for axis in axes:
            cum_sizes.append(np.cumsum((0,) + tileable.nsplits[axis]))

        other_index_to_iter = dict()
        for chunk_index, chunk_index_info in context.chunk_index_to_info.items():
            slcs = []
            for j, axis in enumerate(axes):
                axis_index = chunk_index[axis]
                slcs.append(slice(cum_sizes[j][axis_index],
                                  cum_sizes[j][axis_index + 1]))
            other_index = chunk_index[:axes[0]] + chunk_index[axes[-1] + 1:]
            if other_index not in other_index_to_iter:
                other_index_to_iter[other_index] = itertools.count()
            index = index_info.raw_index[tuple(slcs)]
            output_axis_index = next(other_index_to_iter[other_index])

            # if more than 1 bool index, getitem will rewrite them into fancy
            # but for now, setitem will keep them, thus we cannot record
            # index or shape for this one
            output_axis_index = None if not is_first_bool_index else output_axis_index
            output_size = None if not is_first_bool_index else int(index.sum())

            self.set_chunk_index_info(context, index_info, chunk_index,
                                      chunk_index_info, output_axis_index,
                                      index, output_size)


class TensorBoolIndexHandler(_BoolIndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, TENSOR_TYPE) and \
               raw_index.dtype == np.bool_

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        # check both input tileable and index object itself
        if has_unknown_shape(context.tileable):
            yield []
        if has_unknown_shape(index_info.raw_index):
            yield []

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        index = index_info.raw_index
        # rechunk index into the same chunk size
        nsplits = tileable.nsplits[input_axis: input_axis + index.ndim]
        index = yield from recursive_tile(index.rechunk(nsplits))
        is_first_bool_index = self._is_first_bool_index(context, index_info)

        other_index_to_iter = dict()
        for chunk_index, chunk_index_info in context.chunk_index_to_info.items():
            effected_chunk_index = chunk_index[input_axis: input_axis + index.ndim]
            other_index = chunk_index[:input_axis] + chunk_index[input_axis + index.ndim:]
            if other_index not in other_index_to_iter:
                other_index_to_iter[other_index] = itertools.count()
            output_axis_index = next(other_index_to_iter[other_index])

            # if more than 1 bool index, getitem will rewrite them into fancy
            # but for now, setitem will keep them, thus we cannot record
            # index or shape for this one
            output_axis_index = None if not is_first_bool_index else output_axis_index
            output_size = None if not is_first_bool_index else np.nan

            self.set_chunk_index_info(context, index_info, chunk_index,
                                      chunk_index_info, output_axis_index,
                                      index.cix[tuple(effected_chunk_index)],
                                      output_size)


class _FancyIndexHandler(IndexHandler):
    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        prev_fancy_indexes = context.get_indexes(IndexType.fancy_index)
        is_first_fancy_index = len(prev_fancy_indexes) == 0

        if is_first_fancy_index:
            output_axis = context.output_axis
        else:
            output_axis = prev_fancy_indexes[0].output_axis
        info = FancyIndexInfo(IndexType.fancy_index,
                              context.input_axis,
                              output_axis,
                              raw_index,
                              self)

        context.input_axis += 1
        if is_first_fancy_index:
            context.output_axis += 1
        context.append(info)
        return info

    @classmethod
    def is_first(cls, index_info: IndexInfo, context: IndexHandlerContext) -> bool:
        # check if is first fancy index after parsing
        fancy_indexes = context.get_indexes(index_info.index_type)
        i = fancy_indexes.index(index_info)
        if i > 0:
            # only process for the first fancy indexes
            return False
        else:
            return True

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        fancy_indexe_infos = context.get_indexes(index_info.index_type)
        # check all fancy indexes are all ndarrays
        for fancy_index_info in fancy_indexe_infos:
            if not self.accept(fancy_index_info.raw_index):  # pragma: no cover
                raise TypeError('Fancy indexes should be all ndarrays or tensors')


class NDArrayFancyIndexHandler(_FancyIndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, np.ndarray) and \
               raw_index.dtype != np.bool_

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        is_first = self.is_first(index_info, context)
        if not is_first:
            return

        # check if all ndarrays
        super().preprocess(index_info, context)
        if has_unknown_shape(context.tileable):
            yield []

        fancy_index_infos = context.get_indexes(index_info.index_type)
        # unify shapes of all fancy indexes
        shape = broadcast_shape(*(info.raw_index.shape
                                  for info in fancy_index_infos))
        for fancy_index_info in fancy_index_infos:
            fancy_index_info.shape_unified_index = np.broadcast_to(
                fancy_index_info.raw_index, shape)

        # concat all fancy index together
        concat_fancy_index = np.stack(
            [info.shape_unified_index.ravel() for info in fancy_index_infos])
        effected_nsplits = [context.tileable.nsplits[info.input_axis]
                            for info in fancy_index_infos]
        # split concatenated fancy index into chunks according to input tileable
        split_info = split_indexes_into_chunks(effected_nsplits, concat_fancy_index)
        fancy_index_infos[0].split_info = split_info

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        fancy_index_infos = context.get_indexes(index_info.index_type)
        fancy_index_axes = [info.input_axis for info in fancy_index_infos]
        split_info = fancy_index_infos[0].split_info
        chunk_index_to_fancy_index_arrays = split_info[0]
        i_fancy_index = fancy_index_infos.index(index_info)

        other_index_to_iter = dict()
        chunk_index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in chunk_index_to_info.items():
            effected_chunk_index = tuple(chunk_index[ax] for ax in fancy_index_axes)
            fancy_index_array = \
                chunk_index_to_fancy_index_arrays[effected_chunk_index][i_fancy_index]

            if fancy_index_array.size == 0:
                # not effected
                del context.chunk_index_to_info[chunk_index]
                continue

            if i_fancy_index == 0:
                other_index = tuple(ci for i, ci in enumerate(chunk_index)
                                    if i not in fancy_index_axes)
                if other_index not in other_index_to_iter:
                    other_index_to_iter[other_index] = itertools.count()
                output_axis_index = next(other_index_to_iter[other_index])
                output_axis_shape = fancy_index_array.shape[0]
            else:
                output_axis_index = None
                output_axis_shape = None

            chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                                    processed_index=fancy_index_array,
                                                    output_shape=output_axis_shape))

    @classmethod
    def need_postprocess(cls, context: IndexHandlerContext) -> bool:
        fancy_indexes = context.get_indexes(IndexType.fancy_index)

        if fancy_indexes[0].split_info[2] and \
                fancy_indexes[0].shape_unified_index.ndim == 1:
            # if fancy indexes are asc sorted,
            # and they are 1-d, no further computation required
            return False

        return True

    def postprocess(self,
                    index_info: IndexInfo,
                    context: IndexHandlerContext) -> None:
        fancy_indexes = context.get_indexes(index_info.index_type)

        if not self.need_postprocess(context):
            return

        is_first = self.is_first(index_info, context)
        if not is_first:
            # only need to postprocess fancy indexes once
            return

        # current chunks and nsplits
        chunks, nsplits = context.out_chunks, context.out_nsplits

        index_to_chunks = {c.index: c for c in chunks}
        fancy_index_shape = fancy_indexes[0].shape_unified_index.shape
        reorder_index = calc_pos(fancy_index_shape, fancy_indexes[0].split_info[1])

        to_concat_axis = index_info.output_axis
        new_out_chunks = []
        for chunk_index in itertools.product(
                *(range(len(ns)) for ax, ns in enumerate(nsplits)
                  if ax != to_concat_axis)):
            # concat chunks on output axis of first fancy index
            to_concat_chunks = []
            for i in range(len(nsplits[to_concat_axis])):
                to_concat_index = list(chunk_index)
                to_concat_index.insert(to_concat_axis, i)
                to_concat_chunks.append(index_to_chunks[tuple(to_concat_index)])
            concat_chunk = context.concat_chunks(to_concat_chunks, to_concat_axis)

            reorder_chunk_op = context.op.copy().reset_key()
            reorder_chunk_op._indexes = [slice(None)] * to_concat_axis + [reorder_index]
            reorder_shape = concat_chunk.shape[:to_concat_axis] + fancy_index_shape + \
                concat_chunk.shape[to_concat_axis + 1:]
            chunk_reorder_index = concat_chunk.index[:to_concat_axis] + \
                (0,) * len(fancy_index_shape) + concat_chunk.index[to_concat_axis + 1:]
            reorder_chunk = reorder_chunk_op.new_chunk([concat_chunk],
                                                       shape=reorder_shape,
                                                       index=chunk_reorder_index,
                                                       order=TensorOrder.C_ORDER)
            new_out_chunks.append(reorder_chunk)

        new_nsplits = nsplits[:to_concat_axis] + tuple((s,) for s in fancy_index_shape) \
            + nsplits[to_concat_axis + 1:]
        context.out_chunks = new_out_chunks
        context.out_nsplits = new_nsplits


class TensorFancyIndexHandler(_FancyIndexHandler):
    def accept(self, raw_index):
        return isinstance(raw_index, TENSOR_TYPE) and \
               raw_index.dtype != np.bool_

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        from ..base import broadcast_to
        from ..merge import stack

        is_first = self.is_first(index_info, context)
        if not is_first:
            return

        fancy_index_infos = context.get_indexes(index_info.index_type)

        # check if all tensors
        super().preprocess(index_info, context)
        to_check = [context.tileable] + \
            list(info.raw_index for info in fancy_index_infos)
        if has_unknown_shape(*to_check):
            yield

        # unify shapes of all fancy indexes
        shape = broadcast_shape(
            *(info.raw_index.shape for info in fancy_index_infos))
        fancy_indexes = []
        for fancy_index_info in fancy_index_infos:
            fancy_index = yield from recursive_tile(
                broadcast_to(fancy_index_info.raw_index, shape))
            fancy_indexes.append(fancy_index)
        shape_unified_fancy_indexes = yield from unify_chunks(*fancy_indexes)
        for fancy_index_info, shape_unified_fancy_index in \
                zip(fancy_index_infos, shape_unified_fancy_indexes):
            fancy_index_info.shape_unified_index = shape_unified_fancy_index

        fancy_index_axes = tuple(info.input_axis for info in fancy_index_infos)

        # stack fancy indexes into one
        concat_fancy_index = yield from recursive_tile(
            stack([fancy_index_info.shape_unified_index
                   for fancy_index_info in fancy_index_infos]))
        concat_fancy_index = \
            yield from recursive_tile(
                concat_fancy_index.rechunk({0: len(fancy_index_infos)}))

        self._shuffle_fancy_indexes(concat_fancy_index, context,
                                    index_info, fancy_index_axes)

    @classmethod
    def _shuffle_fancy_indexes(cls,
                               concat_fancy_index: Tileable,
                               context: IndexHandlerContext,
                               index_info: IndexInfo,
                               axes: Tuple):
        from .getitem import FancyIndexingDistribute

        tileable = context.tileable

        # generate shuffle map, for concatenated fancy index,
        # calculated a counterpart index chunk for each chunk of input tensor
        map_chunks = []
        for chunk in concat_fancy_index.chunks:
            map_op = FancyIndexingDistribute(
                stage=OperandStage.map, dest_nsplits=tileable.nsplits,
                axes=axes, dtype=chunk.dtype)
            map_chunk = map_op.new_chunk([chunk], shape=(np.nan,),
                                         index=chunk.index,
                                         order=TensorOrder.C_ORDER)
            map_chunks.append(map_chunk)
        # shuffle proxy
        proxy_chunk = TensorShuffleProxy(dtype=concat_fancy_index.dtype).new_chunk(
            map_chunks, shape=(), order=TensorOrder.C_ORDER)
        chunk_index_to_fancy_index_chunks = OrderedDict()
        chunk_index_to_raw_positions = OrderedDict()
        for chunk_index in itertools.product(
                *(range(tileable.chunk_shape[ax]) for ax in axes)):
            reduce_op = FancyIndexingDistribute(
                stage=OperandStage.reduce, axes=axes, dtype=proxy_chunk.dtype)
            # chunks of fancy indexes on each axis
            kws = [{'axis': ax, 'shape': (np.nan,), 'index': chunk_index,
                    'order': context.op.outputs[0].order}
                   for ax in axes]
            kws.append({'pos': True, 'shape': (np.nan,), 'index': chunk_index})
            reduce_chunks = reduce_op.new_chunks([proxy_chunk], kws=kws)
            chunk_index_to_fancy_index_chunks[chunk_index] = reduce_chunks[:-1]
            chunk_index_to_raw_positions[chunk_index] = reduce_chunks[-1]

        # split info
        #   - chunk_index_to_fancy_index_chunks
        #   - chunk_index_to_raw_positions
        #   - is_fancy_index_asc_sorted, False for tensor fancy indexes
        index_info.split_info = chunk_index_to_fancy_index_chunks, \
            chunk_index_to_raw_positions, False

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        fancy_index_infos = context.get_indexes(index_info.index_type)
        fancy_index_axes = [info.input_axis for info in fancy_index_infos]
        split_info = fancy_index_infos[0].split_info
        chunk_index_to_fancy_index_chunks = split_info[0]
        i_fancy_index = fancy_index_infos.index(index_info)

        other_index_to_iter = dict()
        for chunk_index, chunk_index_info in context.chunk_index_to_info.items():
            effected_chunk_index = tuple(chunk_index[ax] for ax in fancy_index_axes)
            fancy_index_chunk = \
                chunk_index_to_fancy_index_chunks[effected_chunk_index][i_fancy_index]

            if i_fancy_index == 0:
                other_index = tuple(ci for i, ci in enumerate(chunk_index)
                                    if i not in fancy_index_axes)
                if other_index not in other_index_to_iter:
                    other_index_to_iter[other_index] = itertools.count()
                output_axis_index = next(other_index_to_iter[other_index])
                output_axis_shape = fancy_index_chunk.shape[0]
            else:
                output_axis_index = output_axis_shape = None

            chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                                    processed_index=fancy_index_chunk,
                                                    output_shape=output_axis_shape))

    def postprocess(self,
                    index_info: IndexInfo,
                    context: IndexHandlerContext) -> None:
        from .getitem import FancyIndexingConcat

        fancy_index_infos = context.get_indexes(index_info.index_type)

        is_first = self.is_first(index_info, context)
        if not is_first:
            # only need to postprocess fancy indexes once
            return

        # current chunks and nsplits
        chunks, nsplits = context.out_chunks, context.out_nsplits
        chunk_shape = tuple(len(ns) for ns in nsplits)
        to_concat_axis = index_info.output_axis
        tileable = context.tileable
        fancy_index_effected_input_chunk_shapes = tuple(
            tileable.chunk_shape[info.input_axis] for info in fancy_index_infos)
        fancy_indexes = [info.shape_unified_index for info in fancy_index_infos]

        concat_index_to_chunks = dict()
        for chunk in chunks:
            effected_chunk_index = np.unravel_index(
                chunk.index[to_concat_axis], fancy_index_effected_input_chunk_shapes)
            raw_position_chunk = fancy_index_infos[0].split_info[1][effected_chunk_index]
            concat_map_op = FancyIndexingConcat(stage=OperandStage.map,
                                                fancy_index_axis=to_concat_axis,
                                                sparse=chunk.issparse(),
                                                dtype=chunk.dtype)
            map_chunk_shape = \
                chunk.shape[:to_concat_axis] + (np.nan,) + chunk.shape[to_concat_axis + 1:]
            concat_map_chunk = concat_map_op.new_chunk(
                [chunk, raw_position_chunk], index=chunk.index,
                shape=map_chunk_shape, order=TensorOrder.C_ORDER)
            concat_index_to_chunks[concat_map_chunk.index] = concat_map_chunk

        other_index_chunk_shape = chunk_shape[:to_concat_axis] + chunk_shape[to_concat_axis + 1:]
        out_chunks = []
        for chunk_index in itertools.product(*(range(s) for s in other_index_chunk_shape)):
            to_shuffle_chunks = []
            other_shape = None
            for i in range(chunk_shape[to_concat_axis]):
                to_concat_chunk_index = \
                    chunk_index[:to_concat_axis] + (i,) + chunk_index[to_concat_axis:]
                to_concat_chunk = concat_index_to_chunks[to_concat_chunk_index]
                to_shuffle_chunks.append(to_concat_chunk)
                if other_shape is None:
                    other_shape = tuple(s for ax, s in enumerate(to_concat_chunk.shape)
                                        if ax != to_concat_axis)

            proxy_chunk = TensorShuffleProxy(dtype=to_shuffle_chunks[0].dtype).new_chunk(
                to_shuffle_chunks, shape=(), order=TensorOrder.C_ORDER)

            it = itertools.count()
            for reduce_index in itertools.product(
                    *(range(s) for s in fancy_indexes[0].chunk_shape)):
                fancy_index_chunk = fancy_indexes[0].cix[reduce_index]
                concat_reduce_op = FancyIndexingConcat(
                    stage=OperandStage.reduce, fancy_index_axis=to_concat_axis,
                    fancy_index_shape=fancy_index_chunk.shape,
                    dtype=proxy_chunk.dtype, sparse=to_shuffle_chunks[0].issparse(),
                    reducer_index=(next(it),))
                reduce_chunk_shape = other_shape[:to_concat_axis] + \
                    fancy_index_chunk.shape + other_shape[to_concat_axis:]
                reduce_chunk_index = chunk_index[:to_concat_axis] + \
                    fancy_index_chunk.index + chunk_index[to_concat_axis:]
                concat_reduce_chunk = concat_reduce_op.new_chunk(
                    [proxy_chunk], shape=reduce_chunk_shape, index=reduce_chunk_index,
                    order=TensorOrder.C_ORDER)
                out_chunks.append(concat_reduce_chunk)

        context.out_chunks = out_chunks
        context.out_nsplits = nsplits[:to_concat_axis] + \
            fancy_indexes[0].nsplits + nsplits[to_concat_axis + 1:]


class IndexesHandler(ABC):
    def __init__(self):
        self.available_index_handlers = []

    def register(self, *handlers):
        self.available_index_handlers.extend(
            h.get_instance() for h in handlers)

    @abstractmethod
    def create_context(self, op):
        pass

    def handle(self, op, return_context: bool = False):
        indexes = op.indexes
        # create context
        context = self.create_context(op)

        # parse index infos
        index_infos = []
        for index in indexes:
            parsed = False
            for index_handler in self.available_index_handlers:
                if index_handler.accept(index):
                    parsed = True
                    index_infos.append(
                        index_handler.parse(index, context))
                    break
            if not parsed:
                raise TypeError(f'unable to parse index {index}')

        yield from self._preprocess(context, index_infos)
        yield from self._process(context, index_infos)
        self._postprocess(context, index_infos)

        if return_context:
            return context
        else:
            return context.create_tileable()

    @classmethod
    def _preprocess(cls,
                    context: IndexHandlerContext,
                    index_infos: List[IndexInfo]):
        # preprocess
        for index_info in index_infos:
            preprocess = index_info.handler.preprocess(index_info, context)
            if inspect.isgenerator(preprocess):
                yield from preprocess

    @classmethod
    def _process(cls, context, index_infos):
        # process
        for index_info in index_infos:
            process = index_info.handler.process(index_info, context)
            if inspect.isgenerator(process):
                yield from process

        context.processed_chunks = context.out_chunks = out_chunks = []
        for chunk_index, chunk_index_info in context.chunk_index_to_info.items():
            out_chunks.append(context.create_chunk(chunk_index, chunk_index_info))
        index_to_shape = OrderedDict(sorted([(c.index, c.shape) for c in out_chunks],
                                            key=itemgetter(0)))
        context.out_nsplits = calc_nsplits(index_to_shape)

    @classmethod
    def _postprocess(cls, context, index_infos):
        # post process
        for index_info in index_infos:
            index_info.handler.postprocess(index_info, context)


class NDArrayIndexesHandler(IndexesHandler):
    # indexes handler only for slice, integer,
    # boolean ndarray, integer ndarray and None
    def __init__(self):
        super().__init__()
        self.register(NewaxisIndexHandler,
                      SliceIndexHandler,
                      IntegralIndexHandler,
                      NDArrayBoolIndexHandler,
                      NDArrayFancyIndexHandler)

    def create_context(self, op):
        return TensorIndexHandlerContext(op)


class TensorIndexesHandler(IndexesHandler):
    def __init__(self):
        super().__init__()
        self.register(NewaxisIndexHandler,
                      SliceIndexHandler,
                      IntegralIndexHandler,
                      NDArrayBoolIndexHandler,
                      TensorBoolIndexHandler,
                      NDArrayFancyIndexHandler,
                      TensorFancyIndexHandler)

    def create_context(self, op):
        return TensorIndexHandlerContext(op)
