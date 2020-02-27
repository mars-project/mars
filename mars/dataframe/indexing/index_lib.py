# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np

from ...core import TileableEntity, Chunk
from ...tiles import TilesError
from ...tensor.indexing.index_lib import IndexHandlerContext, IndexHandler, \
    IndexInfo, IndexType, ChunkIndexInfo as ChunkIndexInfoBase, \
    SliceIndexHandler as SliceIndexHandlerBase, \
    NDArrayBoolIndexHandler as NDArrayBoolIndexHandlerBase, \
    TensorBoolIndexHandler as TensorBoolIndexHandlerBase, \
    IntegralIndexHandler, IndexesHandler
from ...tensor.utils import split_indexes_into_chunks, calc_pos, filter_inputs
from ...utils import check_chunks_unknown_shape
from ..core import SERIES_CHUNK_TYPE
from ..operands import ObjectType
from ..utils import parse_index


ChunkIndexAxisInfo = namedtuple(
    'chunk_index_axis_info',
    ['output_axis_index', 'processed_index', 'output_shape', 'index_value', 'dtypes'])


class ChunkIndexInfo(ChunkIndexInfoBase):
    def __init__(self):
        super().__init__()
        self.index_values = []
        self.dtypes = None

    def set(self, info: ChunkIndexAxisInfo):
        super().set(info)
        if getattr(info, 'index_value', None) is not None:
            self.index_values.append(info.index_value)
        if getattr(info, 'dtypes', None) is not None:
            self.dtypes = info.dtypes


class FancyIndexInfo(IndexInfo):
    def __init__(self,
                 index_type: IndexType,
                 input_axis: int,
                 output_axis: int,
                 raw_index,
                 handler):
        super().__init__(index_type, input_axis, output_axis,
                         raw_index, handler)

        # extra info for DataFrame fancy index
        # split info
        #   - chunk_index_to_fancy_index_arrays
        #   - chunk_index_to_raw_positions
        #   - is_fancy_index_asc_sorted
        self.split_info = None


class DataFrameIndexHandlerContext(IndexHandlerContext):
    def set_tileable(self, tileable: TileableEntity):
        for chunk in tileable.chunks:
            self.chunk_index_to_info[chunk.index] = ChunkIndexInfo()

    def concat_chunks(self,
                      chunks: List[Chunk],
                      axis: Union[Tuple[int], int]) -> Chunk:
        dataframe_op_type = type(chunks[0].op)
        # create tileable from chunks
        concat_tileable = \
            dataframe_op_type.create_tileable_from_chunks(chunks, inputs=chunks)
        # concat chunks
        chunk = dataframe_op_type.concat_tileable_chunks(concat_tileable).chunks[0]
        if chunk.ndim > 1 and \
                (isinstance(axis, tuple) and len(axis) == 1) or isinstance(axis, int):
            # adjust index and axis
            axis = axis[0] if isinstance(axis, tuple) else axis
            chunk.op._axis = axis
            chunk_index = list(chunk.index)
            chunk_index[1 - axis] = chunks[0].index[1 - axis]
            chunk._index = tuple(chunk_index)
        return chunk

    def create_chunk(self,
                     chunk_index: Tuple[int],
                     chunk_index_info: ChunkIndexInfo) -> Chunk:
        chunk_op = self.op.copy().reset_key()
        chunk_op._indexes = indexes = chunk_index_info.indexes

        chunk_input = self.tileable.cix[chunk_index]
        chunk_inputs = filter_inputs([chunk_input] + indexes)

        kw = {}
        kw['shape'] = shape = tuple(chunk_index_info.output_chunk_shape)
        kw['index'] = tuple(chunk_index_info.output_chunk_index)
        index_values = chunk_index_info.index_values
        if len(shape) == 0:
            # scalar
            chunk_op._object_type = ObjectType.scalar
            kw['dtype'] = self.op.outputs[0].dtype
        elif len(shape) == 1:
            # Series
            chunk_op._object_type = ObjectType.series
            kw['index_value'] = index_values[0]
            kw['dtype'] = self.op.outputs[0].dtype
        else:
            # dataframe
            chunk_op._object_type = ObjectType.dataframe
            kw['index_value'] = index_values[0]
            kw['columns_value'] = index_values[1]
            kw['dtypes'] = chunk_index_info.dtypes

        return chunk_op.new_chunk(chunk_inputs, kws=[kw])


class SliceIndexHandler(SliceIndexHandlerBase):
    @classmethod
    def set_chunk_index_info(cls,
                             context: IndexHandlerContext,
                             index_info: IndexInfo,
                             chunk_index: Tuple[int],
                             chunk_index_info: ChunkIndexInfo,
                             output_axis_index: int,
                             index,
                             output_shape: int):
        tileable = context.tileable
        chunk_input = tileable.cix[chunk_index]
        slc = index

        kw = {
            'output_axis_index': output_axis_index,
            'processed_index': slc,
            'output_shape': output_shape,
            'dtypes': None
        }
        if index_info.input_axis == 0:
            index = chunk_input.index_value.to_pandas()
            kw['index_value'] = parse_index(index[slc], chunk_input, slc,
                                            store_data=False)
        else:
            assert index_info.input_axis == 1
            index = chunk_input.columns_value.to_pandas()
            # do not store index value if output axis is 0
            store_data = True if index_info.output_axis == 1 else False
            kw['index_value'] = parse_index(index[slc], store_data=store_data)
            kw['dtypes'] = chunk_input.dtypes[slc]

        chunk_index_info.set(ChunkIndexAxisInfo(**kw))


class DataFrameIndexHandler:
    @classmethod
    def set_chunk_index_info(cls,
                             context: IndexHandlerContext,
                             index_info: IndexInfo,
                             chunk_index: Tuple[int],
                             chunk_index_info: ChunkIndexInfo,
                             output_axis_index: int,
                             index,
                             output_shape: int):
        tileable = context.tileable
        chunk_input = tileable.cix[chunk_index]

        dtypes = None
        if index_info.input_axis == 0:
            index_value = parse_index(chunk_input.index_value.to_pandas(),
                                      chunk_input, index, store_data=False)
        else:
            columns = chunk_input.columns_value.to_pandas()
            index_value = parse_index(columns[index], store_data=True)
            dtypes = chunk_input.dtypes[index]

        info = ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                  processed_index=index,
                                  output_shape=output_shape,
                                  index_value=index_value,
                                  dtypes=dtypes)
        chunk_index_info.set(info)


class NDArrayBoolIndexHandler(DataFrameIndexHandler, NDArrayBoolIndexHandlerBase):
    pass


class TensorBoolIndexHandler(TensorBoolIndexHandlerBase):
    @classmethod
    def set_chunk_index_info(cls,
                             context: IndexHandlerContext,
                             index_info: IndexInfo,
                             chunk_index: Tuple[int],
                             chunk_index_info: ChunkIndexInfo,
                             output_axis_index: int,
                             index,
                             output_shape: int):
        tileable = context.tileable
        chunk_input = tileable.cix[chunk_index]

        assert index_info.input_axis == 0, \
            'bool indexing on axis columns cannot be tensor'

        index_value = parse_index(chunk_input.index_value.to_pandas(),
                                  chunk_input, index, store_data=False)

        info = ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                  processed_index=index,
                                  output_shape=output_shape,
                                  index_value=index_value,
                                  dtypes=None)
        chunk_index_info.set(info)


class _FancyIndexHandler(DataFrameIndexHandler, IndexHandler):
    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = FancyIndexInfo(IndexType.fancy_index,
                              context.input_axis,
                              context.output_axis,
                              raw_index,
                              self)
        context.input_axis += 1
        context.output_axis += 1
        context.append(info)
        return info


class NDArrayFancyIndexHandler(_FancyIndexHandler):
    def accept(cls, raw_index):
        # raw index like list, and pd.Series
        # would have been converted to ndarray or tensor already
        return isinstance(raw_index, np.ndarray) and \
               raw_index.dtype != np.bool_

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        tileable = context.tileable
        check_chunks_unknown_shape([tileable], TilesError)

        # split raw index into chunks on the given axis
        split_info = split_indexes_into_chunks([tileable.nsplits[index_info.input_axis]],
                                               [index_info.raw_index])
        index_info.split_info = split_info

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        chunk_index_to_fancy_index_arrays = index_info.split_info[0]

        other_index_to_iter = dict()
        chunk_index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in chunk_index_to_info.items():
            i = chunk_index[index_info.input_axis]
            fancy_index_array = chunk_index_to_fancy_index_arrays[i,][0]

            if fancy_index_array.size == 0:
                # not effected
                del context.chunk_index_to_info[chunk_index]
                continue

            other_index = chunk_index[:1] if index_info.input_axis == 1 else chunk_index[1:]
            if other_index not in other_index_to_iter:
                other_index_to_iter[other_index] = itertools.count()
            output_axis_index = next(other_index_to_iter[other_index])
            output_axis_shape = fancy_index_array.shape[0]
            self.set_chunk_index_info(context, index_info, chunk_index,
                                      chunk_index_info, output_axis_index,
                                      fancy_index_array, output_axis_shape)

    @classmethod
    def need_postprocess(cls,
                         index_info: IndexInfo,
                         context: IndexHandlerContext):
        tileable = context.tileable

        if tileable.chunk_shape[index_info.input_axis] == 1:
            # if tileable only has 1 chunk on this axis
            # do not need postprocess
            return False
        # if ascending sorted, no need to postprocess
        return not index_info.split_info[2]

    def postprocess(self,
                    index_info: IndexInfo,
                    context: IndexHandlerContext) -> None:
        # could be 2 fancy indexes at most
        fancy_indexes = context.get_indexes(index_info.index_type)
        i_fancy_index = fancy_indexes.index(index_info)
        need_postprocesses = [fancy_index.handler.need_postprocess(fancy_index, context)
                              for fancy_index in fancy_indexes]

        if not need_postprocesses[i_fancy_index]:
            # do not need postprocess
            return

        if i_fancy_index == 0 and len(fancy_indexes) == 2 and need_postprocesses[1] and \
                isinstance(fancy_indexes[1].raw_index, np.ndarray):
            # check if need postprocess if 2 fancy indexes and now it's the first,
            # if so, skip postprocess for this one,
            # and do MapReduce just once for the second postprocess
            return

        chunks, nsplits = context.out_chunks, context.out_nsplits
        index_to_chunks = {c.index: c for c in chunks}

        to_concat_axes = tuple(fancy_index.output_axis
                               for i, fancy_index in enumerate(fancy_indexes)
                               if need_postprocesses[i])
        reorder_indexes = [calc_pos(fancy_index.raw_index.shape, fancy_index.split_info[1])
                           for i, fancy_index in enumerate(fancy_indexes)
                           if need_postprocesses[i]]
        new_out_chunks = []
        for chunk_index in itertools.product(
                *(range(len(ns)) for ax, ns in enumerate(nsplits)
                  if ax not in to_concat_axes)):
            if len(to_concat_axes) == 2:
                to_concat_chunks = chunks
            else:
                to_concat_chunks = []
                for i in range(len(nsplits[to_concat_axes[0]])):
                    to_concat_index = list(chunk_index)
                    to_concat_index.insert(to_concat_axes[0], i)
                    to_concat_chunks.append(index_to_chunks[tuple(to_concat_index)])
            concat_chunk = context.concat_chunks(to_concat_chunks, to_concat_axes)
            reorder_chunk = self._create_reorder_chunk(concat_chunk, to_concat_axes,
                                                       reorder_indexes, context)
            new_out_chunks.append(reorder_chunk)

        new_nsplits = list(nsplits)
        for ax, fancy_index in zip(to_concat_axes, fancy_indexes):
            new_nsplits[ax] = (fancy_index.raw_index.shape[0],)
        context.out_chunks = new_out_chunks
        context.out_nsplits = new_nsplits

    @classmethod
    def _create_reorder_chunk(cls,
                              concat_chunk: Chunk,
                              to_concat_axes: Tuple,
                              reorder_indexes: List,
                              context: IndexHandlerContext):
        reorder_chunk_op = context.op.copy().reset_key()
        indexes = [slice(None)] * concat_chunk.ndim
        for ax, reorder_index in zip(to_concat_axes, reorder_indexes):
            indexes[ax] = reorder_index
        reorder_chunk_op._indexes = indexes

        params = concat_chunk.params
        if isinstance(concat_chunk, SERIES_CHUNK_TYPE):
            if concat_chunk.index_value.has_value():
                # if concat chunk's index has value, we could calculate the new index
                reorder_index = concat_chunk.index_value.to_pandas()[reorder_indexes[0]]
                params['index_value'] = parse_index(reorder_index, store_data=True)
            else:
                params['index_value'] = parse_index(concat_chunk.index_value.to_pandas(), indexes)
            return reorder_chunk_op.new_chunk([concat_chunk], kws=[params])
        else:
            if 0 in to_concat_axes:
                if concat_chunk.index_value.has_value():
                    # if concat chunk's index has value, and index on axis 0,
                    # we could calculate the new index
                    reorder_index = concat_chunk.index_value.to_pandas()[reorder_indexes[0]]
                    params['index_value'] = parse_index(reorder_index, store_data=True)
                else:
                    params['index_value'] = parse_index(concat_chunk.index_value.to_pandas(),
                                                        indexes[0])
            if 1 in to_concat_axes:
                reorder_columns = concat_chunk.columns_value.to_pandas()[reorder_indexes[-1]]
                params['columns_value'] = parse_index(reorder_columns, store_data=True)
                params['dtypes'] = concat_chunk.dtypes[reorder_indexes[-1]]

        return reorder_chunk_op.new_chunk([concat_chunk], kws=[params])


class DataFrameIlocIndexesHandler(IndexesHandler):
    def __init__(self):
        super().__init__()
        self.register(IntegralIndexHandler,
                      NDArrayFancyIndexHandler,
                      SliceIndexHandler,
                      NDArrayBoolIndexHandler,
                      TensorBoolIndexHandler)

    def create_context(self, op):
        return DataFrameIndexHandlerContext(op)
