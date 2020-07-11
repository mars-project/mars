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
import pandas as pd
from pandas.core.dtypes.cast import find_common_type

from ...core import TileableEntity, Chunk
from ...operands import OperandStage
from ...tiles import TilesError
from ...tensor.core import TENSOR_TYPE
from ...tensor.indexing.index_lib import IndexHandlerContext, IndexHandler, \
    IndexInfo, IndexType, ChunkIndexInfo as ChunkIndexInfoBase, \
    SliceIndexHandler as SliceIndexHandlerBase, \
    NDArrayBoolIndexHandler as NDArrayBoolIndexHandlerBase, \
    TensorBoolIndexHandler as TensorBoolIndexHandlerBase, \
    IntegralIndexHandler, IndexesHandler
from ...tensor.utils import split_indexes_into_chunks, calc_pos, \
    filter_inputs, slice_split, calc_sliced_size, to_numpy
from ...utils import check_chunks_unknown_shape, classproperty
from ..core import SERIES_CHUNK_TYPE, SERIES_TYPE, IndexValue
from ..operands import ObjectType
from ..utils import parse_index
from .utils import convert_labels_into_positions


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


class LabelFancyIndexInfo(IndexInfo):
    def __init__(self,
                 index_type: IndexType,
                 input_axis: int,
                 output_axis: int,
                 raw_index,
                 handler):
        super().__init__(index_type, input_axis, output_axis,
                         raw_index, handler)

        # store chunk_index -> labels
        self.chunk_index_to_labels = None
        self.is_label_asc_sorted = None


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
                ((isinstance(axis, tuple) and len(axis) == 1) or isinstance(axis, int)):
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
        chunk_op._stage = OperandStage.map

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
            kw['name'] = getattr(self.op.outputs[0], 'name', None)
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


class LabelSliceIndexHandler(IndexHandler):
    def accept(cls, raw_index):
        return isinstance(raw_index, slice)

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = IndexInfo(IndexType.label_slice,
                         context.input_axis,
                         context.output_axis,
                         raw_index,
                         self)
        context.input_axis += 1
        context.output_axis += 1
        context.append(info)
        return info

    @staticmethod
    def _slice_all(slc):
        return slc.start is None and slc.stop is None and \
               (slc.step is None or slc.step == 1)

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        if isinstance(tileable, SERIES_TYPE):
            index_value = tileable.index_value
        else:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]

        # check if chunks have unknown shape
        check = False
        if index_value.has_value():
            # index_value has value,
            check = True
        elif self._slice_all(index_info.raw_index):
            # if slice on all data
            check = True

        if check:
            if any(np.isnan(ns) for ns in tileable.nsplits[input_axis]):
                raise TilesError('Input tileable {} has chunks with unknown shape '
                                 'on axis {}'.format(tileable, input_axis))

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
            start, stop = index.slice_locs(slc.start, slc.stop, slc.step, kind='loc')
            pos_slc = slice(start, stop, slc.step)
            kw['index_value'] = parse_index(index[pos_slc], chunk_input, slc,
                                            store_data=False)
        else:
            assert index_info.input_axis == 1
            dtypes = chunk_input.dtypes
            # do not store index value if output axis is 0
            store_data = True if index_info.output_axis == 1 else False
            columns = dtypes.loc[slc].index
            kw['index_value'] = parse_index(columns, store_data=store_data)
            kw['dtypes'] = chunk_input.dtypes[slc]

        chunk_index_info.set(ChunkIndexAxisInfo(**kw))

    def _process_has_value_index(self,
                                 tileable: TileableEntity,
                                 index_info: IndexInfo,
                                 index_value,
                                 input_axis: int,
                                 context: IndexHandlerContext) -> None:
        pd_index = index_value.to_pandas()
        if self._slice_all(index_info.raw_index):
            slc = slice(None)
        else:
            # turn label-based slice into position-based slice
            start, end = pd_index.slice_locs(index_info.raw_index.start,
                                             index_info.raw_index.stop,
                                             index_info.raw_index.step,
                                             kind='loc')
            slc = slice(start, end, index_info.raw_index.step)

        cum_nsplit = [0] + np.cumsum(tileable.nsplits[index_info.input_axis]).tolist()
        # split position-based slice into chunk slices
        effected_i_to_slc = slice_split(slc, tileable.nsplits[index_info.input_axis])
        is_reversed = (slc.step or 0) < 0
        output_axis_index_range = range(len(effected_i_to_slc)) if not is_reversed else \
            range(len(effected_i_to_slc) - 1, -1, -1)
        other_index_to_iter = dict()

        index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in index_to_info.items():
            i = chunk_index[input_axis]
            other_index = chunk_index[:input_axis] + chunk_index[input_axis + 1:]
            size = tileable.nsplits[input_axis][i]
            if i not in effected_i_to_slc:
                # delete it, the input chunk could be ignored
                del context.chunk_index_to_info[chunk_index]
            else:
                chunk_slc = effected_i_to_slc[i]
                output_shape = calc_sliced_size(size, chunk_slc)
                if other_index not in other_index_to_iter:
                    other_index_to_iter[other_index] = iter(output_axis_index_range)
                output_axis_index = next(other_index_to_iter[other_index])

                # turn position-based slice back into label-based slice
                start = chunk_slc.start
                if start is not None:
                    abs_start = cum_nsplit[i] + start
                    label_start = pd_index[abs_start]
                else:
                    label_start = None
                stop = chunk_slc.stop
                if stop is not None:
                    abs_stop = cum_nsplit[i] + stop - 1  # label slice include the stop
                    label_stop = pd_index[abs_stop] if abs_stop < len(pd_index) else None
                else:
                    label_stop = None

                label_slc = slice(label_start, label_stop, chunk_slc.step)
                self.set_chunk_index_info(context, index_info, chunk_index, chunk_index_info,
                                          output_axis_index, label_slc, output_shape)

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        if isinstance(tileable, SERIES_TYPE):
            index_value = tileable.index_value
        else:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]

        if index_value.has_value() or self._slice_all(index_info.raw_index):
            self._process_has_value_index(tileable, index_info,
                                          index_value, input_axis, context)
        else:
            other_index_to_iter = dict()
            # slice on all chunks on the specified axis
            for chunk_index, chunk_index_info in context.chunk_index_to_info.items():
                other_index = chunk_index[:1] if input_axis == 1 else chunk_index[1:]
                if other_index not in other_index_to_iter:
                    other_index_to_iter[other_index] = itertools.count()
                output_axis_index = next(other_index_to_iter[other_index])
                self.set_chunk_index_info(context, index_info, chunk_index,
                                          chunk_index_info, output_axis_index,
                                          index_info.raw_index, np.nan)


class LabelIndexHandler(IndexHandler):
    def accept(cls, raw_index):
        # accept type other than slice, ndarray and tensor
        return not isinstance(raw_index, (slice, np.ndarray, TENSOR_TYPE))

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        tileable = context.tileable
        input_axis = context.input_axis
        if tileable.ndim == 2:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]
        else:
            index_value = tileable.index_value

        if index_value.has_value():
            pd_index = index_value.to_pandas()
            loc = pd_index.get_loc(raw_index)
            if isinstance(loc, slice):
                # if is slice, means index not unique, but monotonic
                # just call LabelSliceIndexHandler
                new_raw_index = slice(raw_index, raw_index)
                return LabelSliceIndexHandler.get_instance().parse(new_raw_index, context)
            elif isinstance(loc, np.ndarray):
                # bool indexing, non unique, and not monotonic
                return NDArrayBoolIndexHandler.get_instance().parse(loc, context)
        else:
            return LabelNDArrayFancyIndexHandler.get_instance().parse(raw_index, context)

        info = IndexInfo(IndexType.label,
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
        # if index has value on input axis,
        # label will be converted to position,
        # thus chunks cannot have unknown shape on this axis
        tileable = context.tileable
        input_axis = index_info.input_axis
        if tileable.ndim == 1:
            index_value = tileable.index_value
        else:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]
        if index_value.has_value():
            if any(np.isnan(ns) for ns in tileable.nsplits[input_axis]):
                raise TilesError('Input tileable {} has chunks with unknown shape '
                                 'on axis {}'.format(tileable, input_axis))

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        if tileable.ndim == 1:
            index_value = tileable.index_value
        else:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]

        if index_value.has_value():
            pd_index = index_value.to_pandas()
            loc = pd_index.get_loc(index_info.raw_index)

            # other situations have been delegated to different handlers
            assert isinstance(loc, int)

            effected_i_to_slc = slice_split(loc, tileable.nsplits[index_info.input_axis])

            index_to_info = context.chunk_index_to_info.copy()
            for chunk_index, chunk_index_info in index_to_info.items():
                i = chunk_index[input_axis]
                if i not in effected_i_to_slc:
                    # delete it, the input chunk could be ignored
                    del context.chunk_index_to_info[chunk_index]
                else:
                    chunk_index_info.set(ChunkIndexAxisInfo(output_axis_index=None,
                                                            processed_index=index_info.raw_index,
                                                            output_shape=None,
                                                            index_value=None,
                                                            dtypes=None))


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
            dtypes = getattr(chunk_input.dtypes, cls.kind)[index]
            columns = dtypes.index
            index_value = parse_index(columns, store_data=True)

        info = ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                  processed_index=index,
                                  output_shape=output_shape,
                                  index_value=index_value,
                                  dtypes=dtypes)
        chunk_index_info.set(info)


class NDArrayBoolIndexHandler(NDArrayBoolIndexHandlerBase):
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

        if index_info.input_axis == 0:
            dtype = chunk_input.index_value.to_pandas().dtype
            index_value = parse_index(pd.Index([], dtype=dtype),
                                      chunk_input, index, store_data=False)
            dtypes = None
        else:
            pd_index = chunk_input.columns_value.to_pandas()
            filtered_index = pd_index[index]
            index_value = parse_index(filtered_index, store_data=True)
            dtypes = chunk_input.dtypes[index]

        info = ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                  processed_index=index,
                                  output_shape=output_shape,
                                  index_value=index_value,
                                  dtypes=dtypes)
        chunk_index_info.set(info)


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

        index_value = parse_index(pd.Index([], chunk_input.index_value.to_pandas().dtype),
                                  chunk_input, index, store_data=False)

        info = ChunkIndexAxisInfo(output_axis_index=output_axis_index,
                                  processed_index=index,
                                  output_shape=output_shape,
                                  index_value=index_value,
                                  dtypes=None)
        chunk_index_info.set(info)


class _FancyIndexHandler(DataFrameIndexHandler, IndexHandler):
    @classproperty
    def kind(self):  # pylint: disable=no-self-use
        return 'iloc'

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
            fancy_index_array = chunk_index_to_fancy_index_arrays[i, ][0]

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
        for fancy_index in fancy_indexes:
            new_nsplits[fancy_index.output_axis] = (fancy_index.raw_index.shape[0],)
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


class _LabelFancyIndexHandler(DataFrameIndexHandler, IndexHandler):
    @classproperty
    def kind(self):  # pylint: disable=no-self-use
        return 'loc'


class LabelNDArrayFancyIndexHandler(_LabelFancyIndexHandler):
    def accept(cls, raw_index):
        return isinstance(raw_index, np.ndarray) and \
               raw_index.dtype != np.bool_

    def parse(self,
              raw_index,
              context: IndexHandlerContext) -> IndexInfo:
        info = LabelFancyIndexInfo(IndexType.label_fancy_index,
                                   context.input_axis,
                                   context.output_axis,
                                   raw_index,
                                   self)
        context.input_axis += 1
        if not np.isscalar(raw_index):
            context.output_axis += 1
        context.append(info)
        return info

    def preprocess(self,
                   index_info: IndexInfo,
                   context: IndexHandlerContext) -> None:
        tileable = context.tileable
        check_chunks_unknown_shape([tileable], TilesError)

        input_axis = index_info.input_axis
        if tileable.ndim == 2:
            index_value = [tileable.index_value, tileable.columns_value][input_axis]
        else:
            index_value = tileable.index_value
        cum_nsplit = [0] + np.cumsum(tileable.nsplits[input_axis]).tolist()
        if index_value.has_value():
            # turn label-based fancy index into position-based
            pd_index = index_value.to_pandas()
            positions = convert_labels_into_positions(pd_index, index_info.raw_index)
            split_info = split_indexes_into_chunks([tileable.nsplits[input_axis]],
                                                   [positions])
            chunk_index_to_pos = split_info[0]
            is_asc_sorted = split_info[-1]

            # convert back to labels for chunk_index
            chunk_index_to_labels = dict()
            for chunk_index, pos in chunk_index_to_pos.items():
                # chunk_index and pos are all list with 1 element
                abs_pos = pos[0] + cum_nsplit[chunk_index[0]]
                chunk_labels = to_numpy(pd_index[abs_pos])
                chunk_index_to_labels[chunk_index[0]] = chunk_labels

            index_info.is_label_asc_sorted = is_asc_sorted
            index_info.chunk_index_to_labels = chunk_index_to_labels
        else:
            index = index_info.raw_index
            if np.isscalar(index):
                # delegation from label index handler
                index = np.atleast_1d(index)
            # does not know the right positions, need postprocess always
            index_info.is_label_asc_sorted = False
            # do df.loc on each chunk
            index_info.chunk_index_to_labels = \
                {i: index for i in range(tileable.chunk_shape[input_axis])}

    def process(self,
                index_info: IndexInfo,
                context: IndexHandlerContext) -> None:
        tileable = context.tileable
        input_axis = index_info.input_axis
        chunk_index_to_labels = index_info.chunk_index_to_labels

        other_index_to_iter = dict()
        chunk_index_to_info = context.chunk_index_to_info.copy()
        for chunk_index, chunk_index_info in chunk_index_to_info.items():
            i = chunk_index[input_axis]
            chunk_labels = chunk_index_to_labels[i]
            size = chunk_labels.size

            if size == 0:
                # not effected
                del context.chunk_index_to_info[chunk_index]
                continue

            if np.isscalar(index_info.raw_index) and \
                    isinstance(tileable.index_value.value, IndexValue.DatetimeIndex) and \
                    isinstance(chunk_labels[0], str):
                # special case when index is DatetimeIndex and loc by string
                # convert back list to scalar because if keep list,
                # KeyError will always happen
                chunk_labels = chunk_labels[0].item()

            other_index = chunk_index[:1] if input_axis == 1 else chunk_index[1:]
            if other_index not in other_index_to_iter:
                other_index_to_iter[other_index] = itertools.count()
            output_axis_index = next(other_index_to_iter[other_index])
            output_axis_shape = size
            self.set_chunk_index_info(context, index_info, chunk_index,
                                      chunk_index_info, output_axis_index,
                                      chunk_labels, output_axis_shape)

    @classmethod
    def need_postprocess(cls,
                         index_info: IndexInfo,
                         context: IndexHandlerContext):
        # if ascending sorted, no need to postprocess
        return not index_info.is_label_asc_sorted

    def postprocess(self,
                    index_info: IndexInfo,
                    context: IndexHandlerContext) -> None:
        if not self.need_postprocess(index_info, context):
            # do not need postprocess
            return

        chunks, nsplits = context.out_chunks, context.out_nsplits
        index_to_chunks = {c.index: c for c in chunks}

        axis = index_info.output_axis
        new_out_chunks = []
        chunk_axis_shapes = dict()
        for chunk_index in itertools.product(*(range(len(ns)) for ax, ns in enumerate(nsplits)
                                               if ax != axis)):
            to_concat_chunks = []
            for i in range(len(nsplits[axis])):
                if axis == 0:
                    to_concat_index = (i,) + chunk_index
                else:
                    to_concat_index = chunk_index + (i,)
                to_concat_chunks.append(index_to_chunks[to_concat_index])
            concat_chunk = context.concat_chunks(to_concat_chunks, axis)
            chunk_op = context.op.copy().reset_key()
            indexes = [slice(None)] * len(nsplits)
            indexes[axis] = index_info.raw_index
            params = concat_chunk.params
            if np.isscalar(index_info.raw_index):
                assert axis == 0
                if 'columns_value' in params:
                    params['index_value'] = params.pop('columns_value')
                    params['dtype'] = find_common_type(params['dtypes'].tolist())
                    del params['dtypes']
                    if getattr(context.op.outputs[0], 'name', None) is not None:
                        params['name'] = context.op.outputs[0].name
                if len(params['index']) == chunks[0].ndim:
                    index = list(params['index'])
                    index.pop(index_info.output_axis)
                    params['index'] = tuple(index)
                    shape = list(params['shape'])
                    shape.pop(index_info.output_axis)
                    params['shape'] = tuple(shape)
                if context.op.outputs[0].ndim == 0:
                    del params['index_value']
            elif axis == 0:
                params['index_value'] = parse_index(pd.Index(index_info.raw_index), store_data=False)
            else:
                params['dtypes'] = dtypes = concat_chunk.dtypes.loc[index_info.raw_index]
                params['columns_value'] = parse_index(dtypes.index, store_data=True)
                shape = list(params['shape'])
                shape[1] = len(dtypes)
            chunk_op._indexes = indexes
            out_chunk = chunk_op.new_chunk([concat_chunk], kws=[params])
            if len(out_chunk.shape) != 0:
                chunk_axis_shapes[out_chunk.index[axis]] = out_chunk.shape[axis]
            new_out_chunks.append(out_chunk)

        new_nsplits = list(nsplits)
        if np.isscalar(index_info.raw_index):
            new_nsplits = new_nsplits[:axis] + new_nsplits[axis + 1:]
        else:
            new_nsplits[axis] = (sum(chunk_axis_shapes.values()),)
        context.out_chunks = new_out_chunks
        context.out_nsplits = new_nsplits


class DataFrameIlocIndexesHandler(IndexesHandler):
    def __init__(self):
        super().__init__()
        self.register(IntegralIndexHandler,
                      SliceIndexHandler,
                      NDArrayBoolIndexHandler,
                      TensorBoolIndexHandler,
                      NDArrayFancyIndexHandler)

    def create_context(self, op):
        return DataFrameIndexHandlerContext(op)


class DataFrameLocIndexesHandler(IndexesHandler):
    def __init__(self):
        super().__init__()
        self.register(LabelIndexHandler,
                      LabelSliceIndexHandler,
                      NDArrayBoolIndexHandler,
                      TensorBoolIndexHandler,
                      LabelNDArrayFancyIndexHandler)

    def create_context(self, op):
        return DataFrameIndexHandlerContext(op)
