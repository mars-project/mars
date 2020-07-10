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

from enum import Enum

from ..operands import Operand, TileableOperandMixin, Fetch, FetchMixin, \
    Fuse, FuseChunkMixin, MapReduceOperand, ShuffleProxy
from ..serialize import ValueType, ListField
from ..tensor.core import TensorChunkData, TensorChunk, TensorData, Tensor, \
    TENSOR_TYPE, CHUNK_TYPE as TENSOR_CHUNK_TYPE
from ..tensor.operands import TensorOperandMixin
from ..tensor.fuse import TensorFuseChunk
from ..tensor.fetch import TensorFetch
from ..dataframe.core import DataFrameChunkData, DataFrameChunk, DataFrameData, DataFrame, \
    SeriesChunkData, SeriesChunk, SeriesData, Series, SERIES_TYPE, SERIES_CHUNK_TYPE, \
    TILEABLE_TYPE as DATAFRAME_TYPE, CHUNK_TYPE as DATAFRAME_CHUNK_TYPE
from ..dataframe.operands import DataFrameOperandMixin, DataFrameFuseChunk, ObjectType
from ..dataframe.fetch import DataFrameFetch
from ..core import ObjectChunkData, ObjectChunk, ObjectData, Object


class OutputType(Enum):
    object = 1
    tensor = 2
    dataframe = 3
    series = 4


def _on_serialize_output_types(output_types):
    if output_types is not None:
        return [ot.value for ot in output_types]


def _on_deserialize_output_types(output_types):
    if output_types is not None:
        return [OutputType(ot) for ot in output_types]


class LearnOperand(Operand):
    _output_types = ListField('output_type', tp=ValueType.int8,
                              on_serialize=_on_serialize_output_types,
                              on_deserialize=_on_deserialize_output_types)

    @property
    def output_types(self):
        return self._output_types


class LearnOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'learn'

    def _create_chunk(self, output_idx, index, **kw):
        output_type = self.output_types[output_idx] \
            if self.output_types is not None else OutputType.tensor
        if output_type == OutputType.tensor:
            chunk_data_type = TensorChunkData
            chunk_type = TensorChunk
        elif output_type == OutputType.dataframe:
            chunk_data_type = DataFrameChunkData
            chunk_type = DataFrameChunk
        elif output_type == OutputType.series:
            chunk_data_type = SeriesChunkData
            chunk_type = SeriesChunk
        else:
            assert output_type == OutputType.object
            chunk_data_type = ObjectChunkData
            chunk_type = ObjectChunk
        data = chunk_data_type(op=self, index=index, **kw)
        return chunk_type(data)

    def _create_tileable(self, output_idx, **kw):
        output_type = self.output_types[output_idx] \
            if self.output_types is not None else OutputType.tensor
        if output_type == OutputType.tensor:
            data_type = TensorData
            entity_type = Tensor
        elif output_type == OutputType.dataframe:
            data_type = DataFrameData
            entity_type = DataFrame
        elif output_type == OutputType.series:
            data_type = SeriesData
            entity_type = Series
        else:
            assert output_type == OutputType.object
            data_type = ObjectData
            entity_type = Object
        data = data_type(op=self, **kw)
        return entity_type(data)

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        if isinstance(tileable, TENSOR_TYPE):
            return TensorOperandMixin.concat_tileable_chunks(tileable)
        elif isinstance(tileable, DATAFRAME_TYPE):
            return DataFrameOperandMixin.concat_tileable_chunks(tileable)
        else:
            # op has to implement its logic of `concat_tileable_chunks`
            raise NotImplementedError

    @classmethod
    def create_tileable_from_chunks(cls, chunks, inputs=None, **kw):
        if isinstance(chunks[0], TENSOR_CHUNK_TYPE):
            return TensorOperandMixin.create_tileable_from_chunks(
                chunks, inputs=inputs, **kw)
        elif isinstance(chunks[0], DATAFRAME_CHUNK_TYPE):
            return DataFrameOperandMixin.create_tileable_from_chunks(
                chunks, inputs=inputs, **kw)
        else:
            # op has to implement its logic of `create_tileable_from_chunks`
            raise NotImplementedError

    def get_fetch_op_cls(self, obj):
        # Shuffle proxy should be tensor or dataframe
        if isinstance(obj, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            return TensorFetch
        elif isinstance(obj, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            def _inner(**kw):
                object_type = ObjectType.series \
                    if isinstance(obj, (SERIES_TYPE, SERIES_CHUNK_TYPE)) else \
                    ObjectType.dataframe
                return DataFrameFetch(object_type=object_type, **kw)

            return _inner
        else:
            def _inner(**kw):
                return LearnObjectFetch(output_types=[OutputType.object], **kw)

            return _inner

    def get_fuse_op_cls(self, obj):
        if isinstance(obj, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            return TensorFuseChunk
        elif isinstance(obj, (DATAFRAME_TYPE, DATAFRAME_CHUNK_TYPE)):
            return DataFrameFuseChunk
        else:
            return LearnObjectFuseChunk


class LearnObjectFetchMixin(LearnOperandMixin, FetchMixin):
    __slots__ = ()


class LearnObjectFetch(Fetch, LearnObjectFetchMixin):
    _output_types = ListField('output_type', tp=ValueType.int8,
                              on_serialize=_on_serialize_output_types,
                              on_deserialize=_on_deserialize_output_types)

    def __init__(self, to_fetch_key=None, output_types=None, **kw):
        super().__init__(_to_fetch_key=to_fetch_key, _output_types=output_types, **kw)

    @property
    def output_types(self):
        return self._output_types

    def _new_chunks(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_chunks(inputs, kws=kws, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if '_key' in kw and self._to_fetch_key is None:
            self._to_fetch_key = kw['_key']
        return super()._new_tileables(inputs, kws=kws, **kw)


class LearnObjectFuseChunkMixin(FuseChunkMixin, LearnOperandMixin):
    __slots__ = ()


class LearnObjectFuseChunk(Fuse, LearnObjectFuseChunkMixin):
    @property
    def output_types(self):
        return [OutputType.object]


class LearnShuffleProxy(ShuffleProxy, LearnOperandMixin):
    _output_types = ListField('output_type', tp=ValueType.int8,
                              on_serialize=_on_serialize_output_types,
                              on_deserialize=_on_deserialize_output_types)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if self._output_types is None:
            self._output_types = [OutputType.object]

    @property
    def output_types(self):
        return self._output_types


class LearnMapReduceOperand(MapReduceOperand):
    _output_types = ListField('output_type', tp=ValueType.int8,
                              on_serialize=_on_serialize_output_types,
                              on_deserialize=_on_deserialize_output_types)

    @property
    def output_types(self):
        return self._output_types
