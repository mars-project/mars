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

from ..operands import Operand, TileableOperandMixin, Fetch, FetchMixin, \
    Fuse, FuseChunkMixin, MapReduceOperand, ShuffleProxy, OutputType
from ..tensor.core import TENSOR_TYPE, TENSOR_CHUNK_TYPE
from ..tensor.operands import TensorOperandMixin
from ..tensor.fuse import TensorFuseChunk
from ..tensor.fetch import TensorFetch
from ..dataframe.core import SERIES_TYPE, SERIES_CHUNK_TYPE, TILEABLE_TYPE as DATAFRAME_TYPE, \
    CHUNK_TYPE as DATAFRAME_CHUNK_TYPE
from ..dataframe.operands import DataFrameOperandMixin, DataFrameFuseChunk
from ..dataframe.fetch import DataFrameFetch


LearnOperand = Operand


class LearnOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'learn'

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
                output_types = [OutputType.series] \
                    if isinstance(obj, (SERIES_TYPE, SERIES_CHUNK_TYPE)) else \
                    [OutputType.dataframe]
                return DataFrameFetch(output_types=output_types, **kw)

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
    def __init__(self, to_fetch_key=None, output_types=None, **kw):
        super().__init__(_to_fetch_key=to_fetch_key, _output_types=output_types, **kw)

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

    _output_type_ = OutputType.object


class LearnObjectFuseChunk(LearnObjectFuseChunkMixin, Fuse):
    pass


class LearnShuffleProxy(ShuffleProxy, LearnOperandMixin):
    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.object]


LearnMapReduceOperand = MapReduceOperand
