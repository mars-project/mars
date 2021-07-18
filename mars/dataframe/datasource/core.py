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

import asyncio
import uuid
from typing import List, Optional

import pandas as pd

from ...core import recursive_tile
from ...core.context import get_context
from ...serialization.serializables import Int64Field, StringField
from ...typing import TileableType, ChunkType, OperandType
from ..core import IndexValue
from ..operands import DataFrameOperand, DataFrameOperandMixin


class HeadOptimizedDataSource(DataFrameOperand, DataFrameOperandMixin):
    __slots__ = ()
    # Data source op that optimized for head,
    # First, it will try to trigger first_chunk.head() and raise TilesError,
    # When iterative tiling is triggered,
    # check if the first_chunk.head() meets requirements.
    _nrows = Int64Field('nrows')

    @property
    def nrows(self):
        return self._nrows

    @property
    def first_chunk(self):
        return getattr(self, '_first_chunk', None)

    @classmethod
    def _tile(cls, op):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _tile_head(cls, op: "HeadOptimizedDataSource"):
        tileds = cls._tile(op)
        chunks = tileds[0].chunks

        # execute first chunk
        yield chunks[:1]

        ctx = get_context()
        chunk_shape = ctx.get_chunks_meta([chunks[0].key])[0]['shape']

        if chunk_shape[0] == op.nrows:
            # the first chunk has enough data
            tileds[0]._nsplits = tuple((s,) for s in chunk_shape)
            chunks[0]._shape = chunk_shape
            tileds[0]._chunks = chunks[:1]
            tileds[0]._shape = chunk_shape
        else:
            for chunk in tileds[0].chunks:
                chunk.op._nrows = None
            # otherwise
            tiled = yield from recursive_tile(tileds[0].iloc[:op.nrows])
            tileds = [tiled]
        return tileds

    @classmethod
    def tile(cls, op: "HeadOptimizedDataSource"):
        if op.nrows is not None:
            return (yield from cls._tile_head(op))
        else:
            return cls._tile(op)


class ColumnPruneSupportedDataSourceMixin(DataFrameOperandMixin):
    __slots__ = ()

    def get_columns(self):  # pragma: no cover
        raise NotImplementedError

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        raise NotImplementedError


class _IncrementalIndexRecorder:
    _semaphores: List[Optional[asyncio.Semaphore]]
    _chunk_sizes: List[Optional[int]]

    def __init__(self, n_chunk: int):
        self._n_chunk = n_chunk
        self._semaphores = [None] * n_chunk
        self._chunk_sizes = [None] * n_chunk

    async def init(self):
        for i in range(self._n_chunk):
            if i > 0:
                sem = asyncio.Semaphore(i)
                for _ in range(i):
                    await sem.acquire()
                self._semaphores[i] = sem

    async def wait(self, i: int):
        if i == 0:
            return 0
        await self._semaphores[i].acquire()
        return sum(self._chunk_sizes[:i])

    async def done(self, i: int, size: int):
        self._chunk_sizes[i] = size
        for j in range(i + 1, self._n_chunk):
            self._semaphores[j].release()
        return i == self._n_chunk - 1


class IncrementalIndexDatasource(HeadOptimizedDataSource):
    __slots__ = ()

    incremental_index_recorder_name = StringField('incremental_index_recorder_name')


class IncrementalIndexDataSourceMixin(DataFrameOperandMixin):
    __slots__ = ()

    def _new_tileables(self, inputs, kws=None, **kw) -> List[TileableType]:
        self.index_as_priority = True
        return super()._new_tileables(inputs, kws=kws, **kw)

    def _new_chunks(self, inputs, kws=None, **kw) -> List[ChunkType]:
        self.index_as_priority = True
        return super()._new_chunks(inputs, kws=kws, **kw)

    @classmethod
    def post_tile(cls, op: OperandType, result: TileableType):
        if result is not None and \
                isinstance(result.index_value.value, IndexValue.RangeIndex):
            n_chunk = len(result.chunks)
            ctx = get_context()
            name = str(uuid.uuid4())
            recorder = ctx.create_remote_object(
                name, _IncrementalIndexRecorder, n_chunk)
            recorder.init()
            for chunk in result.chunks:
                chunk.op.incremental_index_recorder_name = name

    @classmethod
    def post_execute(cls, ctx: dict, op: OperandType):
        out = op.outputs[0]
        result = ctx[out.key]
        if isinstance(result.index, pd.RangeIndex):
            recorder_name = op.incremental_index_recorder_name
            recorder = ctx.get_remote_object(recorder_name)
            index = out.index[0]
            # wait for previous chunks to finish
            size = recorder.wait(index)
            result.index += size
            done = recorder.done(index, len(result))
            if done:
                ctx.destroy_remote_object(recorder_name)
