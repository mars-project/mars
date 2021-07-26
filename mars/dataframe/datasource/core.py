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
from typing import List, Optional, Union

from ...core import recursive_tile
from ...core.context import get_context, Context
from ...oscar import ActorNotExist
from ...serialization.serializables import Int64Field, StringField
from ...typing import TileableType, OperandType
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
    _done: List[Optional[asyncio.Event]]
    _chunk_sizes: List[Optional[int]]

    def __init__(self, n_chunk: int):
        self._n_chunk = n_chunk
        self._done = [asyncio.Event() for _ in range(n_chunk)]
        self._chunk_sizes = [None] * n_chunk
        self._waiters = set()

    def _can_destroy(self):
        return all(e.is_set() for e in self._done) and not self._waiters

    def add_waiter(self, i: int):
        self._waiters.add(i)

    async def wait(self, i: int):
        if i == 0:
            return 0, self._can_destroy()
        self._waiters.add(i)
        try:
            await asyncio.gather(*(e.wait() for e in self._done[:i]))
        finally:
            self._waiters.remove(i)
        # all chunk finished and no waiters
        return sum(self._chunk_sizes[:i]), self._can_destroy()

    async def finish(self, i: int, size: int):
        self._chunk_sizes[i] = size
        self._done[i].set()


class IncrementalIndexDatasource(HeadOptimizedDataSource):
    __slots__ = ()

    incremental_index_recorder_name = StringField('incremental_index_recorder_name')


class IncrementalIndexDataSourceMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def post_tile(cls, op: OperandType, results: List[TileableType]):
        if op.incremental_index and results is not None and \
                isinstance(results[0].index_value.value, IndexValue.RangeIndex):
            result = results[0]
            for chunk in result.chunks:
                chunk.op.priority = -chunk.index[0]
            n_chunk = len(result.chunks)
            ctx = get_context()
            if ctx:
                name = str(uuid.uuid4())
                ctx.create_remote_object(
                    name, _IncrementalIndexRecorder, n_chunk)
                for chunk in result.chunks:
                    chunk.op.incremental_index_recorder_name = name

    @classmethod
    def pre_execute(cls, ctx: Union[dict, Context], op: OperandType):
        out = op.outputs[0]
        if op.incremental_index and \
                isinstance(out.index_value.value, IndexValue.RangeIndex) and \
                getattr(op, 'incremental_index_recorder_name', None):
            index = out.index[0]
            recorder_name = op.incremental_index_recorder_name
            recorder = ctx.get_remote_object(recorder_name)
            recorder.add_waiter(index)

    @classmethod
    def post_execute(cls, ctx: Union[dict, Context], op: OperandType):
        out = op.outputs[0]
        result = ctx[out.key]
        if op.incremental_index and \
                isinstance(out.index_value.value, IndexValue.RangeIndex) and \
                getattr(op, 'incremental_index_recorder_name', None):
            recorder_name = op.incremental_index_recorder_name
            recorder = ctx.get_remote_object(recorder_name)
            index = out.index[0]
            recorder.finish(index, len(result))
            # wait for previous chunks to finish, then update index
            size, can_destroy = recorder.wait(index)
            result.index += size
            if can_destroy:
                try:
                    ctx.destroy_remote_object(recorder_name)
                except ActorNotExist:
                    pass
