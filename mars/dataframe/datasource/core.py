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

import numpy as np

from ...core import recursive_tile
from ...core.context import get_context, Context
from ...config import options
from ...oscar import ActorNotExist
from ...serialization.serializables import Int64Field, StringField
from ...typing import TileableType, OperandType
from ...utils import parse_readable_size
from ..core import IndexValue, OutputType
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import merge_index_value


class HeadOptimizedDataSource(DataFrameOperand, DataFrameOperandMixin):
    __slots__ = ()
    # Data source op that optimized for head,
    # First, it will try to trigger first_chunk.head() and raise TilesError,
    # When iterative tiling is triggered,
    # check if the first_chunk.head() meets requirements.
    nrows = Int64Field("nrows", default=None)

    @property
    def first_chunk(self):
        return getattr(self, "_first_chunk", None)

    @classmethod
    def _tile(cls, op):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _tile_head(cls, op: "HeadOptimizedDataSource"):
        tileds = cls._tile(op)
        chunks = tileds[0].chunks

        # execute first chunk
        yield chunks[:1]

        chunk_shape = chunks[0].shape
        if chunk_shape[0] == op.nrows:
            # the first chunk has enough data
            tileds[0]._nsplits = tuple((s,) for s in chunk_shape)
            chunks[0]._shape = chunk_shape
            tileds[0]._chunks = chunks[:1]
            tileds[0]._shape = chunk_shape
        else:
            for chunk in tileds[0].chunks:
                chunk.op.nrows = None
            # otherwise
            tiled = yield from recursive_tile(tileds[0].iloc[: op.nrows])
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

    incremental_index_recorder_name = StringField("incremental_index_recorder_name")


class IncrementalIndexDataSourceMixin(DataFrameOperandMixin):
    __slots__ = ()

    @classmethod
    def post_tile(cls, op: OperandType, results: List[TileableType]):
        if (
            op.incremental_index
            and results is not None
            and isinstance(results[0].index_value.value, IndexValue.RangeIndex)
        ):
            result = results[0]
            chunks = []
            for chunk in result.chunks:
                if not isinstance(chunk.op, cls):
                    # some chunks are merged, get the inputs
                    chunks.extend(chunk.inputs)
                else:
                    chunks.append(chunk)
            for chunk in chunks:
                chunk.op.priority = -chunk.index[0]
            n_chunk = len(chunks)
            ctx = get_context()
            if ctx:
                name = str(uuid.uuid4())
                ctx.create_remote_object(name, _IncrementalIndexRecorder, n_chunk)
                for chunk in chunks:
                    chunk.op.incremental_index_recorder_name = name

    @classmethod
    def pre_execute(cls, ctx: Union[dict, Context], op: OperandType):
        out = op.outputs[0]
        if (
            op.incremental_index
            and isinstance(out.index_value.value, IndexValue.RangeIndex)
            and getattr(op, "incremental_index_recorder_name", None)
        ):
            index = out.index[0]
            recorder_name = op.incremental_index_recorder_name
            recorder = ctx.get_remote_object(recorder_name)
            recorder.add_waiter(index)

    @classmethod
    def post_execute(cls, ctx: Union[dict, Context], op: OperandType):
        out = op.outputs[0]
        result = ctx[out.key]
        if (
            op.incremental_index
            and isinstance(out.index_value.value, IndexValue.RangeIndex)
            and getattr(op, "incremental_index_recorder_name", None)
        ):
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


def merge_small_files(
    df: TileableType,
    n_sample_file: int = 10,
    merged_file_size: Union[int, float, str] = None,
) -> TileableType:
    from ..merge import DataFrameConcat

    if len(df.chunks) < n_sample_file:
        # if number of chunks is small(less than `n_sample_file`,
        # skip this process
        return df

    if merged_file_size is not None:
        merged_file_size = parse_readable_size(merged_file_size)[0]
    else:
        # Estimated size is relatively large than the real one,
        # so we double the merged size
        merged_file_size = options.chunk_store_limit * 2
    # sample files whose size equals `n_sample_file`
    sampled_chunks = np.random.choice(df.chunks, n_sample_file)
    max_chunk_size = 0
    ctx = dict()
    for sampled_chunk in sampled_chunks:
        sampled_chunk.op.estimate_size(ctx, sampled_chunk.op)
        size = ctx[sampled_chunk.key][0]
        max_chunk_size = max(max_chunk_size, size)
    to_merge_size = merged_file_size // max_chunk_size
    if to_merge_size < 2:
        return df
    # merge files
    n_new_chunks = np.ceil(len(df.chunks) / to_merge_size)
    new_chunks = []
    new_nsplit = []
    for i, chunks in enumerate(np.array_split(df.chunks, n_new_chunks)):
        chunk_size = sum(c.shape[0] for c in chunks)
        kw = dict(
            dtypes=chunks[0].dtypes,
            index_value=merge_index_value({c.index: c.index_value for c in chunks}),
            columns_value=chunks[0].columns_value,
            shape=(chunk_size, chunks[0].shape[1]),
            index=(i, 0),
        )
        new_chunk = DataFrameConcat(output_types=[OutputType.dataframe]).new_chunk(
            chunks.tolist(), **kw
        )
        new_chunks.append(new_chunk)
        new_nsplit.append(chunk_size)
    new_op = df.op.copy()
    params = df.params.copy()
    params["chunks"] = new_chunks
    params["nsplits"] = (tuple(new_nsplit), df.nsplits[1])
    return new_op.new_dataframe(df.op.inputs, kws=[params])
