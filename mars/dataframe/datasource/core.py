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

from ...core import recursive_tile
from ...core.context import get_context
from ...serialization.serializables import Int64Field
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
