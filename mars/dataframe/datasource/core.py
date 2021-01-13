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

from ...context import get_context
from ...serialize import Int64Field, KeyField
from ...tiles import TilesError
from ..operands import DataFrameOperand, DataFrameOperandMixin


class HeadOptimizedDataSource(DataFrameOperand, DataFrameOperandMixin):
    __slots__ = '_tiled',
    # Data source op that optimized for head,
    # First, it will try to trigger first_chunk.head() and raise TilesError,
    # When iterative tiling is triggered,
    # check if the first_chunk.head() meets requirements.
    _nrows = Int64Field('nrows')
    # for chunk
    _first_chunk = KeyField('first_chunk')

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
        if op.first_chunk is None:
            op._tiled = tiled = cls._tile(op)
            chunks = tiled[0].chunks

            err = TilesError('HeadOrTailOptimizeDataSource requires '
                             'some dependencies executed first')
            op._first_chunk = chunk = chunks[0]
            err.partial_tiled_chunks = [chunk.data]
            raise err
        else:
            tiled = op._tiled
            chunks = tiled[0].chunks
            del op._tiled

            ctx = get_context()
            chunk_shape = ctx.get_chunk_metas([op.first_chunk.key])[0].chunk_shape

            # reset first chunk
            op._first_chunk = None
            for c in chunks:
                c.op._first_chunk = None

            if chunk_shape[0] == op.nrows:
                # the first chunk has enough data
                tiled[0]._nsplits = tuple((s,) for s in chunk_shape)
                chunks[0]._shape = chunk_shape
                tiled[0]._chunks = chunks[:1]
                tiled[0]._shape = chunk_shape
            else:
                for chunk in tiled[0].chunks:
                    chunk.op._nrows = None
                # otherwise
                tiled = [tiled[0].iloc[:op.nrows]._inplace_tile()]
            return tiled

    @classmethod
    def tile(cls, op: "HeadOptimizedDataSource"):
        if op.nrows is not None:
            return cls._tile_head(op)
        else:
            return cls._tile(op)


class ColumnPruneSupportedDataSourceMixin(DataFrameOperandMixin):
    __slots__ = ()

    def get_columns(self):  # pragma: no cover
        raise NotImplementedError

    def set_pruned_columns(self, columns):  # pragma: no cover
        raise NotImplementedError
