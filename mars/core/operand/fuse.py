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

from ... import opcodes
from ...serialization.serializables import ReferenceField
from ..entity import FuseChunk, FuseChunkData, NotSupportTile
from ..graph import ChunkGraph
from .base import Operand


class Fuse(Operand):
    __slots__ = '_fuse_graph',
    _op_type_ = opcodes.FUSE

    fuse_graph = ReferenceField('fuse_graph', ChunkGraph)


class FuseChunkMixin:
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _op=self, **kw)
        return FuseChunk(data)

    @classmethod
    def tile(cls, op):
        raise NotSupportTile('FuseChunk is a chunk operand '
                             'which does not support tile')
