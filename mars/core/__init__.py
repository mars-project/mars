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

# noinspection PyUnresolvedReferences
from ..typing import ChunkType, TileableType, EntityType, OperandType
from .entity import Entity, EntityData, ENTITY_TYPE, \
    Chunk, ChunkData, CHUNK_TYPE, \
    Tileable, TileableData, TILEABLE_TYPE, \
    Object, ObjectData, ObjectChunk, ObjectChunkData, OBJECT_TYPE, OBJECT_CHUNK_TYPE, \
    FuseChunk, FuseChunkData, FUSE_CHUNK_TYPE, \
    OutputType, register_output_types, get_output_types, \
    register_fetch_class, get_fetch_class, get_tileable_types, get_chunk_types, \
    HasShapeTileable, HasShapeTileableData, ExecutableTuple, _ExecuteAndFetchMixin, \
    NotSupportTile, register, unregister, tile, recursive_tile
# noinspection PyUnresolvedReferences
from .graph import DirectedGraph, DAG, GraphContainsCycleError, \
    TileableGraph, ChunkGraph, TileableGraphBuilder, ChunkGraphBuilder
from .mode import enter_mode, is_build_mode, is_eager_mode, is_kernel_mode
