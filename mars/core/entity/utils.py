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

from typing import List, Union, Generator

from ...typing import TileableType, ChunkType
from ...utils import has_unknown_shape, calc_nsplits


def refresh_tileable_shape(tileable):
    if has_unknown_shape(tileable):
        # update shape
        nsplits = calc_nsplits(
            {c.index: c.shape for c in tileable.chunks})
        shape = tuple(sum(ns) for ns in nsplits)
        tileable._nsplits = nsplits
        tileable._shape = shape


def tile(tileable, *tileables: TileableType):
    from ..graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder

    raw_tileables = target_tileables = [tileable] + list(tileables)
    target_tileables = [t.data if hasattr(t, 'data') else t
                        for t in target_tileables]

    tileable_graph = TileableGraph(target_tileables)
    tileable_graph_builder = TileableGraphBuilder(tileable_graph)
    next(tileable_graph_builder.build())

    # tile
    tile_context = dict()
    chunk_graph_builder = ChunkGraphBuilder(
        tileable_graph, fuse_enabled=False, tile_context=tile_context)
    next(chunk_graph_builder.build())

    if len(tileables) == 0:
        return type(tileable)(tile_context[target_tileables[0]])
    else:
        return [type(raw_t)(tile_context[t]) for raw_t, t
                in zip(raw_tileables, target_tileables)]


def recursive_tile(tileable: TileableType, *tileables: TileableType) -> \
        Generator[List[ChunkType], List[ChunkType],
                  Union[TileableType, List[TileableType]]]:
    from .tileables import handler

    return_list = len(tileables) > 0
    if not return_list and isinstance(tileable, (list, tuple)):
        return_list = True
        raw = tileable
        tileable = raw[0]
        tileables = raw[1:]

    inputs_set = set(tileable.op.inputs)
    to_tile = [tileable] + list(tileables)
    q = [t for t in to_tile if t.is_coarse()]
    while q:
        t = q[-1]
        cs = [c for c in t.inputs if c.is_coarse()]
        if cs:
            q.extend(cs)
            continue
        for obj in handler.tile(t.op.outputs):
            to_update_inputs = []
            chunks = []
            for inp in t.op.inputs:
                if has_unknown_shape(inp):
                    to_update_inputs.append(inp)
                if inp not in inputs_set:
                    chunks.extend(inp.chunks)
            if obj is None:
                yield chunks + to_update_inputs
            else:
                yield obj + to_update_inputs
        q.pop()

    if not return_list:
        return tileable
    else:
        return [tileable] + list(tileables)
