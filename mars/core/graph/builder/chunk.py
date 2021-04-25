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

import inspect
from typing import List, Dict, Union, Set, Generator, Iterable

from ....utils import copy_tileables, build_fetch
from ...entity.tileables import handler, TilesError
from ...mode import enter_mode
from ...typing import EntityType, TileableType, ChunkType
from ..entity import TileableGraph, ChunkGraph
from .base import AbstractGraphBuilder


class ChunkGraphBuilder(AbstractGraphBuilder):
    _graph = TileableGraph

    def __init__(self,
                 graph: TileableGraph,
                 fuse_enabled: bool = True,
                 tile_context: Dict[TileableType, TileableType] = None):
        super().__init__(graph)
        self.fuse_enabled = fuse_enabled
        self.tile_context = dict() if tile_context is None else tile_context

        self._processed_chunks: Set[ChunkType] = set()
        self._chunk_to_fetch: Dict[ChunkType, ChunkType] = dict()

    @staticmethod
    def _get_data(entity: EntityType):
        return entity.data if hasattr(entity, 'data') else entity

    def _tile(self,
              tileable: TileableType) -> \
            Generator[List[ChunkType], List[ChunkType], List[TileableType]]:
        tileable = self._get_data(tileable)
        assert tileable.is_coarse()

        # copy tileable
        tiled_tileables = copy_tileables(
            tileable.op.outputs,
            inputs=[self.tile_context[inp] for inp in tileable.inputs],
            copy_key=True, copy_id=False)
        tiled_tileables = [self._get_data(t) for t in tiled_tileables]

        # start to tile
        # get tile handler
        op = tiled_tileables[0].op
        tile_handler = handler.get_handler(op)
        if inspect.isgeneratorfunction(tile_handler):
            # new style tile,
            # op.tile can be a generator function,
            # each time an operand yield some chunks,
            # they will be put into ChunkGraph and executed first.
            # After execution, resume from the yield place.
            tiled_result = yield from tile_handler(op)
        else:
            # old style tile
            # op.tile raise TilesError to submit predecessors first.
            while True:
                try:
                    tiled_result = tile_handler(op)
                    break
                except TilesError as e:
                    # failed
                    if getattr(e, 'partial_tiled_chunks', None):
                        yield e.partial_tiled_chunks
                    else:
                        yield []

        tiled_results = [self._get_data(t) for t in tiled_result]
        assert len(tiled_tileables) == len(tiled_results)
        for tileable, tiled_result in zip(tiled_tileables, tiled_results):
            tiled_result.copy_to(tileable)
            tileable.op.outputs = tiled_tileables

        return tiled_tileables

    def _select_inputs(self, inputs: List[ChunkType]):
        new_inputs = []
        for inp in inputs:
            if inp in self._processed_chunks:
                # gen fetch
                if inp not in self._chunk_to_fetch:
                    fetch_chunk = build_fetch(inp).data
                    self._chunk_to_fetch[inp] = fetch_chunk
                new_inputs.append(self._chunk_to_fetch[inp])
            else:
                new_inputs.append(inp)
        return new_inputs

    def _build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
        tileable_graph = self._graph
        tileables = tileable_graph.result_tileables

        process_tile_iter = (
            (tileable, self._tile(tileable))
            for tileable in tileable_graph.topological_iter())
        visited = set()
        chunk_graph = None
        while True:
            if chunk_graph is not None:
                # last tiled chunks, add them to processed
                # so that fetch chunk can be generated
                self._processed_chunks.update(chunk_graph)

            result_chunks = []
            chunk_graph = ChunkGraph(result_chunks)

            need_process_tiles = []
            for tileable, process_tile in process_tile_iter:
                if tileable in self.tile_context:
                    continue

                try:
                    chunks = next(process_tile)
                    if chunks is None:
                        chunks = []
                    chunks = [self._get_data(c) for c in chunks]
                    # not finished yet
                    self._add_nodes(chunk_graph, chunks, visited)
                    need_process_tiles.append((tileable, process_tile))
                    # add intermediate chunks into result chunks
                    # to prevent them being pruned
                    result_chunks.extend(chunks)
                except StopIteration as e:
                    # tile done
                    tiled_tileables = e.value
                    for out, tiled_tileable in zip(tileable.op.outputs, tiled_tileables):
                        out = self._get_data(out)
                        tiled_tileable = self._get_data(tiled_tileable)

                        chunks = tiled_tileable.chunks
                        chunks = [self._get_data(c) for c in chunks]
                        self._add_nodes(chunk_graph, chunks, visited)
                        self.tile_context[out] = tiled_tileable

            if not need_process_tiles:
                # tile finished
                for tileable in tileables:
                    # add chunks that belongs to result tileables
                    # to result chunks
                    chunks = self.tile_context[tileable].chunks
                    for chunk in chunks:
                        chunk = self._get_data(chunk)
                        if chunk in chunk_graph:
                            result_chunks.append(chunk)
            else:
                process_tile_iter = need_process_tiles
                # otherwise, add all chunks that have no successors
                # to result chunks
                result_chunk_set = set()
                for chunk in chunk_graph:
                    if chunk_graph.count_successors(chunk) == 0:
                        if chunk not in result_chunk_set:
                            result_chunks.append(chunk)
                            result_chunk_set.add(chunk)
                for tileable in tileables:
                    if tileable in self.tile_context:
                        for chunk in self.tile_context[tileable].chunks:
                            if chunk in chunk_graph and \
                                    chunk not in result_chunk_set:
                                result_chunks.append(chunk)
                                result_chunk_set.add(chunk)

            yield chunk_graph

            if not need_process_tiles:
                break

    def build(self) -> Generator[Union[TileableGraph, ChunkGraph], None, None]:
        with enter_mode(build=True, kernel=True):
            yield from self._build()
