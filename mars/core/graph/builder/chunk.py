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

from typing import List, Dict, Union, Set, Generator, Iterable

from ....core import FUSE_CHUNK_TYPE, CHUNK_TYPE, TILEABLE_TYPE
from ....typing import EntityType, TileableType, ChunkType
from ....utils import copy_tileables, build_fetch
from ...entity.tileables import handler
from ...mode import enter_mode
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
        from ....core.operand import Fetch

        tileable = self._get_data(tileable)

        if isinstance(tileable.op, Fetch) and not tileable.is_coarse():
            return [tileable]

        assert tileable.is_coarse()

        # copy tileable
        tiled_tileables = copy_tileables(
            tileable.op.outputs,
            inputs=[self.tile_context[inp] for inp in tileable.inputs],
            copy_key=True, copy_id=False)
        tiled_tileables = [self._get_data(t) for t in tiled_tileables]
        # start to tile
        tiled_tileables = yield from handler.tile(tiled_tileables)
        return tiled_tileables

    def _select_inputs(self, inputs: List[ChunkType]):
        new_inputs = []
        for inp in inputs:
            # TODO: remove it when fuse chunk is deprecated
            if isinstance(inp, FUSE_CHUNK_TYPE):
                inp = inp.chunk
            if inp in self._processed_chunks:
                # gen fetch
                if inp not in self._chunk_to_fetch:
                    fetch_chunk = build_fetch(inp).data
                    self._chunk_to_fetch[inp] = fetch_chunk
                new_inputs.append(self._chunk_to_fetch[inp])
            else:
                new_inputs.append(inp)
        return new_inputs

    def _if_add_node(self, node: EntityType, visited: Set):
        return node not in visited and node not in self._processed_chunks

    def _build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
        tileable_graph = self._graph
        tileables = tileable_graph.result_tileables

        process_tile_iter = (
            (tileable, self._tile(tileable))
            for tileable in tileable_graph.topological_iter())
        chunk_graph = None
        while True:
            to_be_updated_tileables = []
            visited = set()
            if chunk_graph is not None:
                # last tiled chunks, add them to processed
                # so that fetch chunk can be generated
                processed_chunks = [
                    c.chunk if isinstance(c, FUSE_CHUNK_TYPE) else c
                    for c in chunk_graph.result_chunks]
                self._processed_chunks.update(processed_chunks)

            result_chunks = []
            chunk_graph = ChunkGraph(result_chunks)

            need_process_tiles = []
            for tileable, process_tile in process_tile_iter:
                if tileable in self.tile_context:
                    continue
                if any(inp not in self.tile_context
                       for inp in tileable_graph.predecessors(tileable)):
                    # predecessors not finished yet
                    need_process_tiles.append((tileable, process_tile))
                    continue

                try:
                    need_processed = next(process_tile)
                    if need_processed is None:
                        chunks = []
                    else:
                        chunks = [self._get_data(c) for c in need_processed
                                  if isinstance(c, CHUNK_TYPE)]
                        to_be_updated_tileables.extend([t for t in need_processed
                                                        if isinstance(t, TILEABLE_TYPE)])
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
                        if chunks is None:  # pragma: no cover
                            raise ValueError(f'tileable({out}) is still coarse '
                                             f'after tile')
                        chunks = [self._get_data(c) for c in chunks]
                        self._add_nodes(chunk_graph, chunks, visited)
                        self.tile_context[out] = tiled_tileable

            # generate result chunks
            result_chunk_set = set()
            if need_process_tiles:
                process_tile_iter = need_process_tiles
                # otherwise, add all chunks that have no successors
                # to result chunks
                for chunk in chunk_graph:
                    if chunk_graph.count_successors(chunk) == 0:
                        if chunk not in result_chunk_set:
                            result_chunks.append(chunk)
                            result_chunk_set.add(chunk)
                for tileable, _ in need_process_tiles:
                    # tileable that tile not completed,
                    # scan inputs to make sure their chunks in result
                    for inp_tileable in tileable_graph.predecessors(tileable):
                        if inp_tileable in self.tile_context:
                            for chunk in self.tile_context[inp_tileable].chunks:
                                chunk = self._get_data(chunk)
                                if chunk in chunk_graph and \
                                        chunk not in result_chunk_set:
                                    result_chunks.append(chunk)
                                    result_chunk_set.add(chunk)
            for tileable in tileables:
                if tileable in self.tile_context:
                    for chunk in self.tile_context[tileable].chunks:
                        chunk = self._get_data(chunk)
                        if chunk in chunk_graph and \
                                chunk not in result_chunk_set:
                            result_chunks.append(chunk)
                            result_chunk_set.add(chunk)

            # yield chunk graph for upcoming optimization and execution
            yield chunk_graph

            for t in to_be_updated_tileables:
                t.refresh_params()
            if not need_process_tiles:
                break

    def build(self) -> Generator[Union[TileableGraph, ChunkGraph], None, None]:
        with enter_mode(build=True, kernel=True):
            yield from self._build()
