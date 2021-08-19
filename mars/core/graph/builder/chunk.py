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

from typing import Callable, Dict, Generator, Iterable, \
    List, Optional, Set, Tuple, Type, Union

from ....core import FUSE_CHUNK_TYPE, CHUNK_TYPE, TILEABLE_TYPE
from ....typing import EntityType, TileableType, ChunkType
from ....utils import copy_tileables, build_fetch
from ...entity.tileables import handler
from ...mode import enter_mode
from ..entity import TileableGraph, ChunkGraph
from .base import AbstractGraphBuilder


tile_gen_type = Generator[List[ChunkType], List[ChunkType], List[TileableType]]


class Tiler:
    _cur_chunk_graph: Optional[ChunkGraph]
    _tileable_handlers: Iterable[Tuple[TileableType, tile_gen_type]]

    def __init__(self,
                 tileable_graph: TileableGraph,
                 tile_context: Dict[TileableType, TileableType],
                 processed_chunks: Set[ChunkType],
                 add_nodes: Callable):
        self._tileable_graph = tileable_graph
        self._tile_context = tile_context
        self._processed_chunks = processed_chunks
        self._add_nodes = add_nodes
        self._cur_chunk_graph = None
        self._tileable_handlers = (
            (tileable, self._tile_handler(tileable))
            for tileable in tileable_graph.topological_iter())

    @staticmethod
    def _get_data(entity: EntityType):
        return entity.data if hasattr(entity, 'data') else entity

    def _tile_handler(self,
                      tileable: TileableType) -> tile_gen_type:
        from ....core.operand import Fetch

        tileable = self._get_data(tileable)

        if isinstance(tileable.op, Fetch) and not tileable.is_coarse():
            return [tileable]

        assert tileable.is_coarse()

        # copy tileable
        tiled_tileables = copy_tileables(
            tileable.op.outputs,
            inputs=[self._tile_context[inp] for inp in tileable.inputs],
            copy_key=True, copy_id=False)
        tiled_tileables = [self._get_data(t) for t in tiled_tileables]
        # start to tile
        tiled_tileables = yield from handler.tile(tiled_tileables)
        return tiled_tileables

    def _gen_tileable_handlers(self,
                               next_tileable_handlers: List[Tuple[TileableType, tile_gen_type]]):
        for tileable, tile_handler in self._tileable_handlers:
            if tileable in self._tile_context:
                continue
            if any(inp not in self._tile_context
                   for inp in self._tileable_graph.predecessors(tileable)):
                # predecessors not finished yet
                next_tileable_handlers.append((tileable, tile_handler))
                continue

            yield tileable, tile_handler

    def _tile(self,
              chunk_graph: ChunkGraph,
              tileable: TileableType,
              tile_handler: tile_gen_type,
              next_tileable_handlers: List[Tuple[TileableType, tile_gen_type]],
              to_update_tileables: List[TileableType],
              visited: Set[EntityType]):
        try:
            need_process = next(tile_handler)
            chunks = []
            if need_process is not None:
                for t in need_process:
                    if isinstance(t, CHUNK_TYPE):
                        chunks.append(self._get_data(t))
                    elif isinstance(t, TILEABLE_TYPE):
                        to_update_tileables.append(self._get_data(t))
            # not finished yet
            self._add_nodes(chunk_graph, chunks.copy(), visited)
            next_tileable_handlers.append((tileable, tile_handler))
            # add intermediate chunks into result chunks
            # to prevent them being pruned
            chunk_graph.result_chunks.extend(chunks)
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
                self._tile_context[out] = tiled_tileable

    def _gen_result_chunks(self,
                           chunk_graph: ChunkGraph,
                           next_tileable_handlers: List[Tuple[TileableType, tile_gen_type]]):
        result_chunks = chunk_graph.result_chunks
        tileable_graph = self._tileable_graph
        # generate result chunks
        result_chunk_set = set()
        if next_tileable_handlers:
            # add all chunks that have no successors to result chunks
            for chunk in chunk_graph:
                if chunk_graph.count_successors(chunk) == 0:
                    if chunk not in result_chunk_set:
                        result_chunks.append(chunk)
                        result_chunk_set.add(chunk)
            for tileable, _ in next_tileable_handlers:
                # tileable that tile not completed,
                # scan inputs to make sure their chunks in result
                for inp_tileable in tileable_graph.predecessors(tileable):
                    if inp_tileable in self._tile_context:
                        for chunk in self._tile_context[inp_tileable].chunks:
                            chunk = self._get_data(chunk)
                            if chunk in chunk_graph and \
                                    chunk not in result_chunk_set:
                                result_chunks.append(chunk)
                                result_chunk_set.add(chunk)
        for tileable in tileable_graph.result_tileables:
            if tileable in self._tile_context:
                for chunk in self._tile_context[tileable].chunks:
                    chunk = self._get_data(chunk)
                    if chunk in chunk_graph and \
                            chunk not in result_chunk_set:
                        result_chunks.append(chunk)
                        result_chunk_set.add(chunk)

    def _iter(self):
        chunk_graph = self._cur_chunk_graph

        to_update_tileables = []
        visited = set()

        if chunk_graph is not None:
            # last tiled chunks, add them to processed
            # so that fetch chunk can be generated
            processed_chunks = [
                c.chunk if isinstance(c, FUSE_CHUNK_TYPE) else c
                for c in chunk_graph.result_chunks]
            self._processed_chunks.update(processed_chunks)

        result_chunks = []
        chunk_graph = self._cur_chunk_graph = ChunkGraph(result_chunks)

        next_tileable_handlers = []
        # tile
        for tileable, tile_handler in \
                self._gen_tileable_handlers(next_tileable_handlers):
            self._tile(chunk_graph, tileable, tile_handler,
                       next_tileable_handlers, to_update_tileables, visited)
        self._tileable_handlers = next_tileable_handlers
        # gen result chunks
        self._gen_result_chunks(chunk_graph, next_tileable_handlers)

        return to_update_tileables

    def __iter__(self):
        while self._tileable_handlers:
            to_update_tileables = self._iter()
            yield self._cur_chunk_graph
            for t in to_update_tileables:
                t.refresh_params()


class ChunkGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(self,
                 graph: TileableGraph,
                 fuse_enabled: bool = True,
                 tile_context: Dict[TileableType, TileableType] = None,
                 tiler_cls: Union[Type[Tiler], Callable] = None):
        super().__init__(graph)
        self.fuse_enabled = fuse_enabled
        self.tile_context = dict() if tile_context is None else tile_context

        self._processed_chunks: Set[ChunkType] = set()
        self._chunk_to_fetch: Dict[ChunkType, ChunkType] = dict()

        tiler_cls = Tiler if tiler_cls is None else tiler_cls
        self.tiler = tiler_cls(self._graph, self.tile_context,
                               self._processed_chunks, self._add_nodes)

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
        yield from self.tiler

    def build(self) -> Generator[Union[TileableGraph, ChunkGraph], None, None]:
        with enter_mode(build=True, kernel=True):
            yield from self._build()
