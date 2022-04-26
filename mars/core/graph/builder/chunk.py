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

import dataclasses
import functools
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from ....core import FUSE_CHUNK_TYPE, CHUNK_TYPE, TILEABLE_TYPE
from ....typing import EntityType, TileableType, ChunkType
from ....utils import copy_tileables, build_fetch
from ...entity.tileables import handler
from ...mode import enter_mode
from ..entity import TileableGraph, ChunkGraph
from .base import AbstractGraphBuilder


tile_gen_type = Generator[List[ChunkType], List[ChunkType], List[TileableType]]
DEFAULT_UPDATED_PROGRESS = 0.4


@dataclasses.dataclass
class _TileableHandler:
    tileable: TileableType
    handler: tile_gen_type
    last_need_processes: List[EntityType] = None


@dataclasses.dataclass
class _TileableTileInfo:
    curr_iter: int
    # incremental progress for this iteration
    tile_progress: float
    # newly generated chunks by a tileable in this iteration
    generated_chunks: List[ChunkType] = dataclasses.field(default_factory=list)


class TileContext(Dict[TileableType, TileableType]):
    _tileables = Set[TileableType]
    _tileable_to_progress: Dict[TileableType, float]
    _tileable_to_tile_infos: Dict[TileableType, List[_TileableTileInfo]]

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._tileables = None
        self._tileable_to_progress = dict()
        self._tileable_to_tile_infos = dict()

    def set_tileables(self, tileables: Set[TileableType]):
        self._tileables = tileables

    def __setitem__(self, key, value):
        self._tileable_to_progress.pop(key, None)
        return super().__setitem__(key, value)

    def set_progress(self, tileable: TileableType, progress: float):
        assert 0.0 <= progress <= 1.0
        last_progress = self._tileable_to_progress.get(tileable, 0.0)
        self._tileable_to_progress[tileable] = max(progress, last_progress)

    def get_progress(self, tileable: TileableType) -> float:
        if tileable in self:
            return 1.0
        else:
            return self._tileable_to_progress.get(tileable, 0.0)

    def get_all_progress(self) -> float:
        return sum(self.get_progress(t) for t in self._tileables) / len(self._tileables)

    def record_tileable_tile_info(
        self, tileable: TileableType, curr_iter: int, generated_chunks: List[ChunkType]
    ):
        if tileable not in self._tileable_to_tile_infos:
            self._tileable_to_tile_infos[tileable] = []
        prev_progress = sum(
            info.tile_progress for info in self._tileable_to_tile_infos[tileable]
        )
        curr_progress = self.get_progress(tileable)
        infos = self._tileable_to_tile_infos[tileable]
        infos.append(
            _TileableTileInfo(
                curr_iter=curr_iter,
                tile_progress=curr_progress - prev_progress,
                generated_chunks=generated_chunks,
            )
        )

    def get_tileable_tile_infos(self) -> Dict[TileableType, List[_TileableTileInfo]]:
        return {t: self._tileable_to_tile_infos.get(t, list()) for t in self._tileables}


@dataclasses.dataclass
class TileStatus:
    entities: List[EntityType] = None
    progress: float = None


class Tiler:
    _cur_iter: int
    _cur_chunk_graph: Optional[ChunkGraph]
    _tileable_handlers: Iterable[_TileableHandler]

    def __init__(
        self,
        tileable_graph: TileableGraph,
        tile_context: TileContext,
        processed_chunks: Set[ChunkType],
        chunk_to_fetch: Dict[ChunkType, ChunkType],
        add_nodes: Callable,
    ):
        self._tileable_graph = tileable_graph
        self._tile_context = tile_context
        self._processed_chunks = processed_chunks
        self._chunk_to_fetch = chunk_to_fetch
        self._add_nodes = self._wrap_add_nodes(add_nodes)
        self._curr_iter = 0
        self._cur_chunk_graph = None
        self._tileable_handlers = (
            _TileableHandler(tileable, self._tile_handler(tileable))
            for tileable in tileable_graph.topological_iter()
        )

    def _wrap_add_nodes(self, add_nodes: Callable):
        @functools.wraps(add_nodes)
        def inner(
            chunk_graph: ChunkGraph,
            chunks: List[ChunkType],
            visited: Set[ChunkType],
            tileable: TileableType,
        ):
            prev_chunks = set(chunk_graph)
            add_nodes(chunk_graph, chunks, visited)
            new_chunks = set(chunk_graph)
            self._tile_context.record_tileable_tile_info(
                tileable, self._curr_iter, list(new_chunks - prev_chunks)
            )

        return inner

    @staticmethod
    def _get_data(entity: EntityType):
        return entity.data if hasattr(entity, "data") else entity

    def _tile_handler(self, tileable: TileableType) -> tile_gen_type:
        from ....core.operand import Fetch

        tileable = self._get_data(tileable)

        if isinstance(tileable.op, Fetch) and not tileable.is_coarse():
            return [tileable]

        assert tileable.is_coarse()

        # copy tileable
        tiled_tileables = copy_tileables(
            tileable.op.outputs,
            inputs=[self._tile_context[inp] for inp in tileable.inputs],
            copy_key=True,
            copy_id=False,
        )
        tiled_tileables = [self._get_data(t) for t in tiled_tileables]
        # start to tile
        tiled_tileables = yield from handler.tile(tiled_tileables)
        return tiled_tileables

    def _gen_tileable_handlers(self, next_tileable_handlers: List[_TileableHandler]):
        for tile_handler in self._tileable_handlers:
            tileable, handler = tile_handler.tileable, tile_handler.handler
            if tileable in self._tile_context:
                continue
            if any(
                inp not in self._tile_context
                for inp in self._tileable_graph.predecessors(tileable)
            ):
                # predecessors not finished yet
                next_tileable_handlers.append(_TileableHandler(tileable, handler))
                continue

            yield _TileableHandler(tileable, handler)

    def _tile(
        self,
        chunk_graph: ChunkGraph,
        tileable: TileableType,
        tile_handler: tile_gen_type,
        next_tileable_handlers: List[_TileableHandler],
        to_update_tileables: List[TileableType],
        visited: Set[EntityType],
    ):
        try:
            need_process = next(tile_handler)

            if isinstance(need_process, TileStatus):
                # process tile that returns progress
                self._tile_context.set_progress(tileable, need_process.progress)
                need_process = need_process.entities
            else:
                # if progress not specified, we just update 0.4 * rest progress
                progress = self._tile_context.get_progress(tileable)
                new_progress = progress + (1.0 - progress) * DEFAULT_UPDATED_PROGRESS
                self._tile_context.set_progress(tileable, new_progress)

            chunks = []
            if need_process is not None:
                for t in need_process:
                    if isinstance(t, CHUNK_TYPE):
                        chunks.append(self._get_data(t))
                    elif isinstance(t, TILEABLE_TYPE):
                        to_update_tileables.append(self._get_data(t))
            # not finished yet
            self._add_nodes(chunk_graph, chunks.copy(), visited, tileable)
            next_tileable_handlers.append(
                _TileableHandler(tileable, tile_handler, need_process)
            )
            # add intermediate chunks into result chunks
            # to prevent them being pruned
            chunk_graph.result_chunks.extend(c for c in chunks if c in chunk_graph)
        except StopIteration as e:
            # tile done
            tiled_tileables = e.value
            for out, tiled_tileable in zip(tileable.op.outputs, tiled_tileables):
                out = self._get_data(out)
                tiled_tileable = self._get_data(tiled_tileable)

                chunks = tiled_tileable.chunks
                if chunks is None:  # pragma: no cover
                    raise ValueError(f"tileable({out}) is still coarse after tile")
                chunks = [self._get_data(c) for c in chunks]
                self._tile_context[out] = tiled_tileable
                self._add_nodes(chunk_graph, chunks, visited, tileable)

    def _gen_result_chunks(
        self,
        chunk_graph: ChunkGraph,
        next_tileable_handlers: List[_TileableHandler],
    ):
        result_chunks = chunk_graph.result_chunks
        tileable_graph = self._tileable_graph
        result_chunk_set = set(result_chunks)

        def _add_result_chunk(c):
            if c not in result_chunk_set:
                result_chunks.append(c)
                result_chunk_set.add(c)

        if next_tileable_handlers:
            for tileable_handler in next_tileable_handlers:
                tileable = tileable_handler.tileable
                # tileable that tile not completed, scan their inputs
                for inp_tileable in tileable_graph.iter_predecessors(tileable):
                    if (
                        tileable_handler.last_need_processes is None
                        or tileable_graph.count_successors(inp_tileable) > 1
                    ):
                        # if nothing yielded inside its tile,
                        # or the input has more than 1 successors,
                        # make sure their chunks in result,
                        # so that they will not be executed repeatedly
                        if inp_tileable in self._tile_context:
                            for chunk in self._tile_context[inp_tileable].chunks:
                                chunk = self._get_data(chunk)
                                if chunk in chunk_graph:
                                    _add_result_chunk(chunk)
        for tileable in tileable_graph.result_tileables:
            if tileable in self._tile_context:
                for chunk in self._tile_context[tileable].chunks:
                    chunk = self._get_data(chunk)
                    if chunk in chunk_graph:
                        _add_result_chunk(chunk)
                    if (
                        chunk in self._chunk_to_fetch
                        and self._chunk_to_fetch[chunk] in chunk_graph
                    ):
                        _add_result_chunk(self._chunk_to_fetch[chunk])

    def _iter(self):
        chunk_graph = self._cur_chunk_graph

        to_update_tileables = []
        visited = set()

        if chunk_graph is not None:
            # last tiled chunks, add them to processed
            # so that fetch chunk can be generated
            processed_chunks = [
                c.chunk if isinstance(c, FUSE_CHUNK_TYPE) else c
                for c in chunk_graph.result_chunks
            ]
            self._processed_chunks.update(processed_chunks)

        result_chunks = []
        chunk_graph = self._cur_chunk_graph = ChunkGraph(result_chunks)

        next_tileable_handlers = []
        # tile
        for tile_handler in self._gen_tileable_handlers(next_tileable_handlers):
            self._tile(
                chunk_graph,
                tile_handler.tileable,
                tile_handler.handler,
                next_tileable_handlers,
                to_update_tileables,
                visited,
            )
        self._tileable_handlers = next_tileable_handlers
        # gen result chunks
        self._gen_result_chunks(chunk_graph, next_tileable_handlers)
        # prune unused chunks
        prune_chunk_graph(chunk_graph)

        self._curr_iter += 1

        return to_update_tileables

    def __iter__(self):
        while self._tileable_handlers:
            to_update_tileables = self._iter()
            yield self._cur_chunk_graph
            for t in to_update_tileables:
                t.refresh_params()


def prune_chunk_graph(chunk_graph: ChunkGraph):
    from ....core.operand import Fetch, VirtualOperand, ShuffleProxy

    result_set = set(chunk_graph.result_chunks)
    stack = list(chunk_graph.result_chunks)
    used = set()
    while stack:
        n = stack.pop()
        if n in used:
            continue
        used.add(n)
        stack.extend(chunk_graph.predecessors(n))
        if isinstance(n.op, ShuffleProxy):
            stack.extend(
                succ for succ in chunk_graph.iter_successors(n) if succ not in used
            )

    unused = {n for n in chunk_graph if n not in used}
    for n in unused:
        # for pruned chunks, we assume we will use them later,
        # so we add the inputs of them into result chunks,
        # to prevent from duplicated submission
        for inp in chunk_graph.iter_predecessors(n):
            if (
                inp in used
                and inp not in result_set
                and not isinstance(inp.op, (Fetch, VirtualOperand))
            ):
                chunk_graph.result_chunks.append(inp)
                result_set.add(inp)
        # prune chunk
        chunk_graph.remove_node(n)


class ChunkGraphBuilder(AbstractGraphBuilder):
    _graph: TileableGraph

    def __init__(
        self,
        graph: TileableGraph,
        fuse_enabled: bool = True,
        tile_context: TileContext = None,
        tiler_cls: Union[Type[Tiler], Callable] = None,
    ):
        super().__init__(graph)
        self.fuse_enabled = fuse_enabled
        self.tile_context = TileContext() if tile_context is None else tile_context
        self.tile_context.set_tileables(set(graph))

        self._processed_chunks: Set[ChunkType] = set()
        self._chunk_to_fetch: Dict[ChunkType, ChunkType] = dict()

        tiler_cls = Tiler if tiler_cls is None else tiler_cls
        self.tiler = tiler_cls(
            self._graph,
            self.tile_context,
            self._processed_chunks,
            self._chunk_to_fetch,
            self._add_nodes,
        )

    def _process_node(self, entity: EntityType):
        if entity in self._processed_chunks:
            if entity not in self._chunk_to_fetch:
                # gen fetch
                fetch_chunk = build_fetch(entity).data
                self._chunk_to_fetch[entity] = fetch_chunk
            return self._chunk_to_fetch[entity]
        return entity

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

    def _if_add_node(self, node: EntityType, visited: Set):
        return node not in visited and node not in self._processed_chunks

    def _build(self) -> Iterable[Union[TileableGraph, ChunkGraph]]:
        tile_iterator = iter(self.tiler)
        while True:
            try:
                with enter_mode(build=True, kernel=True):
                    graph = next(tile_iterator)
                yield graph
            except StopIteration:
                break

    def build(self) -> Generator[Union[TileableGraph, ChunkGraph], None, None]:
        yield from self._build()
