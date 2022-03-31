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
import logging
from functools import partial
from typing import Callable, Dict, List, Iterable, Set

from ....config import Config
from ....core import TileableGraph, ChunkGraph, ChunkGraphBuilder
from ....core.graph.builder.chunk import Tiler, _TileableHandler
from ....core.operand import Fetch
from ....resource import Resource
from ....typing import TileableType, ChunkType
from ....optimization.logical.chunk import optimize as optimize_chunk_graph
from ....optimization.logical.tileable import optimize as optimize_tileable_graph
from ....typing import BandType
from ...subtask import SubtaskGraph
from ..analyzer import GraphAnalyzer
from ..core import Task

logger = logging.getLogger(__name__)


class CancellableTiler(Tiler):
    def __init__(
        self,
        tileable_graph: TileableGraph,
        tile_context: Dict[TileableType, TileableType],
        processed_chunks: Set[ChunkType],
        chunk_to_fetch: Dict[ChunkType, ChunkType],
        add_nodes: Callable,
        cancelled: asyncio.Event = None,
        check_duplicated_submission: bool = False,
    ):
        super().__init__(
            tileable_graph, tile_context, processed_chunks, chunk_to_fetch, add_nodes
        )
        self._cancelled = cancelled
        self._check_duplicated_submission = check_duplicated_submission

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def _gen_tileable_handlers(self, next_tileable_handlers: List[_TileableHandler]):
        for tile_handler in super()._gen_tileable_handlers(next_tileable_handlers):
            if not self.cancelled:
                yield tile_handler
            else:
                break

    def _gen_result_chunks(
        self,
        chunk_graph: ChunkGraph,
        next_tileable_handlers: List[_TileableHandler],
    ):
        if not self.cancelled:
            return super()._gen_result_chunks(chunk_graph, next_tileable_handlers)
        else:
            return

    def _iter_without_check(self):
        while self._tileable_handlers:
            to_update_tileables = self._iter()
            if not self.cancelled:
                yield self._cur_chunk_graph
            if not self.cancelled:
                for t in to_update_tileables:
                    t.refresh_params()
            else:
                break

    def _iter_with_check(self):
        chunk_set = set()
        chunk_graphs = []
        for chunk_graph in self._iter_without_check():
            chunk_graphs.append(chunk_graph)
            chunks = []
            for chunk in chunk_graph:
                if isinstance(chunk.op, Fetch):
                    continue
                if chunk in chunk_set:
                    raise RuntimeError(f"chunk {chunk} submitted repeatedly")
                chunks.append(chunk)
            chunk_set.update(chunks)
            yield chunk_graph

    def __iter__(self):
        if not self._check_duplicated_submission:
            return self._iter_without_check()
        else:
            return self._iter_with_check()


class TaskPreprocessor:
    __slots__ = (
        "_task",
        "tileable_graph",
        "tile_context",
        "_config",
        "tileable_optimization_records",
        "chunk_optimization_records_list",
        "_cancelled",
        "_done",
    )

    tile_context: Dict[TileableType, TileableType]

    def __init__(
        self,
        task: Task,
        tiled_context: Dict[TileableType, TileableType] = None,
        config: Config = None,
    ):
        self._task = task
        self.tileable_graph = task.tileable_graph
        self._config = config

        self.tile_context = tiled_context
        self.tileable_optimization_records = None
        self.chunk_optimization_records_list = []

        self._cancelled = asyncio.Event()
        self._done = asyncio.Event()

    def optimize(self) -> TileableGraph:
        """
        Optimize tileable graph.

        Returns
        -------
        optimized_graph: TileableGraph

        """
        if self._config.optimize_tileable_graph:
            # enable optimization
            self.tileable_optimization_records = optimize_tileable_graph(
                self.tileable_graph
            )
        return self.tileable_graph

    def _fill_fetch_tileable_with_chunks(self, tileable_graph: TileableGraph):
        for t in tileable_graph:
            if isinstance(t.op, Fetch) and t in self.tile_context:
                tiled = self.tile_context[t]
                t._chunks = tiled.chunks
                t._nsplits = tiled.nsplits

    def _get_tiler_cls(self) -> Callable:
        extra_config = self._task.extra_config or dict()
        check_duplicated_submission = extra_config.get(
            "check_duplicated_submission", False
        )
        return partial(
            CancellableTiler,
            cancelled=self._cancelled,
            check_duplicated_submission=check_duplicated_submission,
        )

    def tile(self, tileable_graph: TileableGraph) -> Iterable[ChunkGraph]:
        """
        Generate chunk graphs

        Returns
        -------
        chunk_graph_generator: Generator
             Chunk graphs.
        """
        self._fill_fetch_tileable_with_chunks(tileable_graph)
        # iterative chunk graph builder
        chunk_graph_builder = ChunkGraphBuilder(
            tileable_graph,
            fuse_enabled=self._task.fuse_enabled,
            tile_context=self.tile_context,
            tiler_cls=self._get_tiler_cls(),
        )
        optimize = self._config.optimize_chunk_graph
        meta_updated = set()
        for chunk_graph in chunk_graph_builder.build():
            if len(chunk_graph) == 0:
                continue
            # optimize chunk graph
            if optimize:
                self.chunk_optimization_records_list.append(
                    optimize_chunk_graph(chunk_graph)
                )
            yield chunk_graph
            # update tileables' meta
            self._update_tileables_params(tileable_graph, meta_updated)

    def analyze(
        self,
        chunk_graph: ChunkGraph,
        available_bands: Dict[BandType, Resource],
        stage_id: str = None,
    ) -> SubtaskGraph:
        logger.debug("Start to gen subtask graph for task %s", self._task.task_id)
        task = self._task
        analyzer = GraphAnalyzer(
            chunk_graph, available_bands, task, self._config, stage_id=stage_id
        )
        graph = analyzer.gen_subtask_graph()
        logger.debug(
            "Generated subtask graph of %s subtasks for task %s",
            len(graph),
            self._task.task_id,
        )
        return graph

    def _get_done(self):
        return self._done.is_set()

    def _set_done(self, is_done: bool):
        if is_done:
            self._done.set()
        else:  # pragma: no cover
            self._done.clear()

    done = property(_get_done, _set_done)

    def cancel(self):
        self._cancelled.set()

    def get_tiled(self, tileable: TileableType):
        tileable = tileable.data if hasattr(tileable, "data") else tileable
        return self.tile_context[tileable]

    def _update_tileable_params(
        self, tileable: TileableType, tiled: TileableType  # pylint: disable=no-self-use
    ):
        tiled.refresh_params()
        tileable.params = tiled.params

    def _update_tileables_params(
        self, tileable_graph: TileableGraph, updated: Set[TileableType]
    ):
        for tileable in tileable_graph:
            if tileable in updated:
                continue
            tiled_tileable = self.tile_context.get(tileable)
            if tiled_tileable is not None:
                self._update_tileable_params(tileable, tiled_tileable)
                updated.add(tileable)

    def __await__(self):
        return self._done.wait().__await__()
