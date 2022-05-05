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
import os
import tempfile
import time
from typing import Dict, Iterator, Optional, List, Set

from ....core import ChunkGraph, TileableGraph, Chunk, TileContext
from ....core.operand import Fetch
from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from ....metrics import Metrics
from ....optimization.logical import OptimizationRecords
from ....oscar.profiling import (
    ProfilingData,
    MARS_ENABLE_PROFILING,
)
from ....tensor.core import TENSOR_TYPE
from ....typing import TileableType, ChunkType
from ....utils import Timer
from ...subtask import SubtaskResult, Subtask
from ..core import Task, TaskResult, TaskStatus, new_task_id
from ..execution.api import TaskExecutor, ExecutionChunkResult
from .preprocessor import TaskPreprocessor

logger = logging.getLogger(__name__)

MARS_ENABLE_DUMPING_SUBTASK_GRAPH = int(os.environ.get("MARS_DUMP_SUBTASK_GRAPH", 0))


class TaskProcessor:
    _tileable_to_subtasks: Dict[TileableType, List[Subtask]]
    _tileable_id_to_tileable: Dict[str, TileableType]
    _chunk_to_subtasks: Dict[ChunkType, Subtask]
    _stage_tileables: Set[TileableType]

    def __init__(
        self,
        task: Task,
        preprocessor: TaskPreprocessor,
        executor: TaskExecutor,
    ):
        self._task = task
        self._preprocessor = preprocessor
        self._executor = executor

        self._tileable_id_to_tileable = dict()
        self._chunk_to_subtasks = dict()
        self._stage_tileables = set()

        if MARS_ENABLE_PROFILING:
            ProfilingData.init(task.task_id)
        elif task.extra_config and task.extra_config.get("enable_profiling"):
            ProfilingData.init(task.task_id, task.extra_config["enable_profiling"])

        self._dump_subtask_graph = False
        if MARS_ENABLE_DUMPING_SUBTASK_GRAPH or (
            task.extra_config and task.extra_config.get("dump_subtask_graph")
        ):
            self._dump_subtask_graph = True

        self.result = TaskResult(
            task_id=task.task_id,
            session_id=task.session_id,
            start_time=time.time(),
            status=TaskStatus.pending,
        )
        self.done = asyncio.Event()

        # add metrics
        self._chunk_graph_gen_time = Metrics.gauge(
            "mars.chunk_graph_gen_time_secs",
            "Time consuming in seconds to generate a chunk graph",
            ("session_id", "task_id"),
        )
        self._subtask_graph_gen_time = Metrics.gauge(
            "mars.subtask_graph_gen_time_secs",
            "Time consuming in seconds to generate a subtask graph",
            ("session_id", "task_id", "stage_id"),
        )
        self._task_execution_time = Metrics.gauge(
            "mars.task_execution_time_secs",
            "Time consuming in seconds to execute a task",
            ("session_id", "task_id"),
        )

    @property
    def task_id(self):
        return self._task.task_id

    @property
    def tileable_graph(self):
        return self._preprocessor.tileable_graph

    @property
    def tileable_id_to_tileable(self):
        return self._tileable_id_to_tileable

    @property
    def tile_context(self) -> TileContext:
        return self._preprocessor.tile_context

    @property
    def stage_processors(self):
        # TODO(fyrestone): Remove it.
        return self._executor.get_stage_processors()

    def get_tiled(self, tileable: TileableType):
        return self._preprocessor.get_tiled(tileable)

    def get_subtasks(self, chunks: List[ChunkType]) -> List[Subtask]:
        return [self._chunk_to_subtasks[chunk] for chunk in chunks]

    def get_tileable_to_subtasks(self) -> Dict[TileableType, List[Subtask]]:
        tile_context = self.tile_context
        result = dict()
        for tileable, infos in tile_context.get_tileable_tile_infos().items():
            subtasks = []
            for info in infos:
                chunks = [
                    c for c in info.generated_chunks if not isinstance(c.op, Fetch)
                ]
                subtasks.extend(self.get_subtasks(chunks))
            result[tileable] = subtasks
        return result

    @staticmethod
    async def _get_next_chunk_graph(
        chunk_graph_iter: Iterator[ChunkGraph],
    ) -> Optional[ChunkGraph]:
        def next_chunk_graph():
            try:
                return next(chunk_graph_iter)
            except StopIteration:
                return

        fut = asyncio.to_thread(next_chunk_graph)
        chunk_graph = await fut
        return chunk_graph

    async def _iter_stage_chunk_graph(self):
        tileable_graph = self._preprocessor.tileable_graph
        chunk_graph_iter = iter(self._preprocessor.tile(tileable_graph))
        while True:
            with Timer() as stage_timer:
                with Timer() as timer:
                    chunk_graph = await self._get_next_chunk_graph(chunk_graph_iter)
                    if chunk_graph is None:
                        # tile finished
                        self._preprocessor.done = True
                        return
                stage_id = new_task_id()
                stage_profiler = ProfilingData[self._task.task_id, "general"].nest(
                    f"stage_{stage_id}"
                )
                stage_profiler.set(f"tile({len(chunk_graph)})", timer.duration)
                logger.info(
                    "Time consuming to gen a chunk graph is %ss with session id %s, task id %s",
                    timer.duration,
                    self._task.session_id,
                    self._task.task_id,
                )
                self._chunk_graph_gen_time.record(
                    timer.duration,
                    {
                        "session_id": self._task.session_id,
                        "task_id": self._task.task_id,
                    },
                )
                yield stage_id, stage_profiler, chunk_graph

            stage_profiler.set("total", stage_timer.duration)

    async def _process_stage_chunk_graph(
        self,
        stage_id: str,
        stage_profiler,
        chunk_graph: ChunkGraph,
    ):
        available_bands = await self._executor.get_available_band_resources()
        meta_api = self._executor._meta_api
        get_meta_tasks = []
        fetch_op_keys = []
        for c in chunk_graph.iter_indep():
            if isinstance(c.op, Fetch):
                get_meta_tasks.append(
                    meta_api.get_chunk_meta.delay(c.key, fields=["bands"])
                )
                fetch_op_keys.append(c.op.key)
        # TODO(fyrestone): A more general way to get the key to bands
        # for all execution backends.
        try:
            key_to_bands = await meta_api.get_chunk_meta.batch(*get_meta_tasks)
            fetch_op_to_bands = dict(
                (key, meta["bands"][0])
                for key, meta in zip(fetch_op_keys, key_to_bands)
            )
        except (KeyError, IndexError):
            fetch_op_to_bands = {}
        with Timer() as timer:
            subtask_graph = await asyncio.to_thread(
                self._preprocessor.analyze,
                chunk_graph,
                self._chunk_to_subtasks,
                available_bands,
                stage_id=stage_id,
                op_to_bands=fetch_op_to_bands,
            )
        stage_profiler.set(f"gen_subtask_graph({len(subtask_graph)})", timer.duration)
        logger.info(
            "Time consuming to gen a subtask graph is %ss with session id %s, task id %s, stage id %s",
            timer.duration,
            self._task.session_id,
            self._task.task_id,
            stage_id,
        )
        self._subtask_graph_gen_time.record(
            timer.duration,
            {
                "session_id": self._task.session_id,
                "task_id": self._task.task_id,
                "stage_id": stage_id,
            },
        )

        tile_context = await asyncio.to_thread(
            self._get_stage_tile_context,
            {c for c in chunk_graph.result_chunks if not isinstance(c.op, Fetch)},
        )

        with Timer() as timer:
            chunk_to_result = await self._executor.execute_subtask_graph(
                stage_id, subtask_graph, chunk_graph, tile_context
            )
        stage_profiler.set("run", timer.duration)

        self._preprocessor.post_chunk_graph_execution()
        if self._preprocessor.chunk_optimization_records_list:
            optimization_records = self._preprocessor.chunk_optimization_records_list[
                -1
            ]
        else:
            optimization_records = None
        self._update_stage_meta(chunk_to_result, tile_context, optimization_records)

    def _get_stage_tile_context(self, result_chunks: Set[Chunk]) -> TileContext:
        collected = self._stage_tileables
        tile_context = TileContext()
        for tileable in self.tileable_graph:
            if tileable in collected:
                continue
            tiled_tileable = self._preprocessor.tile_context.get(tileable)
            if tiled_tileable is not None:
                tileable_chunks = [c.data for c in tiled_tileable.chunks]
                if any(c not in result_chunks for c in tileable_chunks):
                    continue
                tile_context[tileable] = tiled_tileable
                collected.add(tileable)
        return tile_context

    @classmethod
    def _update_stage_meta(
        cls,
        chunk_to_result: Dict[Chunk, ExecutionChunkResult],
        tile_context: TileContext,
        optimization_records: OptimizationRecords,
    ):
        for tiled_tileable in tile_context.values():
            cls._update_result_meta(chunk_to_result, tiled_tileable)

        for c, r in chunk_to_result.items():
            c.params = r.meta
            original_chunk = (
                optimization_records and optimization_records.get_original_entity(c)
            )
            if original_chunk is not None:
                original_chunk.params = r.meta

        for tileable, tiled_tileable in tile_context.items():
            tiled_tileable.refresh_params()
            tileable.params = tiled_tileable.params

    @classmethod
    def _update_result_meta(
        cls, chunk_to_result: Dict[Chunk, ExecutionChunkResult], tileable: TileableType
    ):
        chunks = [c.data for c in tileable.chunks]
        if isinstance(tileable, DATAFRAME_TYPE):
            for c in chunks:
                i, j = c.index
                meta = chunk_to_result[c].meta
                shape = meta.get("shape")
                update_shape = shape is None
                shape = shape if not update_shape else [None, None]
                if i > 0:
                    # update dtypes_value
                    c0j = chunk_to_result[tileable.cix[0, j].data].meta
                    meta["dtypes_value"] = c0j["dtypes_value"]
                    if update_shape:
                        shape[1] = c0j["shape"][1]
                if j > 0:
                    # update index_value
                    ci0 = chunk_to_result[tileable.cix[i, 0].data].meta
                    meta["index_value"] = ci0["index_value"]
                    if update_shape:
                        shape[0] = ci0["shape"][0]
                if update_shape:
                    meta["shape"] = tuple(shape)
        elif isinstance(tileable, SERIES_TYPE):
            first_meta = chunk_to_result[chunks[0]].meta
            for c in chunks:
                i = c.index[0]
                meta = chunk_to_result[c].meta
                if i > 0:
                    meta["name"] = first_meta["name"]
                    meta["dtype"] = first_meta["dtype"]
        elif isinstance(tileable, TENSOR_TYPE):
            ndim = tileable.ndim
            for i, c in enumerate(chunks):
                meta = chunk_to_result[c].meta
                if "shape" not in meta:
                    shape = []
                    for i, ind in enumerate(c.index):
                        ind0 = [0] * ndim
                        ind0[i] = ind
                        c0 = tileable.cix[tuple(ind0)].data
                        shape.append(chunk_to_result[c0].meta["shape"][i])
                    meta["shape"] = tuple(shape)
                if i > 0:
                    first = chunk_to_result[chunks[0]].meta
                    meta["dtype"] = first["dtype"]
                    meta["order"] = first["order"]

    async def run(self):
        profiling = ProfilingData[self.task_id, "general"]
        self.result.status = TaskStatus.running
        # optimization
        with Timer() as timer:
            # optimization, run it in executor,
            # since optimization may be a CPU intensive operation
            await asyncio.to_thread(self._preprocessor.optimize)
        profiling.set("optimize", timer.duration)

        self._tileable_id_to_tileable = await asyncio.to_thread(
            self._get_tileable_id_to_tileable, self._preprocessor.tileable_graph
        )

        try:
            async with self._executor:
                async for stage_args in self._iter_stage_chunk_graph():
                    await self._process_stage_chunk_graph(*stage_args)
        except Exception as ex:
            self.result.error = ex
            self.result.traceback = ex.__traceback__
        finally:
            self._gen_result()
            self._finish()

    async def get_progress(self) -> float:
        # get tileable proportion that is tiled
        return await self._executor.get_progress()

    async def cancel(self):
        self._preprocessor.cancel()
        await self._executor.cancel()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        await self._executor.set_subtask_result(subtask_result)

    @staticmethod
    def _get_tileable_id_to_tileable(
        tileable_graph: TileableGraph,
    ) -> Dict[str, TileableType]:
        tileable_id_to_tileable = dict()

        for tileable in tileable_graph:
            tileable_id_to_tileable[str(tileable.key)] = tileable

        return tileable_id_to_tileable

    def _gen_result(self):
        self.result.status = TaskStatus.terminated
        self.result.end_time = time.time()
        cost_time_secs = self.result.end_time - self.result.start_time
        logger.info(
            "Time consuming to execute a task is %ss with session id %s, task id %s",
            cost_time_secs,
            self._task.session_id,
            self._task.task_id,
        )
        self._task_execution_time.record(
            cost_time_secs,
            {"session_id": self._task.session_id, "task_id": self._task.task_id},
        )

    def dump_subtask_graph(self):
        from .graph_visualizer import GraphVisualizer

        try:  # pragma: no cover
            import graphviz
        except ImportError:
            graphviz = None

        dot = GraphVisualizer(self).to_dot()
        directory = tempfile.gettempdir()
        file_name = f"mars-{self.task_id}"
        logger.debug(
            "subtask graph is stored in %s", os.path.join(directory, file_name)
        )
        if graphviz is not None:  # pragma: no cover
            g = graphviz.Source(dot)
            g.view(file_name, directory=directory)
        else:
            with open(os.path.join(directory, file_name), "w") as f:
                f.write(dot)

    def _finish(self):
        self.done.set()
        if self._dump_subtask_graph:
            self.dump_subtask_graph()
        if MARS_ENABLE_PROFILING or (
            self._task.extra_config and self._task.extra_config.get("enable_profiling")
        ):
            ProfilingData[self._task.task_id, "general"].set(
                "total", time.time() - self.result.start_time
            )
            serialization = ProfilingData[self._task.task_id, "serialization"]
            if not serialization.empty():
                serialization.set(
                    "total",
                    sum(serialization.values()),
                )
            data = ProfilingData.pop(self._task.task_id)
            self.result.profiling = {
                "supervisor": data,
            }

    def is_done(self) -> bool:
        return self.done.is_set()
