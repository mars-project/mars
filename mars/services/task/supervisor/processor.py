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
import importlib
import logging
import operator
import os
import sys
import tempfile
import time
from collections import defaultdict
from functools import wraps, reduce
from typing import Callable, Coroutine, Dict, Iterator, Optional, Type, Union, List

from .... import oscar as mo
from ....config import Config
from ....core import ChunkGraph, TileableGraph
from ....core.operand import Fetch, FetchShuffle
from ....execution.api import TaskExecutor
from ....metrics import Metrics
from ....oscar.profiling import (
    ProfilingData,
    MARS_ENABLE_PROFILING,
)
from ....typing import TileableType
from ....utils import build_fetch, Timer
from ...subtask import SubtaskResult, SubtaskStatus, SubtaskGraph, Subtask
from ..core import Task, TaskResult, TaskStatus, new_task_id
from .preprocessor import TaskPreprocessor

logger = logging.getLogger(__name__)

MARS_ENABLE_DUMPING_SUBTASK_GRAPH = int(os.environ.get("MARS_DUMP_SUBTASK_GRAPH", 0))


def _record_error(func: Union[Callable, Coroutine] = None, log_when_error=True):
    assert asyncio.iscoroutinefunction(func)

    @wraps(func)
    async def inner(processor: "TaskProcessor", *args, **kwargs):
        try:
            return await func(processor, *args, **kwargs)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
            if log_when_error:
                logger.exception("Unexpected error happens in %s", func)
            processor._err_infos.append(sys.exc_info())
            raise

    return inner


class TaskProcessor:
    _tileable_to_subtasks: Dict[TileableType, List[Subtask]]
    _tileable_id_to_tileable: Dict[str, TileableType]

    def __init__(
        self,
        task: Task,
        preprocessor: TaskPreprocessor,
        executor: TaskExecutor,
    ):
        self._task = task
        self._preprocessor = preprocessor
        self._executor = executor

        self._tileable_to_subtasks = dict()
        self._tileable_id_to_tileable = dict()

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
    def tileable_to_subtasks(self):
        return self._tileable_to_subtasks

    @property
    def tileable_id_to_tileable(self):
        return self._tileable_id_to_tileable

    @property
    def stage_processors(self):
        # TODO(fyrestone): Remove it.
        return self._executor.get_stage_processors()

    def get_tiled(self, tileable: TileableType):
        return self._preprocessor.get_tiled(tileable)

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

    async def _process_stage_chunk_graph(self, stage_id, stage_profiler, chunk_graph):
        available_bands = await self._executor.get_available_band_slots()
        with Timer() as timer:
            subtask_graph = await asyncio.to_thread(
                self._preprocessor.analyze,
                chunk_graph,
                available_bands,
                stage_id=stage_id,
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

        tileable_to_subtasks = await asyncio.to_thread(
            self._get_tileable_to_subtasks,
            self._preprocessor.tileable_graph,
            self._preprocessor.tile_context,
            subtask_graph,
        )
        self._tileable_to_subtasks.update(tileable_to_subtasks)

        with Timer() as timer:
            execution_chunk_results = await self._executor.execute_subtask_graph(
                stage_id, subtask_graph, chunk_graph
            )
        stage_profiler.set("run", timer.duration)

        if self._preprocessor.chunk_optimization_records_list:
            optimization_records = self._preprocessor.chunk_optimization_records_list[
                -1
            ]
        else:
            optimization_records = None
        self._update_chunks_meta(
            chunk_graph.result_chunks,
            execution_chunk_results,
            optimization_records,
        )

    @staticmethod
    def _update_chunks_meta(
        chunk_graph_results, execution_chunk_results, optimization_records
    ):
        for chunk, execution_chunk_result in zip(
            chunk_graph_results, execution_chunk_results
        ):
            chunk.params = execution_chunk_result.meta
            original_chunk = (
                optimization_records and optimization_records.get_original_entity(chunk)
            )
            if original_chunk is not None:
                original_chunk.params = chunk.params

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

    async def get_progress(self):
        # get tileable proportion that is tiled
        tileable_graph = self._preprocessor.tileable_graph
        tileable_context = self._preprocessor.tile_context
        tiled_percentage = len(tileable_context) / len(tileable_graph)
        return tiled_percentage * await self._executor.get_progress()

    async def cancel(self):
        self._preprocessor.cancel()
        await self._executor.cancel()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        await self._executor.set_subtask_result(subtask_result)

    @staticmethod
    def _get_tileable_to_subtasks(
        tileable_graph: TileableGraph,
        tile_context: Dict[TileableType, TileableType],
        subtask_graph: SubtaskGraph,
    ) -> Dict[TileableType, List[Subtask]]:
        tileable_to_chunks = defaultdict(set)
        chunk_to_subtasks = dict()

        for tileable in tileable_graph:
            if tileable not in tile_context:
                continue
            for chunk in tile_context[tileable].chunks:
                tileable_to_chunks[tileable].add(chunk.key)
                # register chunk mapping for tiled terminals
                chunk_to_subtasks[chunk.key] = set()

        for subtask in subtask_graph:
            for chunk in subtask.chunk_graph:
                # for every non-fuse chunks (including fused),
                # register subtasks if needed
                if (
                    isinstance(chunk.op, (FetchShuffle, Fetch))
                    or chunk.key not in chunk_to_subtasks
                ):
                    continue
                chunk_to_subtasks[chunk.key].add(subtask)

        tileable_to_subtasks = dict()
        # collect subtasks for tileables
        for tileable, chunk_keys in tileable_to_chunks.items():
            tileable_to_subtasks[tileable] = list(
                reduce(
                    operator.or_,
                    [chunk_to_subtasks[chunk_key] for chunk_key in chunk_keys],
                )
            )
        return tileable_to_subtasks

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


class TaskProcessorActor(mo.Actor):
    _task_id_to_processor: Dict[str, TaskProcessor]
    _cur_processor: Optional[TaskProcessor]

    def __init__(
        self,
        session_id: str,
        task_id: str,
        task_name: str = None,
        task_processor_cls: Type[TaskPreprocessor] = None,
    ):
        self.session_id = session_id
        self.task_id = task_id
        self.task_name = task_name

        self._task_processor_cls = self._get_task_processor_cls(task_processor_cls)
        self._task_id_to_processor = dict()
        self._cur_processor = None

    @classmethod
    def gen_uid(cls, session_id: str, task_id: str):
        return f"task_processor_{session_id}_{task_id}"

    async def add_task(
        self,
        task: Task,
        tiled_context: Dict[TileableType, TileableType],
        config: Config,
        task_executor_config: Dict,
        task_preprocessor_cls: Type[TaskPreprocessor],
    ):
        task_preprocessor = task_preprocessor_cls(
            task, tiled_context=tiled_context, config=config
        )
        task_executor = await TaskExecutor.create(
            task_executor_config,
            task=task,
            session_id=self.session_id,
            address=self.address,
            tileable_graph=task_preprocessor.tileable_graph,
            tile_context=task_preprocessor.tile_context,
        )
        processor = self._task_processor_cls(
            task,
            task_preprocessor,
            task_executor,
        )
        self._task_id_to_processor[task.task_id] = processor

        # tell self to start running
        await self.ref().start.tell()

    @classmethod
    def _get_task_processor_cls(cls, task_processor_cls):
        if task_processor_cls is not None:
            assert isinstance(task_processor_cls, str)
            module, name = task_processor_cls.rsplit(".", 1)
            return getattr(importlib.import_module(module), name)
        else:
            return TaskProcessor

    def _get_unprocessed_task_processor(self):
        for processor in self._task_id_to_processor.values():
            if processor.result.status == TaskStatus.pending:
                return processor

    async def start(self):
        if self._cur_processor is not None:  # pragma: no cover
            # some processor is running
            return

        processor = self._get_unprocessed_task_processor()
        if processor is None:  # pragma: no cover
            return
        self._cur_processor = processor
        try:
            yield processor.run()
        finally:
            self._cur_processor = None

    async def wait(self, timeout: int = None):
        fs = [
            asyncio.ensure_future(processor.done.wait())
            for processor in self._task_id_to_processor.values()
        ]

        _, pending = yield asyncio.wait(fs, timeout=timeout)
        if not pending:
            raise mo.Return(self.result())
        else:
            [fut.cancel() for fut in pending]

    async def cancel(self):
        if self._cur_processor:
            await self._cur_processor.cancel()

    def result(self):
        terminated_result = None
        for processor in self._task_id_to_processor.values():
            if processor.result.status != TaskStatus.terminated:
                return processor.result
            else:
                terminated_result = processor.result
        return terminated_result

    async def progress(self):
        processor_progresses = [
            await processor.get_progress()
            for processor in self._task_id_to_processor.values()
        ]
        return sum(processor_progresses) / len(processor_progresses)

    def get_result_tileables(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        result = []
        for result_tileable in tileable_graph.result_tileables:
            tiled = processor.get_tiled(result_tileable)
            result.append(build_fetch(tiled))
        return result

    def get_subtask_graphs(self, task_id: str) -> List[SubtaskGraph]:
        return [
            stage_processor.subtask_graph
            for stage_processor in self._task_id_to_processor[task_id].stage_processors
        ]

    def get_tileable_graph_as_dict(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph

        node_list = []
        edge_list = []

        visited = set()

        for chunk in tileable_graph:
            if chunk.key in visited:
                continue
            visited.add(chunk.key)

            node_name = str(chunk.op)

            node_list.append({"tileableId": chunk.key, "tileableName": node_name})
            for inp, is_pure_dep in zip(chunk.inputs, chunk.op.pure_depends):
                if inp not in tileable_graph:  # pragma: no cover
                    continue
                edge_list.append(
                    {
                        "fromTileableId": inp.key,
                        "toTileableId": chunk.key,
                        "linkType": 1 if is_pure_dep else 0,
                    }
                )

        graph_dict = {"tileables": node_list, "dependencies": edge_list}
        return graph_dict

    def get_tileable_details(self):
        tileable_to_subtasks = dict()
        subtask_results = dict()

        for processor in self._task_id_to_processor.values():
            tileable_to_subtasks.update(processor.tileable_to_subtasks)
            for stage in processor.stage_processors:
                for subtask, result in stage.subtask_results.items():
                    subtask_results[subtask.subtask_id] = result
                for subtask, result in stage.subtask_snapshots.items():
                    if subtask.subtask_id in subtask_results:
                        continue
                    subtask_results[subtask.subtask_id] = result

        tileable_infos = dict()
        for tileable, subtasks in tileable_to_subtasks.items():
            results = [
                subtask_results.get(
                    subtask.subtask_id,
                    SubtaskResult(
                        progress=0.0,
                        status=SubtaskStatus.pending,
                        stage_id=subtask.stage_id,
                    ),
                )
                for subtask in subtasks
            ]

            # calc progress
            if not results:  # pragma: no cover
                progress = 1.0
            else:
                progress = (
                    1.0 * sum(result.progress for result in results) / len(results)
                )

            # calc status
            statuses = set(result.status for result in results)
            if not results or statuses == {SubtaskStatus.succeeded}:
                status = SubtaskStatus.succeeded
            elif statuses == {SubtaskStatus.cancelled}:
                status = SubtaskStatus.cancelled
            elif statuses == {SubtaskStatus.pending}:
                status = SubtaskStatus.pending
            elif SubtaskStatus.errored in statuses:
                status = SubtaskStatus.errored
            else:
                status = SubtaskStatus.running

            fields = tileable.op._FIELDS
            field_values = tileable.op._FIELD_VALUES
            props = {
                fields[attr_name].tag: value
                for attr_name, value in field_values.items()
                if attr_name not in ("_key", "_id")
                and isinstance(value, (int, float, str))
            }

            tileable_infos[tileable.key] = {
                "progress": progress,
                "subtaskCount": len(results),
                "status": status.value,
                "properties": props,
            }

        return tileable_infos

    def get_tileable_subtasks(self, tileable_id: str, with_input_output: bool):
        returned_subtasks = dict()
        subtask_id_to_types = dict()

        subtask_details = dict()
        subtask_graph = subtask_results = subtask_snapshots = None
        for processor in self._task_id_to_processor.values():
            tileable_to_subtasks = processor.tileable_to_subtasks
            tileable_id_to_tileable = processor.tileable_id_to_tileable
            for stage in processor.stage_processors:
                if tileable_id in tileable_id_to_tileable:
                    tileable = tileable_id_to_tileable[tileable_id]
                    returned_subtasks = {
                        subtask.subtask_id: subtask
                        for subtask in tileable_to_subtasks[tileable]
                    }
                    subtask_graph = stage.subtask_graph
                    subtask_results = stage.subtask_results
                    subtask_snapshots = stage.subtask_snapshots
                    break
            if returned_subtasks:
                break

        if subtask_graph is None:  # pragma: no cover
            return {}

        if with_input_output:
            for subtask in list(returned_subtasks.values()):
                for pred in subtask_graph.iter_predecessors(subtask):
                    if pred.subtask_id in returned_subtasks:  # pragma: no cover
                        continue
                    returned_subtasks[pred.subtask_id] = pred
                    subtask_id_to_types[pred.subtask_id] = "Input"
                for succ in subtask_graph.iter_successors(subtask):
                    if succ.subtask_id in returned_subtasks:  # pragma: no cover
                        continue
                    returned_subtasks[succ.subtask_id] = succ
                    subtask_id_to_types[succ.subtask_id] = "Output"

        for subtask in returned_subtasks.values():
            subtask_result = subtask_results.get(
                subtask,
                subtask_snapshots.get(
                    subtask,
                    SubtaskResult(
                        progress=0.0,
                        status=SubtaskStatus.pending,
                        stage_id=subtask.stage_id,
                    ),
                ),
            )
            subtask_details[subtask.subtask_id] = {
                "name": subtask.subtask_name,
                "status": subtask_result.status.value,
                "progress": subtask_result.progress,
                "nodeType": subtask_id_to_types.get(subtask.subtask_id, "Calculation"),
            }

        for subtask in returned_subtasks.values():
            pred_ids = []
            for pred in subtask_graph.iter_predecessors(subtask):
                if pred.subtask_id in returned_subtasks:
                    pred_ids.append(pred.subtask_id)
            subtask_details[subtask.subtask_id]["fromSubtaskIds"] = pred_ids
        return subtask_details

    def get_result_tileable(self, tileable_key: str):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        for result_tileable in tileable_graph.result_tileables:
            if result_tileable.key == tileable_key:
                tiled = processor.get_tiled(result_tileable)
                return build_fetch(tiled)
        raise KeyError(f"Tileable {tileable_key} does not exist")  # pragma: no cover

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        logger.debug(
            "Set subtask %s with result %s.", subtask_result.subtask_id, subtask_result
        )
        if self._cur_processor is not None:
            await self._cur_processor.set_subtask_result(subtask_result)

    def is_done(self) -> bool:
        for processor in self._task_id_to_processor.values():
            if not processor.is_done():
                return False
        return True
