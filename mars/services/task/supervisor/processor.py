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
import itertools
import logging
import operator
import os
import sys
import tempfile
import time
from collections import defaultdict
from functools import reduce, wraps
from typing import Callable, Coroutine, Dict, Iterator, List, Optional, Set, Type, Union

from .... import oscar as mo
from ....config import Config
from ....core import ChunkGraph, TileableGraph
from ....core.operand import (
    Fetch,
    FetchShuffle,
    MapReduceOperand,
    ShuffleProxy,
    OperandStage,
)
from ....metrics import Metrics
from ....optimization.logical import OptimizationRecords
from ....oscar.profiling import (
    ProfilingData,
    MARS_ENABLE_PROFILING,
)
from ....typing import TileableType, BandType
from ....utils import build_fetch, Timer
from ...cluster.api import ClusterAPI
from ...lifecycle.api import LifecycleAPI
from ...meta.api import MetaAPI
from ...scheduling import SchedulingAPI
from ...subtask import Subtask, SubtaskResult, SubtaskStatus, SubtaskGraph
from ..core import Task, TaskResult, TaskStatus, new_task_id
from .preprocessor import TaskPreprocessor
from .stage import TaskStageProcessor

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
    stage_processors: List[TaskStageProcessor]
    cur_stage_processor: Optional[TaskStageProcessor]

    def __init__(
        self,
        task: Task,
        preprocessor: TaskPreprocessor,
        # APIs
        cluster_api: ClusterAPI,
        lifecycle_api: LifecycleAPI,
        scheduling_api: SchedulingAPI,
        meta_api: MetaAPI,
    ):
        self._task = task
        self._preprocessor = preprocessor

        # APIs
        self._cluster_api = cluster_api
        self._lifecycle_api = lifecycle_api
        self._scheduling_api = scheduling_api
        self._meta_api = meta_api

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
        self.stage_processors = []
        self.cur_stage_processor = None

        self._err_infos = []
        self._chunk_graph_iter = None
        self._raw_tile_context = preprocessor.tile_context.copy()
        self._lifecycle_processed_tileables = set()

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
    def preprocessor(self):
        return self._preprocessor

    @property
    def tileable_graph(self):
        return self._preprocessor.tileable_graph

    @property
    def tile_context(self):
        return self._preprocessor.tile_context

    def get_tiled(self, tileable: TileableType):
        return self._preprocessor.get_tiled(tileable)

    @_record_error
    async def optimize(self) -> TileableGraph:
        # optimization, run it in executor,
        # since optimization may be a CPU intensive operation
        return await asyncio.to_thread(self._preprocessor.optimize)

    @_record_error
    async def incref_fetch_tileables(self):
        # incref fetch tileables in tileable graph to prevent them from deleting
        to_incref_tileable_keys = [
            tileable.op.source_key
            for tileable in self.tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context
        ]
        await self._lifecycle_api.incref_tileables(to_incref_tileable_keys)

    @_record_error
    async def decref_fetch_tileables(self):
        fetch_tileable_keys = [
            tileable.op.source_key
            for tileable in self.tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context
        ]
        await self._lifecycle_api.decref_tileables(fetch_tileable_keys)

    @_record_error
    async def incref_result_tileables(self):
        processed = self._lifecycle_processed_tileables
        # track and incref result tileables if tiled
        tracks = [], []
        for result_tileable in self.tileable_graph.result_tileables:
            if result_tileable in processed:  # pragma: no cover
                continue
            try:
                tiled_tileable = self._preprocessor.get_tiled(result_tileable)
                tracks[0].append(result_tileable.key)
                tracks[1].append(
                    self._lifecycle_api.track.delay(
                        result_tileable.key, [c.key for c in tiled_tileable.chunks]
                    )
                )
                processed.add(result_tileable)
            except KeyError:
                # not tiled, skip
                pass
        if tracks:
            await self._lifecycle_api.track.batch(*tracks[1])
            await self._lifecycle_api.incref_tileables(tracks[0])

    @_record_error
    async def decref_result_tileables(self):
        await self._lifecycle_api.decref_tileables(
            [t.key for t in self._lifecycle_processed_tileables]
        )

    @_record_error
    async def incref_stage(self, stage_processor: "TaskStageProcessor"):
        subtask_graph = stage_processor.subtask_graph
        incref_chunk_keys = []
        for subtask in subtask_graph:
            # for subtask has successors, incref number of successors
            n = subtask_graph.count_successors(subtask)
            for c in subtask.chunk_graph.results:
                incref_chunk_keys.extend([c.key] * n)
            # process reducer, since mapper will generate sub keys
            # we incref (main_key, sub_key) for reducer
            for chunk in subtask.chunk_graph:
                if (
                    isinstance(chunk.op, MapReduceOperand)
                    and chunk.op.stage == OperandStage.reduce
                ):
                    # reducer
                    data_keys = chunk.op.get_dependent_data_keys()
                    incref_chunk_keys.extend(data_keys)
                    # main key incref as well, to ensure existence of meta
                    incref_chunk_keys.extend([key[0] for key in data_keys])
        result_chunks = stage_processor.chunk_graph.result_chunks
        incref_chunk_keys.extend([c.key for c in result_chunks])
        logger.debug("Incref chunks %s for stage", incref_chunk_keys)
        await self._lifecycle_api.incref_chunks(incref_chunk_keys)

    @classmethod
    def _get_decref_stage_chunk_keys(
        cls, stage_processor: "TaskStageProcessor"
    ) -> List[str]:
        decref_chunk_keys = []
        error_or_cancelled = stage_processor.error_or_cancelled()
        if stage_processor.subtask_graph:
            subtask_graph = stage_processor.subtask_graph
            if error_or_cancelled:
                # error or cancel, rollback incref for subtask results
                for subtask in subtask_graph:
                    if subtask.subtask_id in stage_processor.decref_subtask:
                        continue
                    stage_processor.decref_subtask.add(subtask.subtask_id)
                    # if subtask not executed, rollback incref of predecessors
                    for inp_subtask in subtask_graph.predecessors(subtask):
                        for result_chunk in inp_subtask.chunk_graph.results:
                            # for reducer chunk, decref mapper chunks
                            if isinstance(result_chunk.op, ShuffleProxy):
                                for chunk in subtask.chunk_graph:
                                    if isinstance(chunk.op, MapReduceOperand):
                                        data_keys = chunk.op.get_dependent_data_keys()
                                        decref_chunk_keys.extend(data_keys)
                                        decref_chunk_keys.extend(
                                            [key[0] for key in data_keys]
                                        )
                        decref_chunk_keys.extend(
                            [c.key for c in inp_subtask.chunk_graph.results]
                        )
            # decref result of chunk graphs
            decref_chunk_keys.extend(
                [c.key for c in stage_processor.chunk_graph.results]
            )
        return decref_chunk_keys

    @mo.extensible
    @_record_error
    async def decref_stage(self, stage_processor: "TaskStageProcessor"):
        decref_chunk_keys = self._get_decref_stage_chunk_keys(stage_processor)
        logger.debug(
            "Decref chunks %s when stage %s finish",
            decref_chunk_keys,
            stage_processor.stage_id,
        )
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    @decref_stage.batch
    @_record_error
    async def decref_stage(self, args_list, kwargs_list):
        decref_chunk_keys = []
        for args, kwargs in zip(args_list, kwargs_list):
            decref_chunk_keys.extend(self._get_decref_stage_chunk_keys(*args, **kwargs))
        logger.debug("Decref chunks %s when stage finish", decref_chunk_keys)
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    async def _get_next_chunk_graph(
        self, chunk_graph_iter: Iterator[ChunkGraph]
    ) -> Optional[ChunkGraph]:
        def next_chunk_graph():
            try:
                return next(chunk_graph_iter)
            except StopIteration:
                return

        fut = asyncio.to_thread(next_chunk_graph)
        chunk_graph = await fut
        return chunk_graph

    async def _get_available_band_slots(self) -> Dict[BandType, int]:
        async for bands in self._cluster_api.watch_all_bands():
            if bands:
                return bands

    def _init_chunk_graph_iter(self, tileable_graph: TileableGraph):
        if self._chunk_graph_iter is None:
            self._chunk_graph_iter = iter(self._preprocessor.tile(tileable_graph))

    def _get_chunk_optimization_records(self) -> OptimizationRecords:
        if self._preprocessor.chunk_optimization_records_list:
            return self._preprocessor.chunk_optimization_records_list[-1]

    def _get_tileable_id_to_tileable(self) -> Dict[str, TileableType]:
        tileable_id_to_tileable = dict()
        tileable_graph = self._preprocessor.tileable_graph

        for tileable in tileable_graph:
            tileable_id_to_tileable[str(tileable.key)] = tileable

        return tileable_id_to_tileable

    def _get_tileable_to_subtasks(
        self, subtask_graph: SubtaskGraph
    ) -> Dict[TileableType, List[Subtask]]:
        tileable_to_chunks = defaultdict(set)
        chunk_to_subtasks = dict()

        tileable_graph = self._preprocessor.tileable_graph
        tile_ctx = self._preprocessor.tile_context

        for tileable in tileable_graph:
            if tileable not in tile_ctx:
                continue
            for chunk in tile_ctx[tileable].chunks:
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

    @_record_error
    async def get_next_stage_processor(self) -> Optional[TaskStageProcessor]:
        tileable_graph = self._preprocessor.tileable_graph
        self._init_chunk_graph_iter(tileable_graph)

        with Timer() as timer:
            chunk_graph = await self._get_next_chunk_graph(self._chunk_graph_iter)
            if chunk_graph is None:
                # tile finished
                self._preprocessor.done = True
                return
        logger.info(
            "Time consuming to gen a chunk graph is %ss with session id %s, task id %s",
            timer.duration,
            self._task.session_id,
            self._task.task_id,
        )
        self._chunk_graph_gen_time.record(
            timer.duration,
            {"session_id": self._task.session_id, "task_id": self._task.task_id},
        )
        stage_id = new_task_id()
        stage_profiling = ProfilingData[self._task.task_id, "general"].nest(
            f"stage_{stage_id}"
        )
        stage_profiling.set(f"tile({len(chunk_graph)})", timer.duration)

        # gen subtask graph
        available_bands = await self._get_available_band_slots()

        with Timer() as timer:
            subtask_graph = await asyncio.to_thread(
                self._preprocessor.analyze,
                chunk_graph,
                available_bands,
                stage_id=stage_id,
            )
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
        stage_profiling.set(f"gen_subtask_graph({len(subtask_graph)})", timer.duration)

        tileable_to_subtasks = await asyncio.to_thread(
            self._get_tileable_to_subtasks, subtask_graph
        )
        tileable_id_to_tileable = await asyncio.to_thread(
            self._get_tileable_id_to_tileable
        )
        stage_processor = TaskStageProcessor(
            stage_id,
            self._task,
            chunk_graph,
            subtask_graph,
            list(available_bands),
            tileable_to_subtasks,
            tileable_id_to_tileable,
            self._get_chunk_optimization_records(),
            self._scheduling_api,
            self._meta_api,
        )
        return stage_processor

    @_record_error
    async def schedule(self, stage_processor: TaskStageProcessor):
        await stage_processor.run()

    def gen_result(self):
        self.result.status = TaskStatus.terminated
        self.result.end_time = time.time()
        for stage_processor in self.stage_processors:
            if stage_processor.result.error is not None:
                err = stage_processor.result.error
                tb = stage_processor.result.traceback
                self._err_infos.append((type(err), err, tb))
        if self._err_infos:
            # grab the last error
            _, err, tb = self._err_infos[-1]
            self.result.error = err
            self.result.traceback = tb
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

    def finish(self):
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
    _processed_task_ids: Set[str]
    _cur_processor: Optional[TaskProcessor]

    def __init__(self, session_id: str, task_id: str, task_name: str = None):
        self.session_id = session_id
        self.task_id = task_id
        self.task_name = task_name

        self._task_id_to_processor = dict()
        self._cur_processor = None
        self._subtask_decref_events = dict()

        self._cluster_api = None
        self._meta_api = None
        self._lifecycle_api = None
        self._scheduling_api = None

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)
        self._scheduling_api = await SchedulingAPI.create(self.session_id, self.address)
        self._meta_api = await MetaAPI.create(self.session_id, self.address)
        self._lifecycle_api = await LifecycleAPI.create(self.session_id, self.address)

    @classmethod
    def gen_uid(cls, session_id: str, task_id: str):
        return f"task_processor_{session_id}_{task_id}"

    async def add_task(
        self,
        task: Task,
        tiled_context: Dict[TileableType, TileableType],
        config: Config,
        task_preprocessor_cls: Type[TaskPreprocessor],
    ):
        task_preprocessor = task_preprocessor_cls(
            task, tiled_context=tiled_context, config=config
        )
        processor = TaskProcessor(
            task,
            task_preprocessor,
            self._cluster_api,
            self._lifecycle_api,
            self._scheduling_api,
            self._meta_api,
        )
        self._task_id_to_processor[task.task_id] = processor

        # tell self to start running
        await self.ref().start.tell()

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
        processor.result.status = TaskStatus.running

        incref_fetch, incref_result = False, False
        profiling = ProfilingData[processor.task_id, "general"]
        try:
            # optimization
            with Timer() as timer:
                yield processor.optimize()
            profiling.set("optimize", timer.duration)
            # incref fetch tileables to ensure fetch data not deleted
            with Timer() as timer:
                yield processor.incref_fetch_tileables()
            profiling.set("incref_fetch_tileables", timer.duration)
            incref_fetch = True
            while True:
                with Timer() as stage_timer:
                    stage_processor = yield processor.get_next_stage_processor()
                    if stage_processor is None:
                        break
                    stage_profiling = profiling.nest(
                        f"stage_{stage_processor.stage_id}"
                    )
                    # track and incref result tileables
                    yield processor.incref_result_tileables()
                    incref_result = True
                    # incref stage
                    yield processor.incref_stage(stage_processor)
                    # schedule stage
                    processor.stage_processors.append(stage_processor)
                    processor.cur_stage_processor = stage_processor
                    logger.info("Start new stage with id %s.", stage_processor.stage_id)
                    with Timer() as timer:
                        yield processor.schedule(stage_processor)
                    stage_profiling.set("run", timer.duration)
                stage_profiling.set("total", stage_timer.duration)
                if stage_processor.error_or_cancelled():
                    break
        finally:
            processor.gen_result()
            processor.cur_stage_processor = None
            try:
                # clean ups
                decrefs = []
                error_or_cancelled = False
                for stage_processor in processor.stage_processors:
                    if stage_processor.error_or_cancelled():
                        error_or_cancelled = True
                    decrefs.append(processor.decref_stage.delay(stage_processor))
                yield processor.decref_stage.batch(*decrefs)
                if incref_fetch:
                    # revert fetch incref
                    yield processor.decref_fetch_tileables()
                if incref_result and error_or_cancelled:
                    # revert result incref if error or cancelled
                    yield processor.decref_result_tileables()
            finally:
                processor.finish()
                self._cur_processor = None

    async def wait(self, timeout: int = None):
        fs = [
            processor.done.wait() for processor in self._task_id_to_processor.values()
        ]

        _, pending = yield asyncio.wait(fs, timeout=timeout)
        if not pending:
            raise mo.Return(self.result())
        else:
            [fut.cancel() for fut in pending]

    async def cancel(self):
        if self._cur_processor:
            if not self._cur_processor.cur_stage_processor:
                # still in preprocess
                self._cur_processor.preprocessor.cancel()
            else:
                # otherwise, already in stages, cancel current running stage
                await self._cur_processor.cur_stage_processor.cancel()

    def result(self):
        terminated_result = None
        for processor in self._task_id_to_processor.values():
            if processor.result.status != TaskStatus.terminated:
                return processor.result
            else:
                terminated_result = processor.result
        return terminated_result

    def progress(self):
        tiled_percentage = 0.0
        i = 0
        for processor in self._task_id_to_processor.values():
            # get tileable proportion that is tiled
            tileable_graph = processor.tileable_graph
            tileable_context = processor.tile_context
            tiled_percentage += len(tileable_context) / len(tileable_graph)
            i += 1
        tiled_percentage /= i

        # get progress of stages
        subtask_progress = 0.0
        n_stage = 0
        stage_processors = itertools.chain(
            *(
                processor.stage_processors
                for processor in self._task_id_to_processor.values()
            )
        )
        for stage_processor in stage_processors:
            if stage_processor.subtask_graph is None:  # pragma: no cover
                # generating subtask
                continue
            n_subtask = len(stage_processor.subtask_graph)
            if n_subtask == 0:  # pragma: no cover
                continue
            progress = sum(
                result.progress for result in stage_processor.subtask_results.values()
            )
            progress += sum(
                result.progress
                for subtask_key, result in stage_processor.subtask_snapshots.items()
                if subtask_key not in stage_processor.subtask_results
            )
            subtask_progress += progress / n_subtask
            n_stage += 1
        if n_stage > 0:
            subtask_progress /= n_stage

        return subtask_progress * tiled_percentage

    def get_result_tileables(self):
        processor = list(self._task_id_to_processor.values())[-1]
        tileable_graph = processor.tileable_graph
        result = []
        for result_tileable in tileable_graph.result_tileables:
            tiled = processor.get_tiled(result_tileable)
            result.append(build_fetch(tiled))
        return result

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
            for stage in processor.stage_processors:
                tileable_to_subtasks.update(stage.tileable_to_subtasks)

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
            for stage in processor.stage_processors:
                if tileable_id in stage.tileable_id_to_tileable:
                    tileable = stage.tileable_id_to_tileable[tileable_id]
                    returned_subtasks = {
                        subtask.subtask_id: subtask
                        for subtask in stage.tileable_to_subtasks[tileable]
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

    async def _decref_input_subtasks(
        self, subtask: Subtask, subtask_graph: SubtaskGraph
    ):
        # make sure subtasks are decreffed only once
        if subtask.subtask_id not in self._subtask_decref_events:
            self._subtask_decref_events[subtask.subtask_id] = asyncio.Event()
        else:  # pragma: no cover
            await self._subtask_decref_events[subtask.subtask_id].wait()
            return

        decref_chunk_keys = []
        for in_subtask in subtask_graph.iter_predecessors(subtask):
            for result_chunk in in_subtask.chunk_graph.results:
                # for reducer chunk, decref mapper chunks
                if isinstance(result_chunk.op, ShuffleProxy):
                    for chunk in subtask.chunk_graph:
                        if (
                            isinstance(chunk.op, MapReduceOperand)
                            and chunk.op.stage == OperandStage.reduce
                        ):
                            data_keys = chunk.op.get_dependent_data_keys()
                            decref_chunk_keys.extend(data_keys)
                            # decref main key as well
                            decref_chunk_keys.extend([key[0] for key in data_keys])
                decref_chunk_keys.append(result_chunk.key)
        logger.debug(
            "Decref chunks %s when subtask %s finish",
            decref_chunk_keys,
            subtask.subtask_id,
        )
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

        # `set_subtask_result` will be called when subtask finished
        # but report progress will call set_subtask_result too,
        # so it have risk to duplicate decrease some subtask input object reference,
        # it will cause object reference count lower zero
        # TODO(Catch-Bull): Pop asyncio.Event when current subtask `set_subtask_result`
        # will never be called
        self._subtask_decref_events[subtask.subtask_id].set()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        logger.debug(
            "Set subtask %s with result %s.", subtask_result.subtask_id, subtask_result
        )
        if (
            self._cur_processor is None
            or self._cur_processor.cur_stage_processor is None
            or (
                subtask_result.stage_id
                and self._cur_processor.cur_stage_processor.stage_id
                != subtask_result.stage_id
            )
        ):
            logger.warning(
                "Stage %s for subtask %s not exists, got stale subtask result %s which may be "
                "speculative execution from previous stages, just ignore it.",
                subtask_result.stage_id,
                subtask_result.subtask_id,
                subtask_result,
            )
            return
        stage_processor = self._cur_processor.cur_stage_processor
        subtask = stage_processor.subtask_id_to_subtask[subtask_result.subtask_id]

        prev_result = stage_processor.subtask_results.get(subtask)
        if prev_result and (
            prev_result.status == SubtaskStatus.succeeded
            or prev_result.progress > subtask_result.progress
        ):
            logger.info(
                "Skip set subtask %s with result %s, previous result is %s.",
                subtask.subtask_id,
                subtask_result,
                prev_result,
            )
            # For duplicate run of subtasks, if the progress is smaller or the subtask has finished or canceled
            # in task speculation, just do nothing.
            # TODO(chaokunyang) If duplicate run of subtasks failed, it may be the fault in worker node,
            #  print the exception, and if multiple failures on the same node, remove the node from the cluster.
            return
        if subtask_result.bands:
            [band] = subtask_result.bands
        else:
            band = None
        stage_processor.subtask_snapshots[subtask] = subtask_result.update(
            stage_processor.subtask_snapshots.get(subtask)
        )
        if subtask_result.status.is_done:
            # update stage_processor.subtask_results to avoid concurrent set_subtask_result
            # since we release lock when `_decref_input_subtasks`.
            stage_processor.subtask_results[subtask] = subtask_result.update(
                stage_processor.subtask_results.get(subtask)
            )
            try:
                # Since every worker will call supervisor to set subtask result,
                # we need to release actor lock to make `decref_chunks` parallel to avoid blocking
                # other `set_subtask_result` calls.
                # If speculative execution enabled, concurrent subtasks may got error since input chunks may
                # got deleted. But it's OK because the current subtask run has succeed.
                if subtask.subtask_id not in stage_processor.decref_subtask:
                    stage_processor.decref_subtask.add(subtask.subtask_id)
                    yield self._decref_input_subtasks(
                        subtask, stage_processor.subtask_graph
                    )

            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                logger.debug(
                    "Decref input subtasks for subtask %s failed.", subtask.subtask_id
                )
                _, err, tb = sys.exc_info()
                if subtask_result.status not in (
                    SubtaskStatus.errored,
                    SubtaskStatus.cancelled,
                ):
                    subtask_result.status = SubtaskStatus.errored
                    subtask_result.error = err
                    subtask_result.traceback = tb
            await stage_processor.set_subtask_result(subtask_result, band=band)

    def is_done(self) -> bool:
        for processor in self._task_id_to_processor.values():
            if not processor.is_done():
                return False
        return True
