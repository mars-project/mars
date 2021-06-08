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

import asyncio
import importlib
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Type, Optional

from .... import oscar as mo
from ....config import Config
from ....core import TileableGraph, ChunkGraph, ChunkGraphBuilder, \
    TileableType, ChunkType, enter_mode
from ....core.context import set_context
from ....core.operand import Fetch, Fuse
from ....optimization.logical.core import OptimizationRecords
from ....optimization.logical.chunk import optimize as optimize_chunk_graph
from ....optimization.logical.tileable import optimize as optimize_tileable_graph
from ....utils import build_fetch, get_params_fields
from ...cluster.api import ClusterAPI
from ...context import ThreadedServiceContext
from ...core import BandType
from ...lifecycle.api import LifecycleAPI
from ...meta import MetaAPI
from ...scheduling import SchedulingAPI
from ...subtask import Subtask, SubtaskResult, SubtaskStatus, SubtaskGraph
from ..analyzer import GraphAnalyzer
from ..config import task_options
from ..core import Task, TaskResult, TaskStatus, new_task_id
from ..errors import TaskNotExist


class TaskConfigurationActor(mo.Actor):
    def __init__(self,
                 task_conf: Dict[str, Any],
                 task_processor_cls: Type["TaskProcessor"] = None):
        for name, value in task_conf.items():
            setattr(task_options, name, value)
        self._task_processor_cls = task_processor_cls

    def get_config(self):
        return {
            'task_options': task_options,
            'task_processor_cls': self._task_processor_cls
        }


class TaskProcessor:
    __slots__ = '_task', 'tileable_graph', 'tile_context', \
                '_config', 'tileable_optimization_records', \
                'chunk_optimization_records_list', '_done'

    tile_context: Dict[TileableType, TileableType]

    def __init__(self,
                 task: Task,
                 tiled_context: Dict[TileableType, TileableType] = None,
                 config: Config = None):
        self._task = task
        self.tileable_graph = task.tileable_graph
        self._config = config

        self.tile_context = tiled_context
        self.tileable_optimization_records = None
        self.chunk_optimization_records_list = []
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
            self.tileable_optimization_records = \
                optimize_tileable_graph(self.tileable_graph)
        return self.tileable_graph

    def _fill_fetch_tileable_with_chunks(self, tileable_graph: TileableGraph):
        for t in tileable_graph:
            if isinstance(t.op, Fetch) and t in self.tile_context:
                tiled = self.tile_context[t]
                t._chunks = tiled.chunks
                t._nsplits = tiled.nsplits

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
            tileable_graph, fuse_enabled=self._task.fuse_enabled,
            tile_context=self.tile_context)
        optimize = self._config.optimize_chunk_graph
        meta_updated = set()
        for chunk_graph in chunk_graph_builder.build():
            # optimize chunk graph
            if optimize:
                self.chunk_optimization_records_list.append(
                    optimize_chunk_graph(chunk_graph))
            yield chunk_graph
            # update tileables' meta
            self._update_tileables_params(tileable_graph, meta_updated)

    def analyze(self,
                chunk_graph: ChunkGraph,
                available_bands: Dict[BandType, int],
                task_stage_info: "TaskStageInfo") -> SubtaskGraph:
        task = self._task
        analyzer = GraphAnalyzer(chunk_graph, available_bands,
                                 task.fuse_enabled, task.extra_config,
                                 task_stage_info)
        return analyzer.gen_subtask_graph()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @done.setter
    def done(self, is_done: bool):
        if is_done:
            self._done.set()
        else:  # pragma: no cover
            self._done.clear()

    def get_tiled(self, tileable: TileableType):
        tileable = tileable.data if hasattr(tileable, 'data') else tileable
        return self.tile_context[tileable]

    def _update_tileable_params(self,  # pylint: disable=no-self-use
                                tileable: TileableType,
                                tiled: TileableType):
        tiled.refresh_params()
        tileable.params = tiled.params

    def _update_tileables_params(self,
                                 tileable_graph: TileableGraph,
                                 updated: Set[TileableType]):
        for tileable in tileable_graph:
            if tileable in updated:
                continue
            tiled_tileable = self.tile_context.get(tileable)
            if tiled_tileable is not None:
                self._update_tileable_params(tileable, tiled_tileable)
                updated.add(tileable)

    def __await__(self):
        return self._done.wait().__await__()


class BandQueue:
    def __init__(self):
        self._queue = deque()
        self._has_data = asyncio.Event()

    def put(self, subtask: Optional[Subtask]):
        self._queue.appendleft(subtask)
        self._has_data.set()

    def get(self) -> Subtask:
        subtask = self._queue.popleft()
        if len(self._queue) == 0:
            self._has_data.clear()
        return subtask

    def __await__(self):
        return self._has_data.wait().__await__()


class SubtaskGraphScheduler:
    def __init__(self,
                 subtask_graph: SubtaskGraph,
                 bands: List[BandType],
                 task_stage_info: "TaskStageInfo",
                 meta_api: MetaAPI,
                 optimization_records: OptimizationRecords,
                 scheduling_api=None):
        self._subtask_graph = subtask_graph
        self._bands = bands
        self._task_stage_info = task_stage_info
        self._meta_api = meta_api
        self._optimization_records = optimization_records
        self._scheduling_api = scheduling_api

        # gen subtask_id to subtask
        self.subtask_id_to_subtask = {subtask.subtask_id: subtask
                                      for subtask in subtask_graph}

        self._subtask_to_bands: Dict[Subtask, BandType] = dict()
        self._subtask_to_results: Dict[Subtask, SubtaskResult] = dict()

        self._band_manager: Dict[BandType, mo.ActorRef] = dict()

        self._done = asyncio.Event()
        self._cancelled = asyncio.Event()

        self._submitted_subtask_ids = set()

    def is_cancelled(self):
        return self._cancelled.is_set()

    async def _schedule_subtasks(self, subtasks: List[Subtask]):
        if not subtasks:
            return
        self._submitted_subtask_ids.update(subtask.subtask_id for subtask in subtasks)
        return await self._scheduling_api.add_subtasks(
            subtasks, [(subtask.priority,) for subtask in subtasks])

    async def _update_chunks_meta(self, chunk_graph: ChunkGraph):
        get_meta = []
        chunks = chunk_graph.result_chunks
        for chunk in chunks:
            if isinstance(chunk.op, Fuse):
                chunk = chunk.chunk
            fields = get_params_fields(chunk)
            get_meta.append(self._meta_api.get_chunk_meta.delay(
                chunk.key, fields=fields))
        metas = await self._meta_api.get_chunk_meta.batch(*get_meta)
        for chunk, meta in zip(chunks, metas):
            chunk.params = meta
            original_chunk = \
                self._optimization_records.get_original_chunk(chunk)
            if original_chunk is not None:
                original_chunk.params = chunk.params

    async def set_subtask_result(self, result: SubtaskResult):
        subtask_id = result.subtask_id
        subtask = self.subtask_id_to_subtask[subtask_id]
        self._subtask_to_results[subtask] = result
        self._submitted_subtask_ids.difference_update([result.subtask_id])

        all_done = len(self._subtask_to_results) == len(self._subtask_graph)
        error_or_cancelled = result.status in (SubtaskStatus.errored, SubtaskStatus.cancelled)

        if all_done or error_or_cancelled:
            if all_done and not error_or_cancelled:
                # subtask graph finished, update result chunks' meta
                await self._update_chunks_meta(
                    self._task_stage_info.chunk_graph)

            await self._scheduling_api.finish_subtasks([result.subtask_id],
                                                       schedule_next=not error_or_cancelled)
            if self._task_stage_info.task_result.status != TaskStatus.terminated:
                self.set_task_info(result.error, result.traceback)
                if not all_done and error_or_cancelled:
                    await self._scheduling_api.cancel_subtasks(list(self._submitted_subtask_ids))
                self._schedule_done()
            return

        # push success subtasks to queue if they are ready
        to_schedule_subtasks = []
        for succ_subtask in self._subtask_graph.successors(subtask):
            if succ_subtask in self._subtask_to_results:
                continue
            pred_subtasks = self._subtask_graph.predecessors(succ_subtask)
            if all(pred_subtask in self._subtask_to_results
                   for pred_subtask in pred_subtasks):
                # all predecessors finished
                to_schedule_subtasks.append(succ_subtask)
        await self._schedule_subtasks(to_schedule_subtasks)
        await self._scheduling_api.finish_subtasks([result.subtask_id])

    def _schedule_done(self):
        self._done.set()

    @contextmanager
    def _ensure_done_set(self):
        try:
            yield
        finally:
            self._done.set()

    async def schedule(self):
        if len(self._subtask_graph) == 0:
            # no subtask to schedule, set status to done
            self._schedule_done()
            return

        # schedule independent subtasks
        indep_subtasks = list(self._subtask_graph.iter_indep())
        await self._schedule_subtasks(indep_subtasks)

        # wait for completion
        await self._done.wait()

    async def cancel(self):
        if self._done.is_set():
            # already finished, ignore cancel
            return
        self._cancelled.set()
        # cancel running subtasks
        await self._scheduling_api.cancel_subtasks(list(self._submitted_subtask_ids))
        self._done.set()

    def set_task_info(self, error=None, traceback=None):
        self._task_stage_info.task_result = TaskResult(
            self._task_stage_info.task_id, self._task_stage_info.task.session_id,
            TaskStatus.terminated, error=error, traceback=traceback)


@dataclass
class TaskInfo:
    task_id: str
    task_name: str
    session_id: str
    tasks: List[Task]
    task_processors: List[TaskProcessor]
    aio_tasks: List[asyncio.Task]
    task_stage_infos: List["TaskStageInfo"]

    def __init__(self,
                 task_id: str,
                 task_name: str,
                 session_id: str):
        self.task_id = task_id
        self.task_name = task_name
        self.session_id = session_id
        self.tasks = []
        self.task_processors = []
        self.aio_tasks = []
        self.task_stage_infos = []

    @property
    def task_result(self):
        for task_stage in self.task_stage_infos:
            if task_stage.task_result.error is not None:
                return task_stage.task_result
        # all succeeded, return the last task result
        return self.task_stage_infos[-1].task_result


@dataclass
class TaskStageInfo:
    task_id: str
    task_info: TaskInfo
    task: Task
    chunk_graph: ChunkGraph = None
    task_result: TaskResult = None
    subtask_graph: SubtaskGraph = None
    subtask_graph_scheduler: SubtaskGraphScheduler = None
    subtask_results: Dict[str, SubtaskResult] = None
    subtask_result_is_set: Dict[str, bool] = None

    def __init__(self,
                 task_id: str,
                 task_info: TaskInfo,
                 task: Task):
        self.task_id = task_id
        self.task_info = task_info
        self.task = task
        self.task_result = TaskResult(
            task_id, task_info.session_id, TaskStatus.pending)
        self.subtask_results = dict()
        self.subtask_result_is_set = dict()


@dataclass
class ResultTileableInfo:
    tileable: TileableType
    processor: TaskProcessor


class TaskManagerActor(mo.Actor):
    _task_name_to_task_info: Dict[str, TaskInfo]
    _task_id_to_task_info: Dict[str, TaskInfo]
    _task_id_to_task_stage_info: Dict[str, TaskStageInfo]
    _tileable_key_to_info: Dict[str, List[ResultTileableInfo]]

    _cluster_api: Optional[ClusterAPI]
    _meta_api: Optional[MetaAPI]
    _lifecycle_api: Optional[LifecycleAPI]

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._config = None
        self._task_processor_cls = None

        self._task_name_to_task_info = dict()
        self._task_id_to_task_info = dict()
        self._task_id_to_task_stage_info = dict()
        self._tileable_key_to_info = defaultdict(list)

        self._cluster_api = None
        self._meta_api = None
        self._lifecycle_api = None
        self._scheduling_api = None
        self._last_idle_time = None

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)
        self._scheduling_api = await SchedulingAPI.create(self._session_id, self.address)
        self._meta_api = await MetaAPI.create(self._session_id, self.address)
        self._lifecycle_api = await LifecycleAPI.create(
            self._session_id, self.address)

        # get config
        configuration_ref = await mo.actor_ref(
            TaskConfigurationActor.default_uid(),
            address=self.address)
        task_conf = await configuration_ref.get_config()
        self._config, self._task_processor_cls = \
            task_conf['task_options'], task_conf['task_processor_cls']
        self._task_processor_cls = self._get_task_processor_cls()

        # init context
        await self._init_context()

    async def _init_context(self):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            self._session_id, self.address, self.address, loop=loop)
        await context.init()
        set_context(context)

    async def _get_available_band_slots(self) -> Dict[BandType, int]:
        return await self._cluster_api.get_all_bands()

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_task_manager'

    @enter_mode(kernel=True)
    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = None,
                                    extra_config: dict = None) -> str:
        self._last_idle_time = None
        if task_name is None:
            task_id = task_name = new_task_id()
        elif task_name in self._task_name_to_main_task_info:
            # task with the same name submitted before
            task_id = self._task_name_to_main_task_info[task_name].task_id
        else:
            task_id = new_task_id()

        if task_name not in self._task_name_to_task_info:
            # gen main task which mean each submission from user
            task_info = TaskInfo(task_id, task_name, self._session_id)
            self._task_name_to_task_info[task_name] = task_info
            self._task_id_to_task_info[task_id] = task_info
        else:
            task_info = self._task_name_to_main_task_info[task_name]

        if fuse_enabled is None:
            fuse_enabled = self._config.fuse_enabled
        # gen task
        task = Task(task_id, self._session_id,
                    graph, task_name,
                    fuse_enabled=fuse_enabled,
                    extra_config=extra_config)
        task_info.tasks.append(task)
        # gen task processor
        tiled_context = self._gen_tiled_context(graph)
        task_processor = self._task_processor_cls(
            task, tiled_context=tiled_context,
            config=self._config)
        task_info.task_processors.append(task_processor)
        # start to run main task
        aio_task = asyncio.create_task(
            self._process_task(task_processor, task_info, task))
        await asyncio.sleep(0)
        task_info.aio_tasks.append(aio_task)

        return task_id

    def _gen_tiled_context(self, graph: TileableGraph):
        # process graph, add fetch node to tiled context
        tiled_context = dict()
        for tileable in graph:
            if isinstance(tileable.op, Fetch) and tileable.is_coarse():
                info = self._tileable_key_to_info[tileable.key][-1]
                tiled = info.processor.tile_context[info.tileable]
                tiled_context[tileable] = build_fetch(tiled).data
        return tiled_context

    def _get_task_processor_cls(self):
        if self._task_processor_cls is not None:
            assert isinstance(self._task_processor_cls, str)
            module, name = self._task_processor_cls.rsplit('.', 1)
            return getattr(importlib.import_module(module), name)
        else:
            return TaskProcessor

    async def _process_task(self,
                            task_processor: TaskProcessor,
                            task_info: TaskInfo,
                            task: Task):
        loop = asyncio.get_running_loop()

        try:
            raw_tileable_context = task_processor.tile_context.copy()
            # optimization, run it in executor,
            # since optimization may be a CPU intensive operation
            tileable_graph = await loop.run_in_executor(None, task_processor.optimize)
            # incref fetch tileables to ensure fetch data not deleted
            await self._incref_fetch_tileables(tileable_graph, raw_tileable_context)

            chunk_graph_iter = task_processor.tile(tileable_graph)
            lifecycle_processed_tileables = set()
            stages = []
            while True:
                task_stage_info = TaskStageInfo(
                    new_task_id(), task_info, task)

                def next_chunk_graph():
                    try:
                        return next(chunk_graph_iter)
                    except StopIteration:
                        return

                future = loop.run_in_executor(None, next_chunk_graph)
                try:
                    chunk_graph = await future
                    if chunk_graph is None:
                        break

                    task_info.task_stage_infos.append(task_stage_info)
                    stages.append(task_stage_info)
                    task_id = task_stage_info.task_id
                    self._task_id_to_task_stage_info[task_id] = task_stage_info
                except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                    # something wrong while tiling
                    _, err, tb = sys.exc_info()
                    task_stage_info.task_result.status = TaskStatus.terminated
                    task_stage_info.task_result.error = err
                    task_stage_info.task_result.traceback = tb

                    task_info.task_stage_infos.append(task_stage_info)
                    stages.append(task_stage_info)
                    task_id = task_stage_info.task_id
                    self._task_id_to_task_stage_info[task_id] = task_stage_info

                    break

                # track and incref result tileables
                await self._track_and_incref_result_tileables(
                    tileable_graph, task_processor,
                    lifecycle_processed_tileables)

                task_stage_info.chunk_graph = chunk_graph
                # get subtask graph
                available_bands = await self._get_available_band_slots()
                subtask_graph = task_processor.analyze(
                    chunk_graph, available_bands, task_stage_info)
                task_stage_info.subtask_graph = subtask_graph
                await self._incref_after_analyze(subtask_graph, chunk_graph.results)

                # schedule subtask graph
                chunk_optimization_records = None
                if task_processor.chunk_optimization_records_list:
                    chunk_optimization_records = \
                        task_processor.chunk_optimization_records_list[-1]
                subtask_scheduler = SubtaskGraphScheduler(
                    subtask_graph, list(available_bands), task_stage_info,
                    self._meta_api, chunk_optimization_records,
                    self._scheduling_api)
                task_stage_info.subtask_graph_scheduler = subtask_scheduler
                await subtask_scheduler.schedule()

            # iterative tiling and execution finished,
            # set task processor done
            for tileable in tileable_graph.result_tileables:
                info = ResultTileableInfo(tileable=tileable,
                                          processor=task_processor)
                self._tileable_key_to_info[tileable.key].append(info)
            await self._decref_when_finish(
                stages, lifecycle_processed_tileables,
                tileable_graph, raw_tileable_context)
        finally:
            task_processor.done = True

    async def _incref_fetch_tileables(self,
                                      tileable_graph: TileableGraph,
                                      raw_tileable_context: Dict[TileableType, TileableType]):
        # incref fetch tileables in tileable graph to prevent them from deleting
        to_incref_tileable_keys = [
            tileable.op.source_key for tileable in tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in raw_tileable_context]
        await self._lifecycle_api.incref_tileables(to_incref_tileable_keys)

    async def _track_and_incref_result_tileables(self,
                                                 tileable_graph: TileableGraph,
                                                 task_processor: TaskProcessor,
                                                 processed: Set):
        # track and incref result tileables if tiled
        tracks = [], []
        for result_tileable in tileable_graph.result_tileables:
            if result_tileable in processed:  # pragma: no cover
                continue
            try:
                tiled_tileable = task_processor.get_tiled(result_tileable)
                tracks[0].append(result_tileable.key)
                tracks[1].append(self._lifecycle_api.track.delay(
                    result_tileable.key, [c.key for c in tiled_tileable.chunks]))
                processed.add(result_tileable)
            except KeyError:
                # not tiled, skip
                pass
        if tracks:
            await self._lifecycle_api.track.batch(*tracks[1])
            await self._lifecycle_api.incref_tileables(tracks[0])

    async def _incref_after_analyze(self,
                                    subtask_graph: SubtaskGraph,
                                    chunk_results: List[ChunkType]):
        incref_chunk_keys = []
        for subtask in subtask_graph:
            # for subtask has successors, incref number of successors
            n = subtask_graph.count_successors(subtask)
            incref_chunk_keys.extend(
                [c.key for c in subtask.chunk_graph.results] * n)
        incref_chunk_keys.extend([c.key for c in chunk_results])
        await self._lifecycle_api.incref_chunks(incref_chunk_keys)

    async def _decref_when_finish(self,
                                  stages: List[TaskStageInfo],
                                  result_tileables: Set[TileableType],
                                  tileable_graph: TileableGraph,
                                  raw_tileable_context: Dict[TileableType, TileableType]):
        decref_chunks = []
        error_or_cancelled = False
        for stage in stages:
            if stage.task_result.error is not None:
                # error happened
                error_or_cancelled = True
            elif stage.subtask_graph_scheduler.is_cancelled():
                # cancelled
                error_or_cancelled = True
            if stage.subtask_graph:
                if error_or_cancelled:
                    # error or cancel, rollback incref for subtask results
                    for subtask in stage.subtask_graph:
                        if stage.subtask_results.get(subtask.subtask_id):
                            continue
                        # if subtask not executed, rollback incref of predecessors
                        for inp_subtask in stage.subtask_graph.predecessors(subtask):
                            decref_chunks.extend(inp_subtask.chunk_graph.results)
                # decref result of chunk graphs
                decref_chunks.extend(stage.chunk_graph.results)
        await self._lifecycle_api.decref_chunks(
            [c.key for c in decref_chunks])

        fetch_tileable_keys = [
            tileable.op.source_key for tileable in tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in raw_tileable_context]

        if error_or_cancelled:
            # if task failed or cancelled, roll back tileable incref
            await self._lifecycle_api.decref_tileables(
                [r.key for r in result_tileables
                 if not isinstance(r.op, Fetch)] + fetch_tileable_keys)
        else:
            # decref fetch tileables only
            await self._lifecycle_api.decref_tileables(fetch_tileable_keys)

    @classmethod
    async def _wait_for(cls, task_info: TaskInfo):
        processors = task_info.task_processors
        aio_tasks = task_info.aio_tasks
        await asyncio.gather(*processors, *aio_tasks)
        return task_info.task_result

    async def _wait_task(self, task_id: str, timeout=None):
        try:
            task_info = self._task_id_to_task_info[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        if timeout is None:
            return await self._wait_for(task_info)

        task = asyncio.create_task(self._wait_for(task_info))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
            return await task
        except asyncio.TimeoutError:
            return

    async def wait_task(self,
                        task_id: str,
                        timeout: int = None):
        # return coroutine to not block task manager
        return self._wait_task(task_id, timeout=timeout)

    async def _cancel_task(self, task_info: TaskInfo):
        # cancel all stages
        coros = [task_stage_info.subtask_graph_scheduler.cancel()
                 for task_stage_info in task_info.task_stage_infos]
        await asyncio.gather(*coros)

    async def cancel_task(self, task_id: str):
        try:
            task_info = self._task_id_to_task_info[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        # return coroutine to not block current actor
        return self._cancel_task(task_info)

    def get_task_result(self, task_id: str):
        try:
            return self._task_id_to_task_info[task_id].task_result
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

    def get_task_result_tileables(self, task_id: str):
        try:
            task_info = self._task_id_to_task_info[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        processor = task_info.task_processors[-1]
        tileable_graph = processor.tileable_graph
        result = []
        for result_tilable in tileable_graph.result_tileables:
            tiled = processor.get_tiled(result_tilable)
            result.append(build_fetch(tiled))
        return result

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        try:
            task_stage_info = self._task_id_to_task_stage_info[subtask_result.task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {subtask_result.task_id} does not exist')

        if task_stage_info.subtask_result_is_set.get(subtask_result.subtask_id):
            return

        task_stage_info.subtask_results[subtask_result.subtask_id] = subtask_result
        if subtask_result.status.is_done:
            task_stage_info.subtask_result_is_set[subtask_result.subtask_id] = True
            subtask = task_stage_info.subtask_graph_scheduler.subtask_id_to_subtask[
                subtask_result.subtask_id]
            await self._decref_input_subtasks(subtask, task_stage_info.subtask_graph)
            await task_stage_info.subtask_graph_scheduler.set_subtask_result(subtask_result)

    async def _decref_input_subtasks(self,
                                     subtask: Subtask,
                                     subtask_graph: SubtaskGraph):
        decref_chunks = []
        for in_subtask in subtask_graph.iter_predecessors(subtask):
            decref_chunks.extend(in_subtask.chunk_graph.results)
        await self._lifecycle_api.decref_chunks([c.key for c in decref_chunks])

    def get_task_progress(self, task_id: str) -> float:
        # first get all processors
        try:
            task_info = self._task_id_to_task_info[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        tiled_percentage = 0.0
        for task_processor in task_info.task_processors:
            # get tileable proportion that is tiled
            tileable_graph = task_processor.tileable_graph
            tileable_context = task_processor.tile_context
            tiled_percentage += len(tileable_context) / len(tileable_graph)
        tiled_percentage /= len(task_info.task_processors)

        # get progress of stages
        subtask_progress = 0.0
        n_stage = 0
        for stage in task_info.task_stage_infos:
            if stage.subtask_graph is None:  # pragma: no cover
                # generating subtask
                continue
            n_subtask = len(stage.subtask_graph)
            if n_subtask == 0:
                continue
            progress = sum(result.progress for result
                           in stage.subtask_results.values())
            subtask_progress += progress / n_subtask
            n_stage += 1
        if n_stage > 0:
            subtask_progress /= n_stage

        return subtask_progress * tiled_percentage

    def get_last_idle_time(self):
        if self._last_idle_time is None:
            for task_info in self._task_id_to_task_info.values():
                for task_processor in task_info.task_processors:
                    if not task_processor.done:
                        break
                else:
                    for stage in task_info.task_stage_infos:
                        if stage.task_result.status != TaskStatus.terminated:
                            break
                    else:
                        continue
                break
            else:
                self._last_idle_time = time.time()
        return self._last_idle_time
