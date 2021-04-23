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
import functools
import importlib
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple, Type, Optional

from .... import oscar as mo
from ....config import Config
from ....core import TileableGraph, ChunkGraph, ChunkGraphBuilder, \
    Tileable, TileableType
from ....core.operand import Fetch, Fuse
from ....dataframe.core import DATAFRAME_CHUNK_TYPE
from ....optimization.logical.chunk import optimize as optimize_chunk_graph
from ....optimization.logical.tileable import optimize as optimize_tileable_graph
from ....utils import build_fetch
from ...cluster.api import ClusterAPI
from ...core import BandType
from ...meta.api import MetaAPI
from ..analyzer import GraphAnalyzer
from ..config import task_options
from ..core import Task, TaskResult, TaskStatus, Subtask, SubtaskResult, \
    SubtaskStatus, SubtaskGraph, new_task_id
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

    tile_context: Dict[Tileable, Tileable]

    def __init__(self,
                 task: Task,
                 tiled_context: Dict[Tileable, Tileable] = None,
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

    def tile(self, tileable_graph: TileableGraph) -> Iterable[ChunkGraph]:
        """
        Generate chunk graphs

        Returns
        -------
        chunk_graph_generator: Generator
             Chunk graphs.
        """
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

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @done.setter
    def done(self, is_done: bool):
        if is_done:
            self._done.set()
        else:  # pragma: no cover
            self._done.clear()

    def get_tiled(self, tileable):
        tileable = tileable.data if hasattr(tileable, 'data') else tileable
        return self.tile_context[tileable]

    @classmethod
    def _update_tileable_params(cls,
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
                 scheduling_api=None):
        self._subtask_graph = subtask_graph
        self._bands = bands
        self._task_stage_info = task_stage_info
        self._meta_api = meta_api
        self._scheduling_api = scheduling_api

        # gen subtask_id to subtask
        self._subtask_id_to_subtask = {subtask.subtask_id: subtask
                                       for subtask in subtask_graph}

        self._subtask_to_bands: Dict[Subtask, BandType] = dict()
        self._subtask_to_results: Dict[Subtask, SubtaskResult] = dict()

        self._band_manager: Dict[BandType, mo.ActorRef] = dict()
        self._band_queue: Dict[BandType, BandQueue] = defaultdict(BandQueue)

        self._band_schedules = []

        self._done = asyncio.Event()
        self._cancelled = asyncio.Event()

    async def _get_band_subtask_manager(self, band: BandType):
        from ..worker.subtask import BandSubtaskManagerActor

        if band in self._band_manager:
            return self._band_manager[band]

        manger_ref = await mo.actor_ref(
            band[0], BandSubtaskManagerActor.gen_uid(band[1]))
        self._band_manager[band] = manger_ref
        return manger_ref

    @functools.lru_cache(30)
    def _calc_expect_band(self, inp_subtasks: Tuple[Subtask]):
        if len(inp_subtasks) == 1 and inp_subtasks[0].virtual:
            # virtual node, get predecessors of virtual node
            calc_subtasks = self._subtask_graph.predecessors(inp_subtasks[0])
        else:
            calc_subtasks = inp_subtasks

        # calculate a expect band
        sorted_size_inp_subtask = sorted(
            calc_subtasks, key=lambda st: self._subtask_to_results[st].data_size,
            reverse=True)
        expect_bands = [self._subtask_to_bands[subtask]
                        for subtask in sorted_size_inp_subtask]
        return expect_bands

    def _get_subtask_band(self, subtask: Subtask):
        if subtask.expect_band is not None:
            # start, already specified band
            self._subtask_to_bands[subtask] = band = subtask.expect_band
            return band
        else:
            inp_subtasks = self._subtask_graph.predecessors(subtask)
            # calculate a expect band
            expect_bands = self._calc_expect_band(tuple(inp_subtasks))
            subtask.expect_bands = expect_bands
            self._subtask_to_bands[subtask] = band = subtask.expect_band
            return band

    async def _direct_submit_subtasks(self, subtasks: List[Subtask]):
        for subtask in subtasks[::-1]:
            band = self._get_subtask_band(subtask)
            # push subtask to queues
            self._band_queue[band].put(subtask)

    async def _schedule_subtasks(self, subtasks: List[Subtask]):
        if self._scheduling_api is not None:
            return await self._scheduling_api.submit_subtasks(
                subtasks, [subtask.priority for subtask in subtasks])
        else:
            return await self._direct_submit_subtasks(subtasks)

    async def _update_chunks_meta(self, chunk_graph: ChunkGraph):
        get_meta = []
        chunks = chunk_graph.result_chunks
        for chunk in chunks:
            if isinstance(chunk.op, Fuse):
                chunk = chunk.chunk
            fields = list(chunk.params)
            if isinstance(chunk, DATAFRAME_CHUNK_TYPE):
                fields.remove('dtypes')
                fields.remove('columns_value')
            get_meta.append(self._meta_api.get_chunk_meta.delay(
                chunk.key, fields=fields))
        metas = await self._meta_api.get_chunk_meta.batch(*get_meta)
        for chunk, meta in zip(chunks, metas):
            chunk.params = meta

    async def set_subtask_result(self, result: SubtaskResult):
        subtask_id = result.subtask_id
        subtask = self._subtask_id_to_subtask[subtask_id]
        self._subtask_to_results[subtask] = result

        all_done = len(self._subtask_to_results) == len(self._subtask_graph)
        error_or_cancelled = result.status in (SubtaskStatus.errored, SubtaskStatus.cancelled)

        if all_done or error_or_cancelled:
            if all_done and not error_or_cancelled:
                # subtask graph finished, update result chunks' meta
                await self._update_chunks_meta(
                    self._task_stage_info.chunk_graph)
            self._schedule_done()
            self.set_task_info(result.error, result.traceback)
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

    def _schedule_done(self):
        self._done.set()
        for q in self._band_queue.values():
            # put None into queue to indicate done
            q.put(None)
        # cancel band schedules
        if not self._cancelled.is_set():
            _ = [schedule.cancel() for schedule in self._band_schedules]

    async def _run_subtask(self,
                           subtask_runner,
                           subtask: Subtask,
                           tasks: Dict):
        try:
            await subtask_runner.run_subtask(subtask)
        except:  # noqa: E722  # pragma: no cover  # pylint: disable=bare-except
            _, err, traceback = sys.exc_info()
            subtask_result = SubtaskResult(
                subtask_id=subtask.subtask_id,
                session_id=subtask.session_id,
                task_id=subtask.task_id,
                status=SubtaskStatus.errored,
                error=err,
                traceback=traceback)
            await self.set_subtask_result(subtask_result)
        del tasks[subtask]

    @contextmanager
    def _ensure_done_set(self):
        try:
            yield
        finally:
            self._done.set()

    async def _schedule_band(self, band: BandType):
        with self._ensure_done_set():
            manager_ref = await self._get_band_subtask_manager(band)
            tasks = dict()

            q = self._band_queue[band]
            while not self._done.is_set():
                # wait for data
                try:
                    # await finish when sth enqueued.
                    # note that now we don't get subtask from queue,
                    # just ensure the process can be continued.
                    # since the slot is released after subtask runner
                    # notifies task manager, we will get subtask when slot released,
                    # so that subtask with higher priority is fetched
                    await q
                except asyncio.CancelledError:
                    pass

                if not self._cancelled.is_set():
                    try:
                        subtask_runner = await manager_ref.get_free_slot()
                    except asyncio.CancelledError:
                        subtask_runner = None
                else:
                    subtask_runner = None

                # now get subtask, the subtask that can run with higher priority
                # has been pushed before slot released
                subtask = q.get()

                done = subtask is None or self._done.is_set() or self._cancelled.is_set()
                if done and subtask_runner:
                    # finished or cancelled, given back slot
                    await manager_ref.mark_slot_free(subtask_runner)

                if self._cancelled.is_set():
                    # force to free running slots
                    free_slots = []
                    for subtask_runner, _ in tasks.values():
                        free_slots.append(
                            manager_ref.free_slot(subtask_runner))
                    await asyncio.gather(*free_slots)
                elif not done:
                    coro = self._run_subtask(subtask_runner, subtask, tasks)
                    tasks[subtask] = (subtask_runner, asyncio.create_task(coro))

            # done, block until all tasks finish
            if not self._cancelled.is_set():
                await asyncio.gather(*[v[1] for v in tasks.values()])

    async def schedule(self):
        if self._scheduling_api is None:
            # use direct submit
            for band in self._bands:
                self._band_schedules.append(
                    asyncio.create_task(self._schedule_band(band)))

        # schedule independent subtasks
        indep_subtasks = list(self._subtask_graph.iter_indep())
        await self._schedule_subtasks(indep_subtasks)

        # wait for completion
        await self._done.wait()
        # wait for schedules to complete
        await asyncio.gather(*self._band_schedules)

    async def cancel(self):
        if self._done.is_set():
            # already finished, ignore cancel
            return
        self._cancelled.set()
        _ = [s.cancel() for s in self._band_schedules]
        await asyncio.gather(*self._band_schedules)
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


@dataclass
class ResultTileableInfo:
    tileable: Tileable
    processor: TaskProcessor


class TaskManagerActor(mo.Actor):
    _task_name_to_task_info: Dict[str, TaskInfo]
    _task_id_to_task_info: Dict[str, TaskInfo]
    _task_id_to_task_stage_info: Dict[str, TaskStageInfo]
    _tileable_key_to_info: Dict[str, List[ResultTileableInfo]]

    def __init__(self,
                 session_id,
                 use_scheduling=True):
        self._session_id = session_id
        self._config = None
        self._task_processor_cls = None

        self._task_name_to_task_info = dict()
        self._task_id_to_task_info = dict()
        self._task_id_to_task_stage_info = dict()
        self._tileable_key_to_info = defaultdict(list)

        self._meta_api = None
        self._cluster_api = None
        self._use_scheduling = use_scheduling
        self._last_idle_time = None

    async def __post_create__(self):
        self._meta_api = await MetaAPI.create(self._session_id, self.address)
        self._cluster_api = await ClusterAPI.create(self.address)

        # get config
        configuration_ref = await mo.actor_ref(
            TaskConfigurationActor.default_uid(),
            address=self.address)
        task_conf = await configuration_ref.get_config()
        self._config, self._task_processor_cls = \
            task_conf['task_options'], task_conf['task_processor_cls']
        self._task_processor_cls = self._get_task_processor_cls()

    async def _get_available_band_slots(self) -> Dict[BandType, int]:
        return await self._cluster_api.get_all_bands()

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_task_manager'

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
            if isinstance(tileable.op, Fetch):
                info = self._tileable_key_to_info[tileable.key][0]
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

        # optimization, run it in executor,
        # since optimization may be a CPU intensive operation
        tileable_graph = await loop.run_in_executor(None, task_processor.optimize)

        chunk_graph_iter = task_processor.tile(tileable_graph)
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
                task_id = task_stage_info.task_id
                self._task_id_to_task_stage_info[task_id] = task_stage_info
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
                # something wrong while tiling
                _, err, tb = sys.exc_info()
                task_stage_info.task_result.status = TaskStatus.terminated
                task_stage_info.task_result.error = err
                task_stage_info.task_result.traceback = tb

                task_info.task_stage_infos.append(task_stage_info)
                task_id = task_stage_info.task_id
                self._task_id_to_task_stage_info[task_id] = task_stage_info

                break

            task_stage_info.chunk_graph = chunk_graph
            # get subtask graph
            available_bands = await self._get_available_band_slots()
            analyzer = GraphAnalyzer(chunk_graph, available_bands,
                                     task.fuse_enabled, task.extra_config,
                                     task_stage_info)
            subtask_graph = analyzer.gen_subtask_graph()
            task_stage_info.subtask_graph = subtask_graph

            # schedule subtask graph
            # TODO(qinxuye): pass scheduling API to scheduler when it's ready
            subtask_scheduler = SubtaskGraphScheduler(
                subtask_graph, list(available_bands), task_stage_info,
                self._meta_api)
            task_stage_info.subtask_graph_scheduler = subtask_scheduler
            await subtask_scheduler.schedule()

        # iterative tiling and execution finished,
        # set task processor done
        for tileable in tileable_graph.result_tileables:
            info = ResultTileableInfo(tileable=tileable,
                                      processor=task_processor)
            self._tileable_key_to_info[tileable.key].append(info)
        task_processor.done = True

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

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = asyncio.create_task(self._wait_for(task_info))

        def cb(_):
            try:
                future.set_result(None)
            except asyncio.InvalidStateError:  # pragma: no cover
                pass

        task.add_done_callback(cb)
        try:
            await asyncio.wait_for(future, timeout)
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

        task_stage_info.subtask_results[subtask_result.subtask_id] = subtask_result
        if subtask_result.status.is_done:
            await task_stage_info.subtask_graph_scheduler.set_subtask_result(subtask_result)

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
        for stage in task_info.task_stage_infos:
            n_subtask = len(stage.subtask_graph)
            progress = sum(result.progress for result
                           in stage.subtask_results.values())
            subtask_progress += progress / n_subtask
        subtask_progress /= len(task_info.task_stage_infos)

        return subtask_progress * tiled_percentage

    def last_idle_time(self):
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
