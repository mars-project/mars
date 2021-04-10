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
import sys
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from .... import oscar as mo
from ....core import TileableGraph, ChunkGraph, ChunkGraphBuilder, get_tiled
from ....utils import build_fetch
from ...cluster.api import ClusterAPI
from ..analyzer import GraphAnalyzer
from ..core import Task, TaskResult, TaskStatus, Subtask, SubtaskResult, \
    SubTaskStatus, SubtaskGraph, new_task_id
from ..errors import TaskNotExist


class TaskProcessor:
    __slots__  = '_task', 'tileable_graph'

    def __init__(self,
                 task: Task):
        self._task = task
        self.tileable_graph = task.tileable_graph

    def optimize(self) -> TileableGraph:
        """
        Optimize tileable graph.

        Returns
        -------
        optimized_graph: TileableGraph

        """
        # TODO(qinxuye): enable optimization
        return self.tileable_graph

    def tile(self, tileable_graph: TileableGraph) -> Iterable[ChunkGraph]:
        """
        Generate chunk graphs

        Returns
        -------
        chunk_graph_generator: Generator
             Chunk graphs.
        """
        # TODO(qinxuye): integrate iterative chunk graph builder
        chunk_graph_builder = ChunkGraphBuilder(
            tileable_graph, fuse_enabled=self._task.fuse_enabled)
        yield from chunk_graph_builder.build()


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
                 bands: List[Tuple[str, str]],
                 task_stage_info: "TaskStageInfo",
                 scheduling_api=None):
        self._subtask_graph = subtask_graph
        self._bands = bands
        self._task_stage_info = task_stage_info
        self._scheduling_api = scheduling_api

        # gen subtask_id to subtask
        self._subtask_id_to_subtask = {subtask.subtask_id: subtask
                                       for subtask in subtask_graph}

        self._subtask_to_bands: Dict[Subtask, Tuple[str, str]] = dict()
        self._subtask_to_results: Dict[Subtask, SubtaskResult] = dict()

        self._band_manager: Dict[Tuple[str, str], mo.ActorRef] = dict()
        self._band_queue: Dict[Tuple[str, str], BandQueue] = defaultdict(BandQueue)

        self._band_schedules = []

        self._done = asyncio.Event()
        self._cancelled = asyncio.Event()

    async def _get_band_subtask_manager(self, band: Tuple[str, str]):
        from ..worker.subtask import BandSubtaskManagerActor

        if band in self._band_manager:
            return self._band_manager[band]

        manger_ref = await mo.actor_ref(
            band[0], BandSubtaskManagerActor.gen_uid(band[1]))
        self._band_manager[band] = manger_ref
        return manger_ref

    def _get_subtask_band(self, subtask: Subtask):
        if subtask.expect_band is not None:
            # start, already specified band
            self._subtask_to_bands[subtask] = band = subtask.expect_band
            return band
        else:
            inp_subtasks = self._subtask_graph.predecessors(subtask)
            # calculate a expect band
            max_size_inp_subtask = max(
                inp_subtasks, key=lambda st: self._subtask_to_results[st].data_size)
            band = self._subtask_to_bands[max_size_inp_subtask]
            self._subtask_to_bands[subtask] = band
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

    async def set_subtask_result(self, result: SubtaskResult):
        subtask_id = result.subtask_id
        subtask = self._subtask_id_to_subtask[subtask_id]
        self._subtask_to_results[subtask] = result

        all_done = len(self._subtask_to_results) == len(self._subtask_graph)
        error_or_cancelled = result.status in (SubTaskStatus.errored, SubTaskStatus.cancelled)

        if all_done or error_or_cancelled:
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
                status=SubTaskStatus.errored,
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

    async def _schedule_band(self, band: Tuple[str, str]):
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


class TaskManagerActor(mo.Actor):
    def __init__(self,
                 session_id,
                 use_scheduling=True):
        self._session_id = session_id

        self._task_name_to_task_info: Dict[str, TaskInfo] = dict()
        self._task_id_to_task_info: Dict[str, TaskInfo] = dict()
        self._task_id_to_task_stage_info: Dict[str, TaskStageInfo] = dict()

        self._cluster_api = None
        self._use_scheduling = use_scheduling

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)

    async def _get_available_band_slots(self) -> Dict[Tuple[str, str], int]:
        return await self._cluster_api.get_all_bands()

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_task_manager'

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True) -> str:

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

        # gen task
        task = Task(task_id, self._session_id,
                    graph, task_name,
                    fuse_enabled=fuse_enabled)
        task_info.tasks.append(task)
        # gen task processor
        task_processor = TaskProcessor(task)
        task_info.task_processors.append(task_processor)
        # start to run main task
        aio_task = asyncio.create_task(
            self._process_task(task_processor, task_info, task))
        await asyncio.sleep(0)
        task_info.aio_tasks.append(aio_task)

        return task_id

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

            # get subtask graph
            available_bands = await self._get_available_band_slots()
            analyzer = GraphAnalyzer(chunk_graph, available_bands,
                                     task_stage_info)
            subtask_graph = analyzer.gen_subtask_graph()
            task_stage_info.subtask_graph = subtask_graph

            # schedule subtask graph
            # TODO(qinxuye): pass scheduling API to scheduler when it's ready
            subtask_scheduler = SubtaskGraphScheduler(
                subtask_graph, list(available_bands), task_stage_info)
            task_stage_info.subtask_graph_scheduler = subtask_scheduler
            await subtask_scheduler.schedule()

    async def _wait_task(self, task_id: str):
        try:
            task_info = self._task_id_to_task_info[task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {task_id} does not exist')

        aio_tasks = task_info.aio_tasks
        await asyncio.gather(*aio_tasks)
        return task_info.task_result

    async def wait_task(self, task_id: str):
        # return coroutine to not block task manager
        return self._wait_task(task_id)

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

        tileable_graph = task_info.task_processors[-1].tileable_graph
        result = []
        for result_tilable in tileable_graph.result_tileables:
            tiled = get_tiled(result_tilable)
            result.append(build_fetch(tiled))
        return result

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        try:
            task_stage_info = self._task_id_to_task_stage_info[subtask_result.task_id]
        except KeyError:  # pragma: no cover
            raise TaskNotExist(f'Task {subtask_result.task_id} does not exist')

        task_stage_info.subtask_results[subtask_result.subtask_id] = subtask_result
        await task_stage_info.subtask_graph_scheduler.set_subtask_result(subtask_result)
