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
import itertools
import logging
import time
from collections import defaultdict
from typing import Dict, List

from ..... import oscar as mo
from .....core import ChunkGraph, Chunk
from .....core.operand import Fuse, Fetch
from .....oscar import ServerClosed
from .....metrics import Metrics
from .....oscar.errors import DuplicatedSubtaskError, DataNotExist
from .....utils import get_chunk_params
from .....typing import BandType, TileableType
from ....context import FailOverContext
from ....meta import MetaAPI, WorkerMetaAPI
from ....scheduling import SchedulingAPI

from ....subtask import Subtask, SubtaskGraph, SubtaskResult, SubtaskStatus
from ....task.core import Task, TaskResult, TaskStatus
from ..api import ExecutionChunkResult

logger = logging.getLogger(__name__)


class TaskStageProcessor:
    def __init__(
        self,
        stage_id: str,
        task: Task,
        chunk_graph: ChunkGraph,
        subtask_graph: SubtaskGraph,
        bands: List[BandType],
        tile_context: Dict[TileableType, TileableType],
        scheduling_api: SchedulingAPI,
        meta_api: MetaAPI,
    ):
        self.stage_id = stage_id
        self.task = task
        self.chunk_graph = chunk_graph
        self.subtask_graph = subtask_graph
        self.generation_order = next(self.task.counter)
        self._bands = bands
        self._tile_context = tile_context

        # APIs
        self._scheduling_api = scheduling_api
        self._meta_api = meta_api

        # gen subtask_id to subtask
        self.subtask_id_to_subtask = {
            subtask.subtask_id: subtask for subtask in subtask_graph
        }
        self._subtask_to_bands: Dict[Subtask, BandType] = dict()
        self.subtask_snapshots: Dict[Subtask, SubtaskResult] = dict()
        self.subtask_results: Dict[Subtask, SubtaskResult] = dict()
        self._submitted_subtask_ids = set()

        # All subtask IDs whose input chunk reference count is reduced.
        self.decref_subtask = set()

        self._band_manager: Dict[BandType, mo.ActorRef] = dict()

        # result
        self.result = TaskResult(
            task.task_id,
            task.session_id,
            self.stage_id,
            status=TaskStatus.pending,
            start_time=time.time(),
        )
        # status
        self._done = asyncio.Event()
        self._cancelled = asyncio.Event()

        # add metrics
        self._stage_execution_time = Metrics.gauge(
            "mars.stage_execution_time_secs",
            "Time consuming in seconds to execute a stage",
            ("session_id", "task_id", "stage_id"),
        )

    def is_cancelled(self):
        return self._cancelled.is_set()

    async def _schedule_subtasks(self, subtasks: List[Subtask]):
        subtasks = [
            subtask
            for subtask in subtasks
            if subtask.subtask_id not in self._submitted_subtask_ids
        ]
        if not subtasks:
            return
        self._submitted_subtask_ids.update(subtask.subtask_id for subtask in subtasks)
        return await self._scheduling_api.add_subtasks(
            subtasks, [subtask.priority for subtask in subtasks]
        )

    async def _get_stage_result(self):
        chunks = []
        get_meta = []
        results_chunks = self.chunk_graph.result_chunks
        for chunk in results_chunks:
            if isinstance(chunk.op, Fetch):
                continue
            chunks.append(chunk)
            if isinstance(chunk.op, Fuse):
                chunk = chunk.chunk
            get_meta.append(
                self._meta_api.get_chunk_meta.delay(
                    chunk.key,
                    # only fetch bands from supervisor meta
                    fields=["bands"],
                )
            )
        metas = await self._meta_api.get_chunk_meta.batch(*get_meta)
        execution_chunk_results = {
            chunk: ExecutionChunkResult(meta=meta, context=None)
            for chunk, meta in zip(chunks, metas)
        }
        await self._update_result_meta(execution_chunk_results)
        return execution_chunk_results

    def _schedule_done(self):
        self._done.set()

    async def set_subtask_result(self, result: SubtaskResult, band: BandType = None):
        assert result.status.is_done
        subtask = self.subtask_id_to_subtask[result.subtask_id]
        #  update subtask_results in `TaskProcessorActor.set_subtask_result`
        self._submitted_subtask_ids.difference_update([result.subtask_id])

        finished = len(self.subtask_results) == len(self.subtask_graph) and all(
            r.status == SubtaskStatus.succeeded for r in self.subtask_results.values()
        )
        errored = result.status == SubtaskStatus.errored
        cancelled = result.status == SubtaskStatus.cancelled

        logger.debug(
            "Setting subtask %s, finished: %s, errored: %s, cancelled: %s, "
            "subtask_results len: %d, subtask_graph len: %d, "
            "result status: %s, self.result.status: %s, "
            "task id: %s, stage id: %s.",
            subtask,
            finished,
            errored,
            cancelled,
            len(self.subtask_results),
            len(self.subtask_graph),
            result.status,
            self.result.status,
            self.task.task_id,
            self.stage_id,
        )

        if finished and self.result.status != TaskStatus.terminated:
            logger.info(
                "Stage %s of task %s is finished.", self.stage_id, self.task.task_id
            )
            await self._scheduling_api.finish_subtasks(
                [result.subtask_id], bands=[band], schedule_next=False
            )

            # Note: Should trigger subtasks which dependent on the finished subtask
            await self._try_trigger_subtasks(subtask)

            self.result = TaskResult(
                self.task.task_id,
                self.task.session_id,
                self.stage_id,
                start_time=self.result.start_time,
                end_time=time.time(),
                status=TaskStatus.terminated,
                error=result.error,
                traceback=result.traceback,
            )
            self._schedule_done()
            cost_time_secs = self.result.end_time - self.result.start_time
            logger.info(
                "Time consuming to execute a stage is %ss with "
                "session id %s, task id %s, stage id %s",
                cost_time_secs,
                self.result.session_id,
                self.result.task_id,
                self.result.stage_id,
            )
            self._stage_execution_time.record(
                cost_time_secs,
                {
                    "session_id": self.result.session_id,
                    "task_id": self.result.task_id,
                    "stage_id": self.result.stage_id,
                },
            )
        elif errored:
            self.result = TaskResult(
                self.task.task_id,
                self.task.session_id,
                self.stage_id,
                start_time=self.result.start_time,
                end_time=time.time(),
                status=TaskStatus.errored,
                error=result.error,
                traceback=result.traceback,
            )
            logger.exception(
                "Subtask %s errored",
                subtask.subtask_id,
                exc_info=(
                    type(result.error),
                    result.error,
                    result.traceback,
                ),
            )
            ret = await self._detect_error(
                subtask,
                result.error,
                (
                    ServerClosed,
                    DataNotExist,
                ),
            )
            if ret:
                await self._scheduling_api.finish_subtasks(
                    [result.subtask_id],
                    bands=[band],
                    schedule_next=True,
                )
                return
            else:
                await self._scheduling_api.finish_subtasks(
                    [result.subtask_id],
                    bands=[band],
                    schedule_next=False,
                )
                logger.info(
                    "Unable to recover data and start to "
                    "cancel stage %s of task %s.",
                    self.stage_id,
                    self.task,
                )
                await self.cancel()
        elif cancelled:
            await self._scheduling_api.finish_subtasks(
                [result.subtask_id], bands=[band], schedule_next=False
            )
            logger.warning(
                "Subtask %s from band %s cancelled.",
                subtask.subtask_id,
                band,
            )
            if self.result.status != TaskStatus.terminated:
                self.result.status = TaskStatus.terminated
                if not self.result.error:
                    self.result.end_time = time.time()
                    self.result.error = result.error
                    self.result.traceback = result.traceback

                logger.info(
                    "Start to cancel stage %s of task %s.",
                    self.stage_id,
                    self.task,
                )
                await self.cancel()
        else:
            await async_call(
                self._scheduling_api.finish_subtasks([result.subtask_id], bands=[band])
            )
            logger.debug(
                "Continue to schedule subtasks after subtask %s finished.",
                subtask.subtask_id,
            )
            # not terminated, push success subtasks to queue if they are ready
            to_schedule_subtasks = []
            for succ_subtask in self.subtask_graph.successors(subtask):
                if succ_subtask in self.subtask_results:  # pragma: no cover
                    continue
                pred_subtasks = self.subtask_graph.predecessors(succ_subtask)
                if all(
                    pred_subtask in self.subtask_results
                    for pred_subtask in pred_subtasks
                ):
                    to_schedule_subtasks.append(succ_subtask)

            for to_schedule_subtask in to_schedule_subtasks:
                try:
                    logger.info(
                        "Start to schedule subtask %s, task id: %s, stage id: %s.",
                        to_schedule_subtask,
                        self.task.task_id,
                        self.stage_id,
                    )
                    await async_call(self._schedule_subtasks([to_schedule_subtask]))
                except KeyError:
                    logger.exception("Got KeyError.")

            if not to_schedule_subtasks:
                await self._try_trigger_subtasks(subtask)

    async def run(self):
        try:
            if self.subtask_graph.num_shuffles() > 0:
                # disable scale-in when shuffle is executing so that we can skip
                # store shuffle meta in supervisor.
                await self._scheduling_api.disable_autoscale_in()
            return await self._run()
        finally:
            if self.subtask_graph.num_shuffles() > 0:
                await self._scheduling_api.try_enable_autoscale_in()

    async def _run(self):
        if len(self.subtask_graph) == 0:
            # no subtask to schedule, set status to done
            self._schedule_done()
            self.result.status = TaskStatus.terminated
            return {}

        # schedule independent subtasks
        indep_subtasks = list(self.subtask_graph.iter_indep())
        await self._schedule_subtasks(indep_subtasks)

        # wait for completion
        await self._done.wait()
        if self.error_or_cancelled():
            if self.result.error is not None:
                raise self.result.error.with_traceback(self.result.traceback)
            else:
                raise asyncio.CancelledError()
        return await self._get_stage_result()

    async def cancel(self):
        if self._done.is_set() or self._cancelled.is_set():  # pragma: no cover
            # already finished, ignore cancel
            return
        logger.info("Start to cancel stage %s of task %s.", self.stage_id, self.task)
        self._cancelled.set()
        # cancel running subtasks
        await self._scheduling_api.cancel_subtasks(list(self._submitted_subtask_ids))
        self._done.set()

    def error_or_cancelled(self) -> bool:
        if self.result.error is not None:
            return True
        if self.is_cancelled():
            return True
        return False

    async def _update_result_meta(
        self, chunk_to_result: Dict[Chunk, ExecutionChunkResult]
    ):
        session_id = self.task.session_id
        tile_context = self._tile_context

        update_meta_chunks = chunk_to_result.keys() - set(
            itertools.chain.from_iterable(
                (c.data for c in tiled_tileable.chunks)
                for tiled_tileable in tile_context.values()
            )
        )

        worker_meta_api_to_chunk_delays = defaultdict(dict)
        for c in update_meta_chunks:
            address = chunk_to_result[c].meta["bands"][0][0]
            meta_api = await WorkerMetaAPI.create(session_id, address)
            call = meta_api.get_chunk_meta.delay(
                c.key, fields=list(get_chunk_params(c).keys())
            )
            worker_meta_api_to_chunk_delays[meta_api][c] = call
        for tileable in tile_context.values():
            chunks = [c.data for c in tileable.chunks]
            for c, params_fields in zip(chunks, self._get_params_fields(tileable)):
                address = chunk_to_result[c].meta["bands"][0][0]
                meta_api = await WorkerMetaAPI.create(session_id, address)
                call = meta_api.get_chunk_meta.delay(c.key, fields=params_fields)
                worker_meta_api_to_chunk_delays[meta_api][c] = call
        coros = []
        for worker_meta_api, chunk_delays in worker_meta_api_to_chunk_delays.items():
            coros.append(worker_meta_api.get_chunk_meta.batch(*chunk_delays.values()))
        worker_metas = await asyncio.gather(*coros)
        for chunk_delays, metas in zip(
            worker_meta_api_to_chunk_delays.values(), worker_metas
        ):
            for c, meta in zip(chunk_delays, metas):
                chunk_to_result[c].meta = meta

    @classmethod
    def _get_params_fields(cls, tileable: TileableType):
        from .....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
        from .....tensor.core import TENSOR_TYPE

        params_fields = []
        fields = get_chunk_params(tileable.chunks[0])
        if isinstance(tileable, DATAFRAME_TYPE):
            for c in tileable.chunks:
                cur_fields = set(fields)
                if c.index[1] > 0:
                    # skip fetch index_value for i >= 1 on column axis
                    cur_fields.remove("index_value")
                if c.index[0] > 0:
                    # skip fetch dtypes_value for i >= 1 on index axis
                    cur_fields.remove("dtypes_value")
                if c.index[0] > 0 and c.index[1] > 0:
                    # fetch shape only for i == 0 on index or column axis
                    cur_fields.remove("shape")
                params_fields.append(list(cur_fields))
        elif isinstance(tileable, SERIES_TYPE):
            for c in tileable.chunks:
                cur_fields = set(fields)
                if c.index[0] > 0:
                    # skip fetch name and dtype for i >= 1
                    cur_fields.remove("name")
                    cur_fields.remove("dtype")
                params_fields.append(list(cur_fields))
        elif isinstance(tileable, TENSOR_TYPE):
            for i, c in enumerate(tileable.chunks):
                cur_fields = set(fields)
                if c.ndim > 1 and all(j > 0 for j in c.index):
                    cur_fields.remove("shape")
                if i > 0:
                    cur_fields.remove("dtype")
                    cur_fields.remove("order")
                params_fields.append(list(cur_fields))
        else:
            for _ in tileable.chunks:
                params_fields.append(list(fields))
        return params_fields

    async def _detect_error(self, subtask, error, expect_error_cls_tuple):
        """
        Detect error and trigger error recovery.
        For example: `ServerClosed`, `DataNotExist`.
        Parameters
        ----------
        subtask: subtask
        error: subtask execution error
        expect_error_cls_tuple: error class or error class tuple

        Returns
        -------
        False if could not rerun subtask else True.
        """
        if not FailOverContext.is_lineage_enabled():
            logger.info("Lineage of failover is not enabled.")
            return False

        # Note: There are some error that do not need to be handled,
        # like `DuplicatedSubtaskError`.
        if isinstance(error, DuplicatedSubtaskError):
            logger.info("Ignored error %s of subtask %s.", error, subtask.subtask_id)
            return True

        if isinstance(error, expect_error_cls_tuple):
            logger.info(
                "%s detected of subtask %s and try to recover it.",
                error,
                subtask.subtask_id,
            )
            try:
                # 1. find dependency subtasks
                dependency_subtasks = self.subtask_graph.predecessors(subtask)
                dependency_subtasks_chunk_data_keys = set(
                    chunk.key
                    for s in dependency_subtasks
                    for chunk in s.chunk_graph.result_chunks
                )
                subtask_chunk_data_keys = set(
                    chunk.key
                    for chunk in subtask.chunk_graph.iter_indep()
                    if isinstance(chunk.op, Fetch)
                )
                diff_chunk_data_keys = (
                    subtask_chunk_data_keys - dependency_subtasks_chunk_data_keys
                )

                dependency_subtask_graph_orders = [self.generation_order] * len(
                    dependency_subtasks
                )

                # Note: Could not import `TaskManagerActor` directly because of
                # circular dependency, so use import_module.
                task_manager_actor_cls = getattr(
                    importlib.import_module("mars.services.task.supervisor.manager"),
                    "TaskManagerActor",
                )
                task_manager_ref = await mo.actor_ref(
                    self._scheduling_api.address,
                    task_manager_actor_cls.gen_uid(subtask.session_id),
                )
                for chunk_data_key in diff_chunk_data_keys:
                    s = await task_manager_ref.get_subtask(chunk_data_key)
                    if not s.retryable:
                        logger.info(
                            "Subtask %s is not retryable, so cannot "
                            "recover subtask %s.",
                            s,
                            subtask,
                        )
                        FailOverContext.cleanup()
                        return False
                    if s not in dependency_subtasks:
                        order = await task_manager_ref.get_generation_order(
                            s.task_id, s.stage_id
                        )
                        dependency_subtasks.append(s)
                        dependency_subtask_graph_orders.append(order)
                        FailOverContext.subtask_to_dependency_subtasks[subtask].add(s)

                # 2. submit the subtask and it's dependency subtasks
                if not dependency_subtasks:
                    logger.warning(
                        "No dependent subtasks to restore of subtask %s.",
                        subtask.subtask_id,
                    )
                    FailOverContext.cleanup()
                    return False
                priorities = [
                    (pri,) + s.priority
                    for s, pri in zip(
                        dependency_subtasks, dependency_subtask_graph_orders
                    )
                ]

                logger.info(
                    "Rescheduling subtasks %s with priorities %s to restore "
                    "data, because subtask %s need them.",
                    dependency_subtasks,
                    priorities,
                    subtask,
                )
                # Note: May add duplicated subtasks, so should catch exception
                for s, p in zip(dependency_subtasks, priorities):
                    try:
                        await self._scheduling_api.add_subtasks([s], [p], True)
                    except DuplicatedSubtaskError:
                        logger.exception(
                            "Adding dependency subtask %s failed.", s.subtask_id
                        )
                return True
            except:
                FailOverContext.cleanup()
                logger.exception("Error recovery failed.")
                return False
        else:
            FailOverContext.cleanup()
            logger.error("Could not to recover the error: %s", error)
            return False

    async def _try_trigger_subtasks(self, subtask: Subtask):
        to_schedule_subtasks = []
        if FailOverContext.subtask_to_dependency_subtasks:
            logger.info(
                "Finding subtasks to be scheduled in history set after subtask %s finished.",
                subtask.subtask_id,
            )
            for (
                succ_subtask,
                pred_subtasks,
            ) in FailOverContext.subtask_to_dependency_subtasks.items():
                try:
                    pred_subtasks.remove(subtask)
                except KeyError:
                    pass
                if not pred_subtasks:
                    to_schedule_subtasks.append(succ_subtask)
            if to_schedule_subtasks:
                logger.info(
                    "Found subtasks %s to be scheduled after subtask %s finished.",
                    to_schedule_subtasks,
                    subtask.subtask_id,
                )
                for s in to_schedule_subtasks:
                    FailOverContext.subtask_to_dependency_subtasks.pop(s, None)
            else:
                logger.info("No subtasks found to be scheduled.")
        for to_schedule_subtask in to_schedule_subtasks:
            try:
                logger.info(
                    "Start to schedule subtask %s, task id: %s, stage id: %s.",
                    to_schedule_subtask,
                    self.task.task_id,
                    self.stage_id,
                )
                await self._scheduling_api.add_subtasks(
                    [to_schedule_subtask], [to_schedule_subtask.priority]
                )
            except KeyError:
                logger.exception("Got KeyError.")
