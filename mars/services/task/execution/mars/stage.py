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
import time
from collections import defaultdict
from typing import Dict, List

from ..... import oscar as mo
from .....core import ChunkGraph, Chunk
from .....core.operand import Fuse, Fetch
from .....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from .....metrics import Metrics
from .....utils import get_chunk_params
from .....tensor.core import TENSOR_TYPE
from .....typing import BandType, TileableType
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

        all_done = len(self.subtask_results) == len(self.subtask_graph)
        error_or_cancelled = result.status in (
            SubtaskStatus.errored,
            SubtaskStatus.cancelled,
        )

        if all_done or error_or_cancelled:
            # tell scheduling to finish subtasks
            await self._scheduling_api.finish_subtasks(
                [result.subtask_id], bands=[band], schedule_next=not error_or_cancelled
            )
            if self.result.status != TaskStatus.terminated:
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
                if not all_done and error_or_cancelled:
                    if result.status == SubtaskStatus.errored:
                        logger.exception(
                            "Subtask %s errored",
                            subtask.subtask_id,
                            exc_info=(
                                type(result.error),
                                result.error,
                                result.traceback,
                            ),
                        )
                    if result.status == SubtaskStatus.cancelled:  # pragma: no cover
                        logger.warning(
                            "Subtask %s from band %s canceled.",
                            subtask.subtask_id,
                            band,
                        )
                    logger.info(
                        "Start to cancel stage %s of task %s.", self.stage_id, self.task
                    )
                    # if error or cancel, cancel all submitted subtasks
                    await self._scheduling_api.cancel_subtasks(
                        list(self._submitted_subtask_ids)
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
        else:
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
                    # all predecessors finished
                    to_schedule_subtasks.append(succ_subtask)
            await self._schedule_subtasks(to_schedule_subtasks)
            await self._scheduling_api.finish_subtasks(
                [result.subtask_id], bands=[band]
            )

    async def run(self):
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
        logger.info("Start to cancel stage %s of task %s.", self.stage_id, self.task)
        if self._done.is_set():  # pragma: no cover
            # already finished, ignore cancel
            return
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
