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
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..... import oscar as mo
from .....core import ChunkGraph, TileContext
from .....core.context import set_context
from .....core.operand import (
    Fetch,
    MapReduceOperand,
    OperandStage,
    ShuffleProxy,
)
from .....lib.aio import alru_cache
from .....oscar.profiling import (
    ProfilingData,
)
from .....resource import Resource
from .....typing import TileableType, BandType
from .....utils import Timer
from ....context import ThreadedServiceContext
from ....cluster.api import ClusterAPI
from ....lifecycle.api import LifecycleAPI
from ....meta.api import MetaAPI
from ....scheduling import SchedulingAPI
from ....subtask import Subtask, SubtaskResult, SubtaskStatus, SubtaskGraph
from ...core import Task
from ..api import ExecutionConfig, TaskExecutor, register_executor_cls
from .resource import ResourceEvaluator
from .stage import TaskStageProcessor

logger = logging.getLogger(__name__)


def _get_n_reducer(subtask: Subtask) -> int:
    return len(
        [
            r
            for r in subtask.chunk_graph
            if isinstance(r.op, MapReduceOperand) and r.op.stage == OperandStage.reduce
        ]
    )


@register_executor_cls
class MarsTaskExecutor(TaskExecutor):
    name = "mars"
    _stage_processors: List[TaskStageProcessor]
    _stage_tile_progresses: List[float]
    _cur_stage_processor: Optional[TaskStageProcessor]
    _meta_updated_tileables: Set[TileableType]

    def __init__(
        self,
        config: ExecutionConfig,
        task: Task,
        tile_context: TileContext,
        cluster_api: ClusterAPI,
        lifecycle_api: LifecycleAPI,
        scheduling_api: SchedulingAPI,
        meta_api: MetaAPI,
        resource_evaluator: ResourceEvaluator,
    ):
        self._config = config
        self._task = task
        self._tileable_graph = task.tileable_graph
        self._raw_tile_context = tile_context.copy()
        self._tile_context = tile_context
        self._session_id = task.session_id

        # api
        self._cluster_api = cluster_api
        self._lifecycle_api = lifecycle_api
        self._scheduling_api = scheduling_api
        self._meta_api = meta_api

        self._stage_processors = []
        self._stage_tile_progresses = []
        self._cur_stage_processor = None
        self._lifecycle_processed_tileables = set()
        self._subtask_decref_events = dict()
        self._meta_updated_tileables = set()

        # Evaluate and initialize subtasks required resource.
        self._resource_evaluator = resource_evaluator

    @classmethod
    async def create(
        cls,
        config: ExecutionConfig,
        *,
        session_id: str,
        address: str,
        task: Task,
        tile_context: TileContext,
        **kwargs,
    ) -> "TaskExecutor":
        assert (
            len(kwargs) == 0
        ), f"Unexpected kwargs for {cls.__name__}.create: {kwargs}"
        cluster_api, lifecycle_api, scheduling_api, meta_api = await cls._get_apis(
            session_id, address
        )
        resource_evaluator = await ResourceEvaluator.create(
            config.get_execution_config(),
            session_id=task.session_id,
            task_id=task.task_id,
            cluster_api=cluster_api,
        )
        await cls._init_context(session_id, address)
        return cls(
            config,
            task,
            tile_context,
            cluster_api,
            lifecycle_api,
            scheduling_api,
            meta_api,
            resource_evaluator,
        )

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def _get_apis(cls, session_id: str, address: str):
        return await asyncio.gather(
            ClusterAPI.create(address),
            LifecycleAPI.create(session_id, address),
            SchedulingAPI.create(session_id, address),
            MetaAPI.create(session_id, address),
        )

    @classmethod
    async def _init_context(cls, session_id: str, address: str):
        loop = asyncio.get_running_loop()
        context = ThreadedServiceContext(
            session_id, address, address, address, loop=loop
        )
        await context.init()
        set_context(context)

    async def __aenter__(self):
        profiling = ProfilingData[self._task.task_id, "general"]
        # incref fetch tileables to ensure fetch data not deleted
        with Timer() as timer:
            await self._incref_fetch_tileables()
        profiling.set("incref_fetch_tileables", timer.duration)

    async def execute_subtask_graph(
        self,
        stage_id: str,
        subtask_graph: SubtaskGraph,
        chunk_graph: ChunkGraph,
        tile_context: TileContext,
        context=None,
    ):
        available_bands = await self.get_available_band_resources()
        await self._incref_result_tileables()
        stage_processor = TaskStageProcessor(
            stage_id,
            self._task,
            chunk_graph,
            subtask_graph,
            list(available_bands),
            tile_context,
            self._scheduling_api,
            self._meta_api,
        )
        await self._incref_stage(stage_processor)
        await self._resource_evaluator.evaluate(stage_processor)
        self._stage_processors.append(stage_processor)
        self._cur_stage_processor = stage_processor
        # get the tiled progress for current stage
        prev_progress = sum(self._stage_tile_progresses)
        curr_tile_progress = self._tile_context.get_all_progress() - prev_progress
        self._stage_tile_progresses.append(curr_tile_progress)
        return await stage_processor.run()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # clean ups
        decrefs = []
        error_or_cancelled = False
        for stage_processor in self._stage_processors:
            if stage_processor.error_or_cancelled():
                error_or_cancelled = True
            decrefs.append(self._decref_stage.delay(stage_processor))
        await self._decref_stage.batch(*decrefs)
        # revert fetch incref
        await self._decref_fetch_tileables()
        if error_or_cancelled:
            # revert result incref if error or cancelled
            await self._decref_result_tileables()
        await self._resource_evaluator.report()

    async def get_available_band_resources(self) -> Dict[BandType, Resource]:
        async for bands in self._cluster_api.watch_all_bands():
            if bands:
                return bands

    async def get_progress(self) -> float:
        # get progress of stages
        executor_progress = 0.0
        assert len(self._stage_tile_progresses) == len(self._stage_processors)
        for stage_processor, stage_tile_progress in zip(
            self._stage_processors, self._stage_tile_progresses
        ):
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
            subtask_progress = progress / n_subtask
            executor_progress += subtask_progress * stage_tile_progress
        return executor_progress

    async def cancel(self):
        if self._cur_stage_processor is not None:
            await self._cur_stage_processor.cancel()

    async def set_subtask_result(self, subtask_result: SubtaskResult):
        if self._cur_stage_processor is None or (
            subtask_result.stage_id
            and self._cur_stage_processor.stage_id != subtask_result.stage_id
        ):
            logger.warning(
                "Stage %s for subtask %s not exists, got stale subtask result %s which may be "
                "speculative execution from previous stages, just ignore it.",
                subtask_result.stage_id,
                subtask_result.subtask_id,
                subtask_result,
            )
            return
        stage_processor = self._cur_stage_processor
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
                    await self._decref_input_subtasks(
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

    def get_stage_processors(self):
        return self._stage_processors

    async def _incref_fetch_tileables(self):
        # incref fetch tileables in tileable graph to prevent them from deleting
        to_incref_tileable_keys = [
            tileable.op.source_key
            for tileable in self._tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context
        ]
        await self._lifecycle_api.incref_tileables(to_incref_tileable_keys)

    async def _decref_fetch_tileables(self):
        fetch_tileable_keys = [
            tileable.op.source_key
            for tileable in self._tileable_graph
            if isinstance(tileable.op, Fetch) and tileable in self._raw_tile_context
        ]
        await self._lifecycle_api.decref_tileables(fetch_tileable_keys)

    def _get_tiled(self, tileable: TileableType):
        tileable = tileable.data if hasattr(tileable, "data") else tileable
        return self._tile_context[tileable]

    async def _incref_result_tileables(self):
        processed = self._lifecycle_processed_tileables
        # track and incref result tileables if tiled
        tracks = [], []
        for result_tileable in self._tileable_graph.result_tileables:
            if result_tileable in processed:  # pragma: no cover
                continue
            try:
                tiled_tileable = self._get_tiled(result_tileable)
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

    async def _decref_result_tileables(self):
        await self._lifecycle_api.decref_tileables(
            [t.key for t in self._lifecycle_processed_tileables]
        )

    async def _incref_stage(self, stage_processor: "TaskStageProcessor"):
        subtask_graph = stage_processor.subtask_graph
        incref_chunk_key_to_counts = defaultdict(lambda: 0)
        for subtask in subtask_graph:
            # for subtask has successors, incref number of successors
            n = subtask_graph.count_successors(subtask)
            for c in subtask.chunk_graph.results:
                incref_chunk_key_to_counts[c.key] += n
            # process reducer, incref mapper chunks
            for pre_graph in subtask_graph.iter_predecessors(subtask):
                for chk in pre_graph.chunk_graph.results:
                    if isinstance(chk.op, ShuffleProxy):
                        n_reducer = _get_n_reducer(subtask)
                        for map_chunk in chk.inputs:
                            incref_chunk_key_to_counts[map_chunk.key] += n_reducer
        result_chunks = stage_processor.chunk_graph.result_chunks
        for c in result_chunks:
            incref_chunk_key_to_counts[c.key] += 1
        logger.debug(
            "Incref chunks for stage %s: %s",
            stage_processor.stage_id,
            incref_chunk_key_to_counts,
        )
        await self._lifecycle_api.incref_chunks(
            list(incref_chunk_key_to_counts),
            counts=list(incref_chunk_key_to_counts.values()),
        )

    @classmethod
    def _get_decref_stage_chunk_key_to_counts(
        cls, stage_processor: "TaskStageProcessor"
    ) -> Dict[str, int]:
        decref_chunk_key_to_counts = defaultdict(lambda: 0)
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
                        for c in inp_subtask.chunk_graph.results:
                            decref_chunk_key_to_counts[c.key] += 1
        # decref result of chunk graphs
        for c in stage_processor.chunk_graph.results:
            decref_chunk_key_to_counts[c.key] += 1
        return decref_chunk_key_to_counts

    @mo.extensible
    async def _decref_stage(self, stage_processor: "TaskStageProcessor"):
        decref_chunk_key_to_counts = self._get_decref_stage_chunk_key_to_counts(
            stage_processor
        )
        logger.debug(
            "Decref chunks when stage %s finish: %s",
            stage_processor.stage_id,
            decref_chunk_key_to_counts,
        )
        await self._lifecycle_api.decref_chunks(
            list(decref_chunk_key_to_counts),
            counts=list(decref_chunk_key_to_counts.values()),
        )

    @_decref_stage.batch
    async def _decref_stage(self, args_list, kwargs_list):
        decref_chunk_key_to_counts = defaultdict(lambda: 0)
        for args, kwargs in zip(args_list, kwargs_list):
            chunk_key_to_counts = self._get_decref_stage_chunk_key_to_counts(
                *args, **kwargs
            )
            for k, c in chunk_key_to_counts.items():
                decref_chunk_key_to_counts[k] += c
        logger.debug("Decref chunks when stages finish: %s", decref_chunk_key_to_counts)
        await self._lifecycle_api.decref_chunks(
            list(decref_chunk_key_to_counts),
            counts=list(decref_chunk_key_to_counts.values()),
        )

    async def _decref_input_subtasks(
        self, subtask: Subtask, subtask_graph: SubtaskGraph
    ):
        # make sure subtasks are decreffed only once
        if subtask.subtask_id not in self._subtask_decref_events:
            self._subtask_decref_events[subtask.subtask_id] = asyncio.Event()
        else:  # pragma: no cover
            await self._subtask_decref_events[subtask.subtask_id].wait()
            return

        decref_chunk_key_to_counts = defaultdict(lambda: 0)
        for in_subtask in subtask_graph.iter_predecessors(subtask):
            for result_chunk in in_subtask.chunk_graph.results:
                # for reducer chunk, decref mapper chunks
                if isinstance(result_chunk.op, ShuffleProxy):
                    n_reducer = _get_n_reducer(subtask)
                    for inp in result_chunk.inputs:
                        decref_chunk_key_to_counts[inp.key] += n_reducer
                decref_chunk_key_to_counts[result_chunk.key] += 1
        logger.debug(
            "Decref chunks %s when subtask %s finish",
            decref_chunk_key_to_counts,
            subtask.subtask_id,
        )
        await self._lifecycle_api.decref_chunks(
            list(decref_chunk_key_to_counts),
            counts=list(decref_chunk_key_to_counts.values()),
        )

        # `set_subtask_result` will be called when subtask finished
        # but report progress will call set_subtask_result too,
        # so it have risk to duplicate decrease some subtask input object reference,
        # it will cause object reference count lower zero
        # TODO(Catch-Bull): Pop asyncio.Event when current subtask `set_subtask_result`
        # will never be called
        self._subtask_decref_events[subtask.subtask_id].set()
