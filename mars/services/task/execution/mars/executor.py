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
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..... import oscar as mo
from .....core import ChunkGraph
from .....core.operand import (
    Fetch,
    MapReduceOperand,
    OperandStage,
    ShuffleProxy,
)
from .....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from .....lib.aio import alru_cache
from .....oscar.profiling import (
    ProfilingData,
)
from .....resource import Resource
from .....typing import TileableType, BandType, ChunkType
from .....utils import Timer, get_chunk_params
from .....tensor.core import TENSOR_TYPE
from ....cluster.api import ClusterAPI
from ....lifecycle.api import LifecycleAPI
from ....meta.api import MetaAPI, WorkerMetaAPI
from ....scheduling import SchedulingAPI
from ....subtask import Subtask, SubtaskResult, SubtaskStatus, SubtaskGraph
from ..api import TaskExecutor, ExecutionChunkResult, register_executor_cls
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
    _cur_stage_processor: Optional[TaskStageProcessor]
    _meta_updated_tileables: Set[TileableType]

    def __init__(
        self,
        config,
        task,
        tile_context,
        cluster_api,
        lifecycle_api,
        scheduling_api,
        meta_api,
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
        self._cur_stage_processor = None
        self._lifecycle_processed_tileables = set()
        self._subtask_decref_events = dict()
        self._meta_updated_tileables = set()

    @classmethod
    async def create(
        cls,
        config: Dict,
        *,
        session_id: str,
        address: str,
        task,
        tile_context,
        **kwargs,
    ) -> "TaskExecutor":
        assert (
            len(kwargs) == 0
        ), f"Unexpected kwargs for {cls.__name__}.create: {kwargs}"
        cluster_api, lifecycle_api, scheduling_api, meta_api = await cls._get_apis(
            session_id, address
        )
        return cls(
            config,
            task,
            tile_context,
            cluster_api,
            lifecycle_api,
            scheduling_api,
            meta_api,
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
        tile_context: Dict[TileableType, TileableType],
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
            self._scheduling_api,
            self._meta_api,
        )
        await self._incref_stage(stage_processor)
        # Evaluate and initialize subtasks required resource.
        resource_evaluator = ResourceEvaluator(stage_processor)
        resource_evaluator.evaluate()
        self._stage_processors.append(stage_processor)
        self._cur_stage_processor = stage_processor
        execution_chunk_results = await stage_processor.run()
        await self._update_result_meta(
            chunk_graph.result_chunks, execution_chunk_results, tile_context
        )
        return execution_chunk_results

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

    async def get_available_band_resources(self) -> Dict[BandType, Resource]:
        async for bands in self._cluster_api.watch_all_bands():
            if bands:
                return bands

    async def get_progress(self) -> float:
        # get progress of stages
        subtask_progress = 0.0
        n_stage = 0

        for stage_processor in self._stage_processors:
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

        return subtask_progress

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
        incref_chunk_keys = []
        for subtask in subtask_graph:
            # for subtask has successors, incref number of successors
            n = subtask_graph.count_successors(subtask)
            for c in subtask.chunk_graph.results:
                incref_chunk_keys.extend([c.key] * n)
            # process reducer, incref mapper chunks
            for pre_graph in subtask_graph.iter_predecessors(subtask):
                for chk in pre_graph.chunk_graph.results:
                    if isinstance(chk.op, ShuffleProxy):
                        n_reducer = _get_n_reducer(subtask)
                        incref_chunk_keys.extend(
                            [map_chunk.key for map_chunk in chk.inputs] * n_reducer
                        )
        result_chunks = stage_processor.chunk_graph.result_chunks
        incref_chunk_keys.extend([c.key for c in result_chunks])
        logger.debug(
            "Incref chunks for stage %s: %s",
            stage_processor.stage_id,
            incref_chunk_keys,
        )
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
                        decref_chunk_keys.extend(
                            [c.key for c in inp_subtask.chunk_graph.results]
                        )
            # decref result of chunk graphs
            decref_chunk_keys.extend(
                [c.key for c in stage_processor.chunk_graph.results]
            )
        return decref_chunk_keys

    @mo.extensible
    async def _decref_stage(self, stage_processor: "TaskStageProcessor"):
        decref_chunk_keys = self._get_decref_stage_chunk_keys(stage_processor)
        logger.debug(
            "Decref chunks when stage %s finish: %s",
            stage_processor.stage_id,
            decref_chunk_keys,
        )
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

    @_decref_stage.batch
    async def _decref_stage(self, args_list, kwargs_list):
        decref_chunk_keys = []
        for args, kwargs in zip(args_list, kwargs_list):
            decref_chunk_keys.extend(self._get_decref_stage_chunk_keys(*args, **kwargs))
        logger.debug("Decref chunks when stages finish: %s", decref_chunk_keys)
        await self._lifecycle_api.decref_chunks(decref_chunk_keys)

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
                    n_reducer = _get_n_reducer(subtask)
                    decref_chunk_keys.extend(
                        [inp.key for inp in result_chunk.inputs] * n_reducer
                    )
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

    async def _update_result_meta(
        self,
        chunk_graph_results: List[ChunkType],
        execution_chunk_results: List[ExecutionChunkResult],
        tile_context: Dict[TileableType, TileableType],
    ):
        chunk_to_chunk_result = dict(zip(chunk_graph_results, execution_chunk_results))
        update_meta_chunks = chunk_to_chunk_result.keys() - set(
            itertools.chain(
                (c.data for c in tiled_tileable.chunks)
                for tiled_tileable in tile_context.values()
            )
        )

        chunk_results = []
        worker_meta_api_to_chunk_delays = defaultdict(list)
        for c in update_meta_chunks:
            r = chunk_to_chunk_result[c]
            address = r.meta["bands"][0][0]
            meta_api = await WorkerMetaAPI.create(self._session_id, address)
            call = meta_api.get_chunk_meta.delay(
                c.key, fields=list(get_chunk_params(c).keys())
            )
            chunk_results.append(r)
            worker_meta_api_to_chunk_delays[meta_api].append(call)
        for tileable in tile_context.values():
            chunks = [c.data for c in tileable.chunks]
            for c, params_fields in zip(chunks, self._get_params_fields(tileable)):
                r = chunk_to_chunk_result[c]
                address = r.meta["bands"][0][0]
                meta_api = await WorkerMetaAPI.create(self._session_id, address)
                call = meta_api.get_chunk_meta.delay(c.key, fields=params_fields)
                chunk_results.append(r)
                worker_meta_api_to_chunk_delays[meta_api].append(call)
        coros = []
        for worker_meta_api, chunk_delays in worker_meta_api_to_chunk_delays.items():
            coros.append(worker_meta_api.get_chunk_meta.batch(*chunk_delays))
        worker_metas = await asyncio.gather(*coros)
        for r, meta in zip(chunk_results, itertools.chain(*worker_metas)):
            r.meta = meta

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
