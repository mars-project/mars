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
import functools
import itertools
import logging

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Set

from .....core import ChunkGraph, Chunk, TileContext
from .....core.context import set_context
from .....core.operand import (
    Fetch,
    Fuse,
    VirtualOperand,
    MapReduceOperand,
    execute,
    ShuffleProxy,
)
from .....lib.aio import alru_cache
from .....resource import Resource
from .....serialization import serialize, deserialize
from .....typing import BandType
from .....utils import (
    calc_data_size,
    lazy_import,
    get_chunk_params,
    get_chunk_key_to_data_keys,
    ensure_coverage,
)
from ....lifecycle.api import LifecycleAPI
from ....meta.api import MetaAPI
from ....subtask import Subtask, SubtaskGraph
from ....subtask.utils import iter_input_data_keys, iter_output_data
from ...core import Task
from ..api import (
    TaskExecutor,
    ExecutionConfig,
    ExecutionChunkResult,
    register_executor_cls,
    _get_n_reducer,
)
from .context import (
    RayExecutionContext,
    RayExecutionWorkerContext,
    RayRemoteObjectManager,
)

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


@dataclass
class _RayChunkMeta:
    memory_size: int


class RayTaskState(RayRemoteObjectManager):
    @classmethod
    def gen_name(cls, task_id: str):
        return f"{cls.__name__}_{task_id}"


_optimize_physical = None


def _optimize_subtask_graph(subtask_graph):
    global _optimize_physical

    if _optimize_physical is None:
        from .....optimization.physical import optimize as _optimize_physical
    return _optimize_physical(subtask_graph)


async def _cancel_ray_task(obj_ref, kill_timeout: int = 3):
    await asyncio.to_thread(ray.cancel, obj_ref, force=False)
    try:
        await asyncio.to_thread(ray.get, obj_ref, timeout=kill_timeout)
    except ray.exceptions.TaskCancelledError:  # pragma: no cover
        logger.info("Cancel ray task %s successfully.", obj_ref)
    except BaseException as e:
        logger.info(
            "Failed to cancel ray task %s with exception %s, "
            "force cancel the task by killing the worker.",
            e,
            obj_ref,
        )
        await asyncio.to_thread(ray.cancel, obj_ref, force=True)


def execute_subtask(
    task_id: str,
    subtask_id: str,
    subtask_chunk_graph: ChunkGraph,
    output_meta_keys: Set[str],
    input_keys: List[str],
    *inputs,
):
    logger.info("Begin to execute subtask: %s", subtask_id)
    ensure_coverage()
    subtask_chunk_graph = deserialize(*subtask_chunk_graph)
    # inputs = [i[1] for i in inputs]
    context = RayExecutionWorkerContext(
        RayTaskState.gen_name(task_id), zip(input_keys, inputs)
    )
    # optimize chunk graph.
    subtask_chunk_graph = _optimize_subtask_graph(subtask_chunk_graph)
    # from data_key to results
    for chunk in subtask_chunk_graph.topological_iter():
        if chunk.key not in context:
            execute(context, chunk.op)

    output = {
        key: data for key, data, _ in iter_output_data(subtask_chunk_graph, context)
    }
    output_values = []
    if output_meta_keys:
        output_meta = {}
        for chunk in subtask_chunk_graph.result_chunks:
            chunk_key = chunk.key
            if chunk_key in output_meta_keys and chunk_key not in output_meta:
                if isinstance(chunk.op, Fuse):
                    # fuse op
                    chunk = chunk.chunk
                data = context[chunk_key]
                memory_size = calc_data_size(data)
                output_meta[chunk_key] = get_chunk_params(chunk), memory_size
        assert len(output_meta_keys) == len(output_meta)
        output_values.append(output_meta)
    output_values.extend(output.values())

    logger.info("Finish executing subtask: %s", subtask_id)
    return output_values[0] if len(output_values) == 1 else output_values


@register_executor_cls
class RayTaskExecutor(TaskExecutor):
    name = "ray"

    def __init__(
        self,
        config: ExecutionConfig,
        task: Task,
        tile_context: TileContext,
        task_context: Dict[str, "ray.ObjectRef"],
        task_chunks_meta: Dict[str, _RayChunkMeta],
        task_state_actor: "ray.actor.ActorHandle",
        lifecycle_api: LifecycleAPI,
        meta_api: MetaAPI,
    ):
        self._config = config
        self._task = task
        self._tile_context = tile_context
        self._task_context = task_context
        self._task_chunks_meta = task_chunks_meta
        self._task_state_actor = task_state_actor
        self._ray_executor = self._get_ray_executor()

        # api
        self._lifecycle_api = lifecycle_api
        self._meta_api = meta_api

        self._available_band_resources = None

        # For progress and task cancel
        self._cur_stage_subtask_graph = None
        self._pre_all_stages_progress = 0.0
        self._pre_all_stages_tile_progress = 0.0
        self._cur_stage_progress = 0.0
        self._cur_stage_tile_progress = 0.0
        self._cur_stage_subtask_to_output_object_refs = dict()
        self._cur_stage_output_object_refs = []
        self._cur_stage_unfinished_subtasks = set()
        self._execute_subtask_graph_aiotask = None
        self._done = asyncio.Event()
        self._cancelled = False

        # For task context gc
        self._subtask_running_monitor = None
        self._chunk_ref_counts = defaultdict(lambda: 0)

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
        lifecycle_api, meta_api = await cls._get_apis(session_id, address)
        task_state_actor = (
            ray.remote(RayTaskState)
            .options(name=RayTaskState.gen_name(task.task_id))
            .remote()
        )
        task_context = {}
        task_chunks_meta = {}
        await cls._init_context(
            task_context, task_chunks_meta, task_state_actor, session_id, address
        )
        return cls(
            config,
            task,
            tile_context,
            task_context,
            task_chunks_meta,
            task_state_actor,
            lifecycle_api,
            meta_api,
        )

    # noinspection DuplicatedCode
    def destroy(self):
        self._config = None
        self._task = None
        self._tile_context = None
        self._task_context = None
        self._task_chunks_meta = None
        self._task_state_actor = None
        self._ray_executor = None

        # api
        self._lifecycle_api = None
        self._meta_api = None

        self._available_band_resources = None

        # For progress and task cancel
        self._cur_stage_subtask_graph = None
        self._pre_all_stages_progress = None
        self._pre_all_stages_tile_progress = None
        self._cur_stage_progress = 1.0
        self._cur_stage_tile_progress = None
        self._cur_stage_subtask_to_output_object_refs = None
        self._cur_stage_output_object_refs = None
        self._cur_stage_unfinished_subtasks = None
        self._execute_subtask_graph_aiotask = None
        self._done = None
        self._cancelled = None

        # For task context gc
        self._subtask_running_monitor = None
        self._chunk_ref_counts = None

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def _get_apis(cls, session_id: str, address: str):
        return await asyncio.gather(
            LifecycleAPI.create(session_id, address),
            MetaAPI.create(session_id, address),
        )

    @staticmethod
    @functools.lru_cache(maxsize=None)  # Specify maxsize=None to make it faster
    def _get_ray_executor():
        # Export remote function once.
        return ray.remote(execute_subtask)

    @classmethod
    async def _init_context(
        cls,
        task_context: Dict[str, "ray.ObjectRef"],
        task_chunks_meta: Dict[str, _RayChunkMeta],
        task_state_actor: "ray.actor.ActorHandle",
        session_id: str,
        address: str,
    ):
        loop = asyncio.get_running_loop()
        context = RayExecutionContext(
            task_context,
            task_chunks_meta,
            task_state_actor,
            session_id,
            address,
            address,
            address,
            loop=loop,
        )
        await context.init()
        set_context(context)

    async def execute_subtask_graph(
        self,
        stage_id: str,
        subtask_graph: SubtaskGraph,
        chunk_graph: ChunkGraph,
        tile_context: TileContext,
        context: Any = None,
    ) -> Dict[Chunk, ExecutionChunkResult]:
        if self._cancelled is True:  # pragma: no cover
            raise asyncio.CancelledError()
        self._execute_subtask_graph_aiotask = asyncio.current_task()

        logger.info("Stage %s start.", stage_id)
        await self._incref_stage(stage_id, subtask_graph, chunk_graph)
        self._cur_stage_subtask_graph = subtask_graph
        task_context = self._task_context
        output_meta_object_refs = []
        self._pre_all_stages_tile_progress = (
            self._pre_all_stages_tile_progress + self._cur_stage_tile_progress
        )
        self._cur_stage_tile_progress = (
            self._tile_context.get_all_progress() - self._pre_all_stages_tile_progress
        )
        logger.info("Submitting %s subtasks of stage %s.", len(subtask_graph), stage_id)
        result_meta_keys = {
            chunk.key
            for chunk in chunk_graph.result_chunks
            if not isinstance(chunk.op, Fetch)
        }
        subtask_ids = []
        for subtask in subtask_graph.topological_iter():
            subtask_ids.append(subtask.subtask_id)
            subtask_chunk_graph = subtask.chunk_graph
            key_to_input = await self._load_subtask_inputs(
                stage_id, subtask, subtask_chunk_graph, task_context
            )
            output_keys = self._get_subtask_output_keys(subtask_chunk_graph)
            output_meta_keys = result_meta_keys & output_keys
            output_count = len(output_keys) + bool(output_meta_keys)
            subtask_max_retries = (
                self._config.subtask_max_retries if subtask.retryable else 0
            )
            output_object_refs = self._ray_executor.options(
                num_returns=output_count, max_retries=subtask_max_retries
            ).remote(
                subtask.task_id,
                subtask.subtask_id,
                serialize(subtask_chunk_graph),
                output_meta_keys,
                list(key_to_input.keys()),
                *key_to_input.values(),
            )
            self._cur_stage_unfinished_subtasks.add(subtask)
            if output_count == 0:
                continue
            elif output_count == 1:
                output_object_refs = [output_object_refs]
            self._cur_stage_subtask_to_output_object_refs[subtask] = set(
                output_object_refs
            )
            self._cur_stage_output_object_refs.extend(output_object_refs)
            if output_meta_keys:
                meta_object_ref, *output_object_refs = output_object_refs
                # TODO(fyrestone): Fetch(not get) meta object here.
                output_meta_object_refs.append(meta_object_ref)
            task_context.update(zip(output_keys, output_object_refs))
        logger.info("Submitted %s subtasks of stage %s.", len(subtask_graph), stage_id)

        self._subtask_running_monitor = asyncio.create_task(
            self._check_subtask_results_periodically(
                self._config.subtask_check_interval
            )
        )

        key_to_meta = {}
        if len(output_meta_object_refs) > 0:
            # TODO(fyrestone): Optimize update meta by fetching partial meta.
            meta_count = len(output_meta_object_refs)
            logger.info("Getting %s metas of stage %s.", meta_count, stage_id)
            try:
                meta_list = await asyncio.gather(*output_meta_object_refs)
            except BaseException:
                self._cancelled = True
                self._subtask_running_monitor.cancel()
                raise
            for meta in meta_list:
                for key, (params, memory_size) in meta.items():
                    key_to_meta[key] = params
                    self._task_chunks_meta[key] = _RayChunkMeta(memory_size=memory_size)
            assert len(key_to_meta) == len(result_meta_keys)
            logger.info("Got %s metas of stage %s.", meta_count, stage_id)

        chunk_to_meta = {}
        # ray.wait requires the object ref list is unique.
        output_object_refs = set()
        for chunk in chunk_graph.result_chunks:
            chunk_key = chunk.key
            object_ref = task_context[chunk_key]
            output_object_refs.add(object_ref)
            chunk_params = key_to_meta.get(chunk_key)
            if chunk_params is not None:
                chunk_to_meta[chunk] = ExecutionChunkResult(chunk_params, object_ref)

        logger.info("Waiting for stage %s complete.", stage_id)
        # Patched the asyncio.to_thread for Python < 3.9 at mars/lib/aio/__init__.py
        await asyncio.to_thread(ray.wait, list(output_object_refs), fetch_local=False)

        await self._done.wait()

        # Just use `self._cur_stage_tile_progress` as current stage progress
        # because current stage is finished, its progress is 1.
        self._pre_all_stages_progress += self._cur_stage_tile_progress
        self._cur_stage_subtask_to_output_object_refs.clear()
        self._cur_stage_unfinished_subtasks.clear()
        self._cur_stage_output_object_refs.clear()
        logger.info("Stage %s is complete.", stage_id)
        return chunk_to_meta

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return

        # Update info if no exception occurs.
        tileable_keys = []
        update_metas = []
        update_lifecycles = []
        for tileable in self._task.tileable_graph.result_tileables:
            tileable_keys.append(tileable.key)
            tileable = tileable.data if hasattr(tileable, "data") else tileable
            chunk_keys = []
            for chunk in self._tile_context[tileable].chunks:
                chunk_key = chunk.key
                chunk_keys.append(chunk_key)
                if chunk_key in self._task_context:
                    # Some tileable graph may have result chunks that not be executed,
                    # for example:
                    # r, b = cut(series, bins, retbins=True)
                    #     r_result = r.execute().fetch()
                    #     b_result = b.execute().fetch() <- This is the case
                    object_ref = self._task_context[chunk_key]
                    chunk_meta = self._task_chunks_meta[chunk_key]
                    update_metas.append(
                        self._meta_api.set_chunk_meta.delay(
                            chunk,
                            bands=[],
                            object_ref=object_ref,
                            memory_size=chunk_meta.memory_size,
                        )
                    )
                update_lifecycles.append(
                    self._lifecycle_api.track.delay(tileable.key, chunk_keys)
                )
        await self._meta_api.set_chunk_meta.batch(*update_metas)
        await self._lifecycle_api.track.batch(*update_lifecycles)
        await self._lifecycle_api.incref_tileables(tileable_keys)

    async def get_available_band_resources(self) -> Dict[BandType, Resource]:
        if self._available_band_resources is None:
            band_resources = self._config.get_band_resources()
            virtual_band_resources = {}
            idx = 0
            for band_resource in band_resources:
                for band, resource in band_resource.items():
                    virtual_band_resources[(f"ray_virtual://{idx}", band)] = resource
                    idx += 1
            self._available_band_resources = virtual_band_resources

        return self._available_band_resources

    async def get_progress(self) -> float:
        """Get the execution progress."""
        return self._cur_stage_progress

    async def cancel(self):
        """
        Cancel the task execution.

        1. Try to cancel the `execute_subtask_graph`
        2. Try to cancel the submitted subtasks by `ray.cancel`
        """
        logger.info("Start to cancel task %s.", self._task)
        if self._done.is_set():
            return
        self._cancelled = True
        if (
            self._execute_subtask_graph_aiotask is not None
            and not self._execute_subtask_graph_aiotask.cancelled()
        ):
            self._execute_subtask_graph_aiotask.cancel()
        if self._subtask_running_monitor is not None:
            self._subtask_running_monitor.cancel()
        timeout = self._config.subtask_cancel_timeout
        to_be_cancelled_coros = []
        for (
            _,
            output_object_refs,
        ) in self._cur_stage_subtask_to_output_object_refs.items():
            if output_object_refs:
                to_be_cancelled_coros.append(
                    _cancel_ray_task(next(iter(output_object_refs)), timeout)
                )
        await asyncio.gather(*to_be_cancelled_coros)

    async def _load_subtask_inputs(
        self, stage_id: str, subtask: Subtask, chunk_graph: ChunkGraph, context: Dict
    ):
        """
        Load a dict of input key to object ref of subtask from context.

        It updates the context if the input object refs are fetched from
        the meta service.
        """
        key_to_input = {}
        key_to_get_meta = {}
        chunk_key_to_data_keys = get_chunk_key_to_data_keys(chunk_graph)
        for key, _ in iter_input_data_keys(
            subtask, chunk_graph, chunk_key_to_data_keys
        ):
            if key in context:
                key_to_input[key] = context[key]
            else:
                key_to_get_meta[key] = self._meta_api.get_chunk_meta.delay(
                    key, fields=["object_refs"]
                )
        if key_to_get_meta:
            logger.info(
                "Fetch %s metas and update context of stage %s.",
                len(key_to_get_meta),
                stage_id,
            )
            meta_list = await self._meta_api.get_chunk_meta.batch(
                *key_to_get_meta.values()
            )
            for key, meta in zip(key_to_get_meta.keys(), meta_list):
                object_ref = meta["object_refs"][0]
                key_to_input[key] = object_ref
                context[key] = object_ref
        return key_to_input

    @staticmethod
    def _get_subtask_output_keys(chunk_graph: ChunkGraph):
        output_keys = {}
        for chunk in chunk_graph.results:
            if isinstance(chunk.op, VirtualOperand):
                continue
            elif isinstance(chunk.op, MapReduceOperand):
                # TODO(fyrestone): Handle shuffle operands.
                raise NotImplementedError(
                    "The shuffle operands are not supported by the ray executor."
                )
            else:
                output_keys[chunk.key] = 1
        return output_keys.keys()

    async def _get_ready_objects(self):
        total = len(self._cur_stage_output_object_refs)
        if total > 0:
            ready_objects, _ = ray.wait(
                self._cur_stage_output_object_refs,
                num_returns=total,
                timeout=0,  # Avoid blocking the asyncio loop.
                fetch_local=False,
            )
            return ready_objects

    async def _calculate_progress(self, ready_objects):
        stage_progress = 0.0
        if ready_objects is not None:
            stage_progress = (
                len(ready_objects)
                / len(self._cur_stage_output_object_refs)
                * self._cur_stage_tile_progress
            )
        self._cur_stage_progress = self._pre_all_stages_progress + stage_progress

    async def _get_newly_finished_subtasks(self, ready_objects: Set["ray.ObjectRef"]):
        finished_subtasks = set()
        if ready_objects is not None:
            for subtask in self._cur_stage_unfinished_subtasks:
                if self._cur_stage_subtask_to_output_object_refs[subtask].issubset(
                    ready_objects
                ):
                    finished_subtasks.add(subtask)
            self._cur_stage_unfinished_subtasks -= finished_subtasks
        return finished_subtasks

    async def _check_subtask_results_periodically(self, time_interval: float):
        while True:
            if self._cancelled:
                return
            ready_objects = await self._get_ready_objects()
            await self._calculate_progress(ready_objects)
            finished_subtasks = await self._get_newly_finished_subtasks(
                set(ready_objects)
            )
            coros = [
                self._decref_input_subtasks(subtask) for subtask in finished_subtasks
            ]
            await asyncio.gather(*coros)
            if not self._cur_stage_unfinished_subtasks:
                self._done.set()
            await asyncio.sleep(time_interval)

    @classmethod
    def _check_ref_counts(cls, keys: List[str], ref_counts: List[int]):
        if ref_counts is not None and len(keys) != len(ref_counts):
            raise ValueError(
                f"`ref_counts` should have same size as `keys`, expect {len(keys)}, got {len(ref_counts)}"
            )

    def __incref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        counts = counts if counts is not None else itertools.repeat(1)
        for chunk_key, count in zip(chunk_keys, counts):
            self._chunk_ref_counts[chunk_key] += count

    async def _incref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        self._check_ref_counts(chunk_keys, counts)
        return await asyncio.to_thread(self.__incref_chunks, chunk_keys, counts=counts)

    def _get_removed_chunk_keys(self, chunk_keys: List[str], counts: List[int] = None):
        to_remove_chunk_keys = []
        counts = counts if counts is not None else itertools.repeat(1)
        for chunk_key, count in zip(chunk_keys, counts):
            ref_count = self._chunk_ref_counts[chunk_key]
            ref_count -= count
            assert ref_count >= 0, f"chunk key {chunk_key} will have negative ref count"
            self._chunk_ref_counts[chunk_key] = ref_count
            if ref_count == 0:
                to_remove_chunk_keys.append(chunk_key)
        return to_remove_chunk_keys

    def _remove_chunks(self, to_remove_chunk_keys: List[str]):
        if not to_remove_chunk_keys:
            return
        for chunk_key in to_remove_chunk_keys:
            self._task_context.pop(chunk_key, None)
            self._chunk_ref_counts.pop(chunk_key, None)

    async def _decref_chunks(self, chunk_keys: List[str], counts: List[int] = None):
        self._check_ref_counts(chunk_keys, counts)
        to_remove_chunk_keys = await asyncio.to_thread(
            self._get_removed_chunk_keys, chunk_keys, counts=counts
        )
        self._remove_chunks(to_remove_chunk_keys)

    async def _incref_stage(
        self, stage_id: str, subtask_graph: SubtaskGraph, chunk_graph: ChunkGraph
    ):
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
        for c in chunk_graph.result_chunks:
            incref_chunk_key_to_counts[c.key] += 1
        logger.debug(
            "Incref chunks for stage %s: %s",
            stage_id,
            incref_chunk_key_to_counts,
        )
        await self._incref_chunks(
            list(incref_chunk_key_to_counts),
            counts=list(incref_chunk_key_to_counts.values()),
        )

    async def _decref_input_subtasks(self, subtask: Subtask):
        decref_chunk_key_to_counts = defaultdict(lambda: 0)
        for in_subtask in self._cur_stage_subtask_graph.iter_predecessors(subtask):
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
        await self._decref_chunks(
            list(decref_chunk_key_to_counts),
            counts=list(decref_chunk_key_to_counts.values()),
        )
