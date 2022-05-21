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

from dataclasses import dataclass
from typing import Dict, Any, Set, Optional

from .....core import ChunkGraph, Chunk, TileContext
from .....core.context import set_context
from .....core.operand import (
    Fetch,
    Fuse,
    VirtualOperand,
    MapReduceOperand,
    execute,
    OperandStage,
)
from .....core.operand.fetch import PushShuffle, FetchShuffle
from .....lib.aio import alru_cache
from .....resource import Resource
from .....serialization import serialize, deserialize
from .....typing import BandType
from .....utils import (
    calc_data_size,
    lazy_import,
    get_chunk_params,
    ensure_coverage,
    log_exception_wrapper,
)
from ....lifecycle.api import LifecycleAPI
from ....meta.api import MetaAPI
from ....ray_utils import _ray_export_once
from ....subtask import Subtask, SubtaskGraph
from ....subtask.utils import iter_output_data
from ...core import Task
from ..api import (
    TaskExecutor,
    ExecutionConfig,
    ExecutionChunkResult,
    register_executor_cls,
)
from .context import (
    RayExecutionContext,
    RayExecutionWorkerContext,
    RayRemoteObjectManager,
)
from .shuffle import ShuffleManager

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


def execute_subtask(
    task_id: str,
    subtask_id: str,
    subtask_chunk_graph: ChunkGraph,
    output_meta_keys: Set[str],
    *inputs,
):
    """The function used for execute subtask in ray task.
    If subtask is shuffle mapper, no chunk meta will be returner, otherwise return chunk mata."""
    ensure_coverage()
    subtask_chunk_graph = deserialize(*subtask_chunk_graph)
    subtask_digraph = subtask_chunk_graph.to_dot()
    logger.info(
        "Begin to execute subtask %s with graph %s.", subtask_id, subtask_digraph
    )
    # optimize chunk graph.
    subtask_chunk_graph = _optimize_subtask_graph(subtask_chunk_graph)
    start_chunks = list(subtask_chunk_graph.iter_indep())
    maybe_mapper_chunk = subtask_chunk_graph.result_chunks[0]
    is_mapper = (
        isinstance(maybe_mapper_chunk.op, MapReduceOperand)
        and maybe_mapper_chunk.op.stage == OperandStage.map
    )
    if isinstance(start_chunks[0].op, PushShuffle):
        assert len(start_chunks) == 1, start_chunks
        # the subtask is a reducer subtask
        n_mapper = len(inputs)
        # some reducer may have multiple output chunks, see `PSRSshuffle._execute_reduce` and
        # https://user-images.githubusercontent.com/12445254/168569524-f09e42a7-653a-4102-bdf0-cc1631b3168d.png
        reducer_chunks = subtask_chunk_graph.successors(start_chunks[0])
        reducer_operands = set(c.op for c in reducer_chunks)
        assert len(reducer_operands) == 1, (
            reducer_operands,
            reducer_chunks,
            subtask_digraph,
        )
        reducer_operand = reducer_chunks[0].op
        reducer_index = reducer_operand.reducer_index
        # mock input keys, keep this in sync with `MapReducerOperand#_iter_mapper_key_idx_pairs`
        input_keys = [(i, reducer_index) for i in range(n_mapper)]
    else:
        input_keys = [c.key for c in start_chunks if isinstance(c.op, Fetch)]
    context = RayExecutionWorkerContext(
        RayTaskState.gen_name(task_id), zip(input_keys, inputs)
    )

    for chunk in subtask_chunk_graph.topological_iter():
        if chunk.key not in context:
            wrapped_execute = log_exception_wrapper(
                execute,
                "Execute operand %s of graph %s failed.",
                chunk.op,
                subtask_digraph,
            )
            wrapped_execute(context, chunk.op)

    # For non-mapper subtask, output context is chunk key to results.
    # For mapper subtasks, output context is data key to results.
    # `iter_output_data` must ensure values order since we only return values.
    output = {
        key: data for key, data, _ in iter_output_data(subtask_chunk_graph, context)
    }
    output_values = []
    if output_meta_keys:
        assert not is_mapper
        output_meta = {}
        # for non-shuffle subtask, record meta in supervisor.
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

    # assert output keys order consistent
    output_keys = output.keys()
    if is_mapper:
        chunk_keys, reducer_indices = zip(*output_keys)
        assert len(set(chunk_keys)) == 1, chunk_keys
        assert sorted(reducer_indices) == list(reducer_indices), (
            reducer_indices,
            sorted(reducer_indices),
        )
    else:
        expect_output_keys, _, _ = _get_subtask_out_info(
            subtask_chunk_graph, None, None
        )
        assert expect_output_keys == output_keys, (expect_output_keys, output_keys)

    logger.info("Finish executing subtask %s.", subtask_id)
    return output_values[0] if len(output_values) == 1 else output_values


def _get_subtask_out_info(
    subtask_chunk_graph: ChunkGraph,
    subtask: Optional[Subtask],
    shuffle_manager: Optional[ShuffleManager],
):
    # output_keys might be duplicate in chunk graph, use dict to deduplicate.
    # output_keys order should be consistent with remote `execute_subtask`,
    # dict can preserve insert order.
    output_keys = {}
    for chunk in subtask_chunk_graph.result_chunks:
        if isinstance(
            chunk.op, VirtualOperand
        ):  # FIXME(chaokunyang) no need to check this?
            continue
        elif (
            isinstance(chunk.op, MapReduceOperand)
            and chunk.op.stage == OperandStage.map
        ):
            assert (
                len(subtask_chunk_graph.result_chunks) == 1
            ), subtask_chunk_graph.result_chunks
            n_reducer = shuffle_manager.get_n_reducers(subtask)
            return set(), n_reducer, True
        else:
            output_keys[chunk.key] = 1
    return output_keys.keys(), len(output_keys), False


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
        self._ray_executor = _ray_export_once(execute_subtask)

        # api
        self._lifecycle_api = lifecycle_api
        self._meta_api = meta_api

        self._available_band_resources = None

        # For progress
        self._pre_all_stages_progress = 0.0
        self._pre_all_stages_tile_progress = 0
        self._cur_stage_tile_progress = 0
        self._cur_stage_output_object_refs = []

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
            _ray_export_once(RayTaskState)
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

        # For progress
        self._pre_all_stages_progress = 1
        self._pre_all_stages_tile_progress = 1
        self._cur_stage_tile_progress = 1
        self._cur_stage_output_object_refs = []

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def _get_apis(cls, session_id: str, address: str):
        return await asyncio.gather(
            LifecycleAPI.create(session_id, address),
            MetaAPI.create(session_id, address),
        )

    def get_execution_config(self):
        return self._config

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
        logger.info("Stage %s start.", stage_id)
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
        shuffle_manager = ShuffleManager(subtask_graph)
        for subtask in subtask_graph.topological_iter():
            if subtask.virtual:
                continue
            subtask_chunk_graph = subtask.chunk_graph
            input_object_refs = await self._load_subtask_inputs(
                stage_id, subtask, task_context, shuffle_manager
            )
            # can't use `subtask_graph.count_successors(subtask) == 0` to check whether output meta,
            # because a subtask can have some outputs which is dependent by downstream, but other outputs are not.
            # see https://user-images.githubusercontent.com/12445254/168484663-a4caa3f4-0ccc-4cd7-bf20-092356815073.png
            output_keys, out_count, is_shuffle_mapper = _get_subtask_out_info(
                subtask_chunk_graph, subtask, shuffle_manager
            )
            subtask_output_meta_keys = result_meta_keys & output_keys
            if is_shuffle_mapper:
                # shuffle meta won't be recorded in meta service.
                output_count = out_count
            else:
                output_count = out_count + bool(subtask_output_meta_keys)
            subtask_max_retries = (
                self._config.subtask_max_retries if subtask.retryable else 0
            )
            output_object_refs = self._ray_executor.options(
                num_returns=output_count, max_retries=subtask_max_retries
            ).remote(
                subtask.task_id,
                subtask.subtask_id,
                serialize(subtask_chunk_graph),
                subtask_output_meta_keys,
                *input_object_refs,
            )
            if output_count == 0:
                continue
            elif output_count == 1:
                output_object_refs = [output_object_refs]
            self._cur_stage_output_object_refs.extend(output_object_refs)
            if subtask_output_meta_keys:
                assert not is_shuffle_mapper
                meta_object_ref, *output_object_refs = output_object_refs
                # TODO(fyrestone): Fetch(not get) meta object here.
                output_meta_object_refs.append(meta_object_ref)
            if is_shuffle_mapper:
                shuffle_manager.add_mapper_output_refs(subtask, output_object_refs)
            else:
                subtask_outputs = zip(output_keys, output_object_refs)
                task_context.update(subtask_outputs)
        logger.info("Submitted %s subtasks of stage %s.", len(subtask_graph), stage_id)

        key_to_meta = {}
        if len(output_meta_object_refs) > 0:
            # TODO(fyrestone): Optimize update meta by fetching partial meta.
            meta_count = len(output_meta_object_refs)
            logger.info("Getting %s metas of stage %s.", meta_count, stage_id)
            meta_list = await asyncio.gather(*output_meta_object_refs)
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
        # Just use `self._cur_stage_tile_progress` as current stage progress
        # because current stage is finished, its progress is 1.
        self._pre_all_stages_progress += self._cur_stage_tile_progress
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
        stage_progress = 0.0
        total = len(self._cur_stage_output_object_refs)
        if total > 0:
            finished_objects, _ = ray.wait(
                self._cur_stage_output_object_refs,
                num_returns=total,
                timeout=0,  # Avoid blocking the asyncio loop.
                fetch_local=False,
            )
            stage_progress = (
                len(finished_objects) / total * self._cur_stage_tile_progress
            )
        return self._pre_all_stages_progress + stage_progress

    async def cancel(self):
        """Cancel execution."""

    async def _load_subtask_inputs(
        self,
        stage_id: str,
        subtask: Subtask,
        context: Dict,
        shuffle_manager: ShuffleManager,
    ):
        """
        Load input object refs of subtask from context.

        It updates the context if the input object refs are fetched from
        the meta service.
        """
        input_object_refs = []
        key_to_get_meta = {}
        # for non-shuffle chunks, chunk key will be used for indexing object refs.
        # for shuffle chunks, mapper subtasks will have only one mapper chunk, and all outputs for mapper
        # subtask will be shuffle blocks, the downstream reducers will receive inputs in the mappers order.
        start_chunks = list(subtask.chunk_graph.iter_indep())
        for index, start_chunk in enumerate(start_chunks):
            if isinstance(start_chunk.op, Fetch):
                # don't skip `pure_depend_keys`, otherwise pure_depend subtask execution order can't be ensured,
                # since those ray tasks don't has dependencies on other subtasks.
                chunk_key = start_chunk.key
                if chunk_key in context:
                    input_object_refs.append(context[chunk_key])
                else:
                    input_object_refs.append(None)
                    key_to_get_meta[index] = self._meta_api.get_chunk_meta.delay(
                        chunk_key, fields=["object_refs"]
                    )
            elif isinstance(start_chunk.op, PushShuffle):
                assert len(start_chunks) == 1, start_chunks
                # shuffle meta won't be recorded in meta service, query it from shuffle manager.
                return shuffle_manager.get_reducer_input_refs(subtask)
            else:
                assert not isinstance(start_chunk.op, FetchShuffle), start_chunk

        if key_to_get_meta:
            logger.info(
                "Fetch %s metas and update context of stage %s.",
                len(key_to_get_meta),
                stage_id,
            )
            meta_list = await self._meta_api.get_chunk_meta.batch(
                *key_to_get_meta.values()
            )
            for index, meta in zip(key_to_get_meta.keys(), meta_list):
                object_ref = meta["object_refs"][0]
                input_object_refs[index] = object_ref
                context[start_chunks[index].key] = object_ref
        return input_object_refs
