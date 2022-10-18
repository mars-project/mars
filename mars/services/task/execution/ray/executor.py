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
import logging

import operator
import sys
from dataclasses import dataclass

from typing import List, Dict, Any, Set, Callable
from .....core import ChunkGraph, Chunk, TileContext
from .....core.context import set_context
from .....core.operand import (
    Fetch,
    Fuse,
    VirtualOperand,
    execute,
)
from .....core.operand.fetch import FetchShuffle
from .....lib.aio import alru_cache
from .....lib.ordered_set import OrderedSet
from .....resource import Resource
from .....serialization import serialize, deserialize
from .....typing import BandType
from .....utils import (
    calc_data_size,
    lazy_import,
    get_chunk_params,
)
from ....lifecycle.api import LifecycleAPI
from ....meta.api import MetaAPI
from ....subtask import Subtask, SubtaskGraph
from ....subtask.utils import iter_output_data
from ...core import Task
from ..api import (
    TaskExecutor,
    ExecutionChunkResult,
    register_executor_cls,
)
from ..utils import ResultTileablesLifecycle
from .config import RayExecutionConfig, IN_RAY_CI
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
    handle = None

    @classmethod
    def get_handle(cls):
        """Get the RayTaskState actor handle."""
        logger.info("Getting RayTaskState handle.")
        return ray.get_actor(cls.__name__)

    @classmethod
    def create(cls):
        """Create a RayTaskState actor."""
        logger.info("Creating RayTaskState actor.")
        name = cls.__name__
        try:
            cls.handle = ray.get_actor(name)
        except ValueError:
            # Attempt to create it (may race with other attempts).
            try:
                cls.handle = ray.remote(cls).options(name=name).remote()
            except ValueError:  # pragma: no cover
                # We lost the creation race, ignore.
                cls.handle = ray.get_actor(name)
        return cls.handle


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
    subtask_id: str,
    subtask_chunk_graph: ChunkGraph,
    output_meta_n_keys: int,
    is_mapper,
    *inputs,
):
    """
    The function used for execute subtask in ray task.

    Parameters
    ----------
    subtask_id: str
        id of subtask
    subtask_chunk_graph: ChunkGraph
        chunk graph for subtask
    output_meta_n_keys: int
        will be 0 if subtask is a shuffle mapper.
    is_mapper: bool
        Whether current subtask is a shuffle mapper. Note that shuffle reducers such as `DataFrameDropDuplicates`
        can be a mapper at the same time.
    inputs:
        inputs for current subtask

    Returns
    -------
        subtask outputs and meta for outputs if `output_meta_keys` is provided.
    """
    subtask_chunk_graph = deserialize(*subtask_chunk_graph)
    logger.info("Begin to execute subtask: %s", subtask_id)
    # optimize chunk graph.
    subtask_chunk_graph = _optimize_subtask_graph(subtask_chunk_graph)
    fetch_chunks, shuffle_chunk = _get_fetch_chunks(subtask_chunk_graph)
    input_data = {}
    if shuffle_chunk is not None:
        # the subtask is a reducer subtask
        n_mappers = shuffle_chunk.op.n_mappers
        # some reducer may have multiple output chunks, see `PSRSshuffle._execute_reduce` and
        # https://user-images.githubusercontent.com/12445254/168569524-f09e42a7-653a-4102-bdf0-cc1631b3168d.png
        reducer_chunks = subtask_chunk_graph.successors(shuffle_chunk)
        reducer_operands = set(c.op for c in reducer_chunks)
        if len(reducer_operands) != 1:  # pragma: no cover
            raise ValueError(
                f"Subtask {subtask_id} has more than 1 reduce operands: {subtask_chunk_graph.to_dot()}"
            )
        reducer_operand = reducer_chunks[0].op
        reducer_index = reducer_operand.reducer_index
        # mock input keys, keep this in sync with `MapReducerOperand#_iter_mapper_key_idx_pairs`
        input_data.update(
            {(i, reducer_index): block for i, block in enumerate(inputs[-n_mappers:])}
        )
        inputs = inputs[:-n_mappers]
    input_keys = [start_chunk.key for start_chunk in fetch_chunks]
    input_data.update(zip(input_keys, inputs))
    context = RayExecutionWorkerContext(RayTaskState.get_handle, input_data)

    for chunk in subtask_chunk_graph.topological_iter():
        if chunk.key not in context:
            try:
                context.set_current_chunk(chunk)
                execute(context, chunk.op)
            except Exception:
                logger.exception(
                    "Execute operand %s of graph %s failed.",
                    chunk.op,
                    subtask_chunk_graph.to_dot(),
                )
                raise

    # For non-mapper subtask, output context is chunk key to results.
    # For mapper subtasks, output context is data key to results.
    # `iter_output_data` must ensure values order since we only return values.
    normal_output = {}
    mapper_output = {}
    for key, data, is_mapper_block in iter_output_data(subtask_chunk_graph, context):
        if is_mapper_block:
            mapper_output[key] = data
        else:
            normal_output[key] = data
    output_values = []
    # assert output keys order consistent
    if is_mapper:
        # mapper may produce outputs which isn't shuffle blocks, such as TensorUnique._execute_agg_reduce.
        mapper_main_keys = set(k[0] for k in mapper_output.keys())
        assert len(mapper_main_keys) == 1, mapper_main_keys
        # sorted reducer_index's consistency with reducer_ordinal is checked in
        # `OperandTilesHandler._check_shuffle_reduce_chunks`.
        # So sort keys by reducer_index to ensure mapper outputs consist with reducer_ordinal,
        # then downstream can fetch shuffle blocks by reducer_ordinal.
        mapper_output = dict(sorted(mapper_output.items(), key=lambda item: item[0][1]))
    if output_meta_n_keys:
        output_meta = {}
        # for non-shuffle subtask, record meta in supervisor.
        for chunk in subtask_chunk_graph.result_chunks[:output_meta_n_keys]:
            chunk_key = chunk.key
            if chunk_key not in output_meta:
                if isinstance(chunk.op, Fuse):  # pragma: no cover
                    # fuse op
                    chunk = chunk.chunk
                data = context[chunk_key]
                memory_size = calc_data_size(data)
                output_meta[chunk_key] = get_chunk_params(chunk), memory_size
        output_values.append(output_meta)
    output_values.extend(normal_output.values())
    output_values.extend(mapper_output.values())
    logger.info("Finish executing subtask %s.", subtask_id)
    return output_values[0] if len(output_values) == 1 else output_values


def _get_fetch_chunks(chunk_graph):
    fetch_chunks = []
    shuffle_chunk = None
    for start_chunk in chunk_graph.iter_indep():
        if isinstance(start_chunk.op, FetchShuffle):
            assert shuffle_chunk is None, shuffle_chunk
            shuffle_chunk = start_chunk
        elif isinstance(start_chunk.op, Fetch):
            fetch_chunks.append(start_chunk)
    return sorted(fetch_chunks, key=operator.attrgetter("key")), shuffle_chunk


def _get_subtask_out_info(
    subtask_chunk_graph: ChunkGraph, is_mapper: bool, n_reducers: int = None
):
    # output_keys might be duplicate in chunk graph, use dict to deduplicate.
    # output_keys order should be consistent with remote `execute_subtask`,
    # dict can preserve insert order.
    output_keys = {}
    shuffle_chunk = None
    if is_mapper:
        assert n_reducers is not None
        if len(subtask_chunk_graph.result_chunks) == 1:
            return set(), n_reducers
        for chunk in subtask_chunk_graph.result_chunks:
            if not chunk.is_mapper:
                output_keys[chunk.key] = 1
                # mapper may produce outputs which isn't shuffle blocks, such as TensorUnique._execute_agg_reduce
                # which is  mapper too, but some outputs are not mapper blocks:
                # https://user-images.githubusercontent.com/12445254/184132642-a19259fd-43d6-4a27-a033-4aaa97d7586e.svg
            else:
                assert shuffle_chunk is None, (shuffle_chunk, chunk)
                shuffle_chunk = chunk
        return output_keys.keys(), len(output_keys) + n_reducers
    for chunk in subtask_chunk_graph.result_chunks:
        if isinstance(
            chunk.op, VirtualOperand
        ):  # FIXME(chaokunyang) no need to check this?
            continue
        else:
            output_keys[chunk.key] = 1
    return output_keys.keys(), len(output_keys)


@register_executor_cls
class RayTaskExecutor(TaskExecutor):
    name = "ray"

    def __init__(
        self,
        config: RayExecutionConfig,
        task: Task,
        tile_context: TileContext,
        task_context: Dict[str, "ray.ObjectRef"],
        task_chunks_meta: Dict[str, _RayChunkMeta],
        lifecycle_api: LifecycleAPI,
        meta_api: MetaAPI,
    ):
        self._config = config
        self._task = task
        self._tile_context = tile_context
        self._task_context = task_context
        self._task_chunks_meta = task_chunks_meta
        self._ray_executor = self._get_ray_executor()

        # api
        self._lifecycle_api = lifecycle_api
        self._meta_api = meta_api

        self._available_band_resources = None
        self._result_tileables_lifecycle = None

        # For progress and task cancel
        self._pre_all_stages_progress = 0.0
        self._pre_all_stages_tile_progress = 0.0
        self._cur_stage_progress = 0.0
        self._cur_stage_tile_progress = 0.0
        self._cur_stage_first_output_object_ref_to_subtask = dict()
        self._execute_subtask_graph_aiotask = None
        self._cancelled = False

    @classmethod
    async def create(
        cls,
        config: RayExecutionConfig,
        *,
        session_id: str,
        address: str,
        task: Task,
        tile_context: TileContext,
        **kwargs,
    ) -> "RayTaskExecutor":
        lifecycle_api, meta_api = await cls._get_apis(session_id, address)
        task_context = {}
        task_chunks_meta = {}

        executor = cls(
            config,
            task,
            tile_context,
            task_context,
            task_chunks_meta,
            lifecycle_api,
            meta_api,
        )
        available_band_resources = await executor.get_available_band_resources()
        worker_addresses = list(
            map(operator.itemgetter(0), available_band_resources.keys())
        )
        await cls._init_context(
            config,
            task_context,
            task_chunks_meta,
            RayTaskState.create,
            worker_addresses,
            session_id,
            address,
        )
        return executor

    def get_execution_config(self):
        return self._config

    # noinspection DuplicatedCode
    def destroy(self):
        self._task = None
        self._tile_context = None
        self._task_context = {}
        self._task_chunks_meta = {}
        self._ray_executor = None

        # api
        self._lifecycle_api = None
        self._meta_api = None

        self._available_band_resources = None
        self._result_tileables_lifecycle = None

        # For progress and task cancel
        self._pre_all_stages_progress = 1.0
        self._pre_all_stages_tile_progress = 1.0
        self._cur_stage_progress = 1.0
        self._cur_stage_tile_progress = 1.0
        self._cur_stage_first_output_object_ref_to_subtask = dict()
        self._execute_subtask_graph_aiotask = None
        self._cancelled = None
        self._config = None

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
        config: RayExecutionConfig,
        task_context: Dict[str, "ray.ObjectRef"],
        task_chunks_meta: Dict[str, _RayChunkMeta],
        create_task_state_actor: Callable[[], "ray.actor.ActorHandle"],
        worker_addresses: List[str],
        session_id: str,
        address: str,
    ):
        loop = asyncio.get_running_loop()
        context = RayExecutionContext(
            config,
            task_context,
            task_chunks_meta,
            worker_addresses,
            create_task_state_actor,
            session_id,
            address,
            address,
            address,
            loop=loop,
        )
        await context.init()
        set_context(context)

    async def __aenter__(self):
        self._result_tileables_lifecycle = ResultTileablesLifecycle(
            self._task.tileable_graph, self._tile_context, self._lifecycle_api
        )

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
        logger.info("Stage %s start.", stage_id)
        # Make sure each stage use a clean dict.
        self._cur_stage_first_output_object_ref_to_subtask = dict()

        def _on_monitor_aiotask_done(fut):
            # Print the error of monitor task.
            try:
                fut.result()
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover
                logger.exception(
                    "The monitor task of stage %s is done with exception.", stage_id
                )
                if IN_RAY_CI:  # pragma: no cover
                    logger.warning(
                        "The process will be exit due to the monitor task exception "
                        "when MARS_CI_BACKEND=ray."
                    )
                    sys.exit(-1)

        result_meta_keys = {
            chunk.key
            for chunk in chunk_graph.result_chunks
            if not isinstance(chunk.op, Fetch)
        }
        # Create a monitor task to update progress and collect garbage.
        monitor_aiotask = asyncio.create_task(
            self._update_progress_and_collect_garbage(
                stage_id,
                subtask_graph,
                result_meta_keys,
                self._config.get_subtask_monitor_interval(),
            )
        )
        monitor_aiotask.add_done_callback(_on_monitor_aiotask_done)

        def _on_execute_aiotask_done(_):
            # Make sure the monitor task is cancelled.
            monitor_aiotask.cancel()
            # Just use `self._cur_stage_tile_progress` as current stage progress
            # because current stage is completed, its progress is 1.0.
            self._cur_stage_progress = 1.0
            self._pre_all_stages_progress += self._cur_stage_tile_progress

        self._execute_subtask_graph_aiotask = asyncio.current_task()
        self._execute_subtask_graph_aiotask.add_done_callback(_on_execute_aiotask_done)

        task_context = self._task_context
        output_meta_object_refs = []
        self._pre_all_stages_tile_progress = (
            self._pre_all_stages_tile_progress + self._cur_stage_tile_progress
        )
        self._cur_stage_tile_progress = (
            self._tile_context.get_all_progress() - self._pre_all_stages_tile_progress
        )
        # Previous execution may have duplicate tileable ids, the tileable may be decref
        # during execution, so we should track and incref the result tileables before execute.
        await self._result_tileables_lifecycle.incref_tiled()
        logger.info("Submitting %s subtasks of stage %s.", len(subtask_graph), stage_id)
        shuffle_manager = ShuffleManager(subtask_graph)
        subtask_max_retries = self._config.get_subtask_max_retries()
        subtask_num_cpus = self._config.get_subtask_num_cpus()
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
            is_mapper, n_reducers = shuffle_manager.is_mapper(subtask), None
            if is_mapper:
                n_reducers = shuffle_manager.get_n_reducers(subtask)
            output_keys, out_count = _get_subtask_out_info(
                subtask_chunk_graph, is_mapper, n_reducers
            )
            if is_mapper:
                # shuffle meta won't be recorded in meta service.
                output_count = out_count
            else:
                output_count = out_count + bool(subtask.stage_n_outputs)
            subtask_max_retries = subtask_max_retries if subtask.retryable else 0
            output_object_refs = self._ray_executor.options(
                num_cpus=subtask_num_cpus,
                num_returns=output_count,
                max_retries=subtask_max_retries,
                scheduling_strategy="DEFAULT" if len(input_object_refs) else "SPREAD",
            ).remote(
                subtask.subtask_id,
                serialize(subtask_chunk_graph, context={"serializer": "ray"}),
                subtask.stage_n_outputs,
                is_mapper,
                *input_object_refs,
            )
            if output_count == 0:
                continue
            elif output_count == 1:
                output_object_refs = [output_object_refs]
            self._cur_stage_first_output_object_ref_to_subtask[
                output_object_refs[0]
            ] = subtask
            if subtask.stage_n_outputs:
                meta_object_ref, *output_object_refs = output_object_refs
                # TODO(fyrestone): Fetch(not get) meta object here.
                output_meta_object_refs.append(meta_object_ref)
            if is_mapper:
                shuffle_manager.add_mapper_output_refs(
                    subtask, output_object_refs[-n_reducers:]
                )
                output_object_refs = output_object_refs[:-n_reducers]
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
            # The result chunk may be in previous stage result,
            # then the chunk does not have to be processed.
            if chunk_key in task_context:
                object_ref = task_context[chunk_key]
                output_object_refs.add(object_ref)
                chunk_params = key_to_meta.get(chunk_key)
                if chunk_params is not None:
                    chunk_to_meta[chunk] = ExecutionChunkResult(
                        chunk_params, object_ref
                    )

        logger.info("Waiting for stage %s complete.", stage_id)
        # Patched the asyncio.to_thread for Python < 3.9 at mars/lib/aio/__init__.py
        await asyncio.to_thread(ray.wait, list(output_object_refs), fetch_local=False)

        logger.info("Stage %s is complete.", stage_id)
        return chunk_to_meta

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self._result_tileables_lifecycle.decref_tracked()
            try:
                await self.cancel()
            except BaseException:  # noqa: E722  # nosec  # pylint: disable=bare-except
                pass
            return

        # Update info if no exception occurs.
        update_metas = []
        for tileable in self._task.tileable_graph.result_tileables:
            tileable = tileable.data if hasattr(tileable, "data") else tileable
            chunk_keys = []
            for chunk in self._tile_context[tileable].chunks:
                chunk_key = chunk.key
                chunk_keys.append(chunk_key)
                if (
                    chunk_key in self._task_context
                    and chunk_key in self._task_chunks_meta
                ):
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
        if update_metas:
            await self._meta_api.set_chunk_meta.batch(*update_metas)

    async def get_available_band_resources(self) -> Dict[BandType, Resource]:
        if self._available_band_resources is None:
            band_resources = self._config.get_band_resources()
            virtual_band_resources = {}
            idx = 0
            for band_resource in band_resources:
                for band, resource in band_resource.items():
                    virtual_band_resources[
                        (f"ray_virtual_address_{idx}:0", band)
                    ] = resource
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
        if self._task is None or self._cancelled is True:
            return
        self._cancelled = True
        if self._execute_subtask_graph_aiotask is not None:
            self._execute_subtask_graph_aiotask.cancel()
        timeout = self._config.get_subtask_cancel_timeout()
        to_be_cancelled_coros = [
            _cancel_ray_task(object_ref, timeout)
            for object_ref in self._cur_stage_first_output_object_ref_to_subtask.keys()
        ]
        await asyncio.gather(*to_be_cancelled_coros)

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
        normal_object_refs = []
        shuffle_object_refs = []
        key_to_get_meta = {}
        # for non-shuffle chunks, chunk key will be used for indexing object refs.
        # for shuffle chunks, mapper subtasks will have only one mapper chunk, and all outputs for mapper
        # subtask will be shuffle blocks, the downstream reducers will receive inputs in the mappers order.
        fetch_chunks, shuffle_chunk = _get_fetch_chunks(subtask.chunk_graph)
        for index, fetch_chunk in enumerate(fetch_chunks):
            chunk_key = fetch_chunk.key
            # pure_depend data is not used, skip it.
            if chunk_key in subtask.pure_depend_keys:
                normal_object_refs.append(None)
            elif chunk_key in context:
                normal_object_refs.append(context[chunk_key])
            else:
                normal_object_refs.append(None)
                key_to_get_meta[index] = self._meta_api.get_chunk_meta.delay(
                    chunk_key, fields=["object_refs"]
                )
        if shuffle_chunk is not None:
            # shuffle meta won't be recorded in meta service, query it from shuffle manager.
            shuffle_object_refs = list(shuffle_manager.get_reducer_input_refs(subtask))

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
                normal_object_refs[index] = object_ref
                context[fetch_chunks[index].key] = object_ref
        return normal_object_refs + shuffle_object_refs

    async def _update_progress_and_collect_garbage(
        self,
        stage_id: str,
        subtask_graph: SubtaskGraph,
        result_meta_keys: Set[str],
        interval_seconds: float,
    ):
        object_ref_to_subtask = self._cur_stage_first_output_object_ref_to_subtask
        total = len(subtask_graph)
        completed_subtasks = OrderedSet()

        def gc():
            """
            Consume the completed subtasks and collect garbage.

            GC the output object refs of the subtask which successors are submitted
            (not completed as above) can reduce the memory peaks, but we can't cancel
            and rerun slow subtasks because the input object refs of running subtasks
            may be deleted.
            """
            i = 0
            gc_subtasks = set()

            while i < total:
                while i >= len(completed_subtasks):
                    yield
                # Iterate the completed subtasks once.
                subtask = completed_subtasks[i]
                i += 1
                logger.debug("GC[stage=%s]: %s", stage_id, subtask)

                # Note: There may be a scenario in which delayed gc occurs.
                # When a subtask has more than one predecessor, like A, B,
                # and in the `for ... in ...` loop we get A firstly while
                # B's successors are completed, A's not. Then we cannot remove
                # B's results chunks before A's.
                for pred in subtask_graph.iter_predecessors(subtask):
                    if pred in gc_subtasks:
                        continue
                    while not all(
                        succ in completed_subtasks
                        for succ in subtask_graph.iter_successors(pred)
                    ):
                        yield
                    for chunk in pred.chunk_graph.results:
                        chunk_key = chunk.key
                        # We need to check the GC chunk key is not in the
                        # result meta keys, because there are some special
                        # cases that the result meta keys are not the leaves.
                        #
                        # example: test_cut_execution
                        if chunk_key not in result_meta_keys:
                            logger.debug("GC[stage=%s]: %s", stage_id, chunk)
                            self._task_context.pop(chunk_key, None)
                    gc_subtasks.add(pred)

            # TODO(fyrestone): Check the remaining self._task_context.keys()
            # in the result subtasks

        collect_garbage = gc()

        while len(completed_subtasks) < total:
            if len(object_ref_to_subtask) <= 0:  # pragma: no cover
                await asyncio.sleep(interval_seconds)
                # We should run ray.wait after at least one Ray task is submitted.
                # Please refer to: https://github.com/mars-project/mars/issues/3274
                continue

            # Only wait for unready subtask object refs.
            ready_objects, _ = await asyncio.to_thread(
                ray.wait,
                list(object_ref_to_subtask.keys()),
                num_returns=len(object_ref_to_subtask),
                timeout=0,
                fetch_local=False,
            )
            if len(ready_objects) == 0:
                await asyncio.sleep(interval_seconds)
                continue

            # Pop the completed subtasks from object_ref_to_subtask.
            completed_subtasks.update(map(object_ref_to_subtask.pop, ready_objects))
            # Update progress.
            stage_progress = (
                len(completed_subtasks) / total * self._cur_stage_tile_progress
            )
            self._cur_stage_progress = self._pre_all_stages_progress + stage_progress
            # Collect garbage, use `for ... in ...` to avoid raising StopIteration.
            for _ in collect_garbage:
                break
            # Fast to next loop and give it a chance to update object_ref_to_subtask.
            await asyncio.sleep(0)
