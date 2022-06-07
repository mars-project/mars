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
    MapReduceOperand,
    execute,
)
from .....lib.aio import alru_cache
from .....lib.ordered_set import OrderedSet
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
    ExecutionChunkResult,
    register_executor_cls,
)
from .config import RayExecutionConfig, IN_RAY_CI
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
    def _gen_name(cls, task_id: str):
        return f"{cls.__name__}_{task_id}"

    @classmethod
    def get_handle(cls, task_id: str):
        """Get the RayTaskState actor handle."""
        name = cls._gen_name(task_id)
        logger.info("Getting %s handle.", name)
        return ray.get_actor(name)

    @classmethod
    def create(cls, task_id: str):
        """Create a RayTaskState actor."""
        name = cls._gen_name(task_id)
        logger.info("Creating %s.", name)
        return ray.remote(cls).options(name=name).remote()


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
        lambda: RayTaskState.get_handle(task_id), zip(input_keys, inputs)
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
        if config.create_task_state_actor_as_needed():
            create_task_state_actor = lambda: RayTaskState.create(  # noqa: E731
                task_id=task.task_id
            )
        else:
            actor_handle = RayTaskState.create(task_id=task.task_id)
            create_task_state_actor = lambda: actor_handle  # noqa: E731
        await cls._init_context(
            config,
            task_context,
            task_chunks_meta,
            create_task_state_actor,
            worker_addresses,
            session_id,
            address,
        )
        return executor

    # noinspection DuplicatedCode
    def destroy(self):
        self._config = None
        self._task = None
        self._tile_context = None
        self._task_context = {}
        self._task_chunks_meta = {}
        self._ray_executor = None

        # api
        self._lifecycle_api = None
        self._meta_api = None

        self._available_band_resources = None

        # For progress and task cancel
        self._pre_all_stages_progress = 1.0
        self._pre_all_stages_tile_progress = 1.0
        self._cur_stage_progress = 1.0
        self._cur_stage_tile_progress = 1.0
        self._cur_stage_first_output_object_ref_to_subtask = dict()
        self._execute_subtask_graph_aiotask = None
        self._cancelled = None

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
        logger.info("Submitting %s subtasks of stage %s.", len(subtask_graph), stage_id)
        subtask_max_retries = self._config.get_subtask_max_retries()
        for subtask in subtask_graph.topological_iter():
            subtask_chunk_graph = subtask.chunk_graph
            key_to_input = await self._load_subtask_inputs(
                stage_id, subtask, subtask_chunk_graph, task_context
            )
            output_keys = self._get_subtask_output_keys(subtask_chunk_graph)
            output_meta_keys = result_meta_keys & output_keys
            output_count = len(output_keys) + bool(output_meta_keys)
            max_retries = subtask_max_retries if subtask.retryable else 0
            output_object_refs = self._ray_executor.options(
                num_returns=output_count, max_retries=max_retries
            ).remote(
                subtask.task_id,
                subtask.subtask_id,
                serialize(subtask_chunk_graph),
                output_meta_keys,
                list(key_to_input.keys()),
                *key_to_input.values(),
            )
            if output_count == 0:
                continue
            elif output_count == 1:
                output_object_refs = [output_object_refs]
            self._cur_stage_first_output_object_ref_to_subtask[
                output_object_refs[0]
            ] = subtask
            if output_meta_keys:
                meta_object_ref, *output_object_refs = output_object_refs
                # TODO(fyrestone): Fetch(not get) meta object here.
                output_meta_object_refs.append(meta_object_ref)
            task_context.update(zip(output_keys, output_object_refs))
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

        logger.info("Stage %s is complete.", stage_id)
        return chunk_to_meta

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                await self.cancel()
            except BaseException:  # noqa: E722  # nosec  # pylint: disable=bare-except
                pass
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
