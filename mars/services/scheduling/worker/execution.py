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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from .... import oscar as mo
from ....core.graph import DAG
from ....core.operand import Fetch
from ....lib.aio import alru_cache
from ....storage import StorageLevel
from ....utils import dataslots
from ...cluster import ClusterAPI
from ...meta import MetaAPI
from ...storage import StorageAPI
from ...subtask import Subtask, SubtaskAPI, SubtaskResult, SubtaskStatus
from ..supervisor import GlobalSlotManagerActor
from .workerslot import BandSlotManagerActor
from .quota import QuotaActor

logger = logging.getLogger(__name__)


@dataslots
@dataclass
class SubtaskExecutionInfo:
    aio_task: asyncio.Task
    band_name: str
    supervisor_address: str
    result: SubtaskResult = field(default_factory=SubtaskResult)
    cancelling: bool = False
    max_runs: int = 1
    num_runs: int = 0
    slot_id: Optional[int] = None
    kill_timeout: Optional[int] = None


class SubtaskExecutionActor(mo.Actor):
    _subtask_info: Dict[str, SubtaskExecutionInfo]

    def __init__(self):
        self._cluster_api = None
        self._global_slot_ref = None

        self._subtask_info = dict()
        self._size_pool = ThreadPoolExecutor(1)

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)
        try:
            [self._global_slot_ref] = await self._cluster_api.get_supervisor_refs(
                [GlobalSlotManagerActor.default_uid()])
        except mo.ActorNotExist:
            self._global_slot_ref = None

    @alru_cache
    async def _get_slot_manager_ref(self, band: str) -> Union[mo.ActorRef, BandSlotManagerActor]:
        return await mo.actor_ref(BandSlotManagerActor.gen_uid(band), address=self.address)

    @alru_cache
    async def _get_band_quota_ref(self, band: str) -> Union[mo.ActorRef, QuotaActor]:
        return await mo.actor_ref(QuotaActor.gen_uid(band), address=self.address)

    @staticmethod
    @alru_cache
    async def _get_task_api(supervisor_address: str, session_id: str):
        from ...task import TaskAPI
        return await TaskAPI.create(session_id, supervisor_address)

    async def _prepare_input_data(self, subtask: Subtask):
        queries = []
        storage_api = await StorageAPI.create(subtask.session_id, address=self.address)
        pure_dep_keys = set()
        for n in subtask.chunk_graph:
            pure_dep_keys.update(
                inp.key for inp, pure_dep in zip(n.inputs, n.op.pure_depends) if pure_dep)
        for n in subtask.chunk_graph:
            if not isinstance(n.op, Fetch) or n.key in pure_dep_keys:
                continue
            queries.append(storage_api.fetch.delay(n.key))
        if queries:
            await storage_api.fetch.batch(*queries)

    async def _collect_input_sizes(self, subtask: Subtask, supervisor_address: str):
        storage_api = await StorageAPI.create(subtask.session_id, address=self.address)
        meta_api = await MetaAPI.create(subtask.session_id, address=supervisor_address)

        graph = subtask.chunk_graph
        sizes = dict()

        fetch_keys = list(
            set(n.key for n in graph.iter_indep() if isinstance(n.op, Fetch)))
        fetch_metas = await meta_api.get_chunk_meta.batch(
            *(meta_api.get_chunk_meta.delay(k, fields=['memory_size', 'store_size'])
              for k in fetch_keys))
        data_infos = await storage_api.get_infos.batch(
            *(storage_api.get_infos.delay(k) for k in fetch_keys)
        )

        for key, meta, infos in zip(fetch_keys, fetch_metas, data_infos):
            level = functools.reduce(operator.or_, (info.level for info in infos))
            if level & StorageLevel.MEMORY:
                sizes[key] = (meta['memory_size'], meta['memory_size'] - meta['store_size'])
            else:
                sizes[key] = (meta['memory_size'], meta['memory_size'])

        return sizes

    @classmethod
    def _estimate_sizes(cls, subtask: Subtask, input_sizes: Dict):
        size_context = {k: (s, 0) for k, (s, _c) in input_sizes.items()}
        graph = subtask.chunk_graph

        key_to_ops = defaultdict(set)
        for n in graph:
            key_to_ops[n.op.key].add(n.op)
        key_to_ops = {k: list(v) for k, v in key_to_ops.items()}

        # condense op key graph
        op_key_graph = DAG()
        for n in graph.topological_iter():
            if n.op.key not in op_key_graph:
                op_key_graph.add_node(n.op.key)
            for succ in graph.iter_successors(n):
                if succ.op.key not in op_key_graph:
                    op_key_graph.add_node(succ.op.key)
                op_key_graph.add_edge(n.op.key, succ.op.key)

        key_stack = list(op_key_graph.iter_indep())
        pred_ref_count = {k: op_key_graph.count_predecessors(k) for k in op_key_graph}
        succ_ref_count = {k: op_key_graph.count_successors(k) for k in op_key_graph}

        visited_op_keys = set()
        max_memory_cost = 0
        while key_stack:
            key = key_stack.pop()
            op = key_to_ops[key][0]

            if not isinstance(op, Fetch):
                op.estimate_size(size_context, op)
            max_memory_cost = max(
                sum(ms for _, ms in size_context.values()), max_memory_cost)
            visited_op_keys.add(op.key)

            for succ_op_key in op_key_graph.iter_successors(key):
                pred_ref_count[succ_op_key] -= 1
                if pred_ref_count[succ_op_key] == 0:
                    key_stack.append(succ_op_key)
            for pred_op_key in op_key_graph.iter_predecessors(key):
                succ_ref_count[pred_op_key] -= 1
                if succ_ref_count[pred_op_key] == 0:
                    size_context.pop(pred_op_key, None)
        return sum(t[0] for t in size_context.values()), \
            sum(t[1] for t in size_context.values())

    @classmethod
    def _build_quota_request(cls, subtask_id: str, calc_size: int, input_sizes: Dict):
        batch_req = dict()
        for input_key, (_input_store_size, input_calc_size) in input_sizes.items():
            if input_calc_size > 0:
                batch_req[(subtask_id, input_key)] = input_calc_size
        batch_req[(subtask_id, subtask_id)] = calc_size
        return batch_req

    async def internal_run_subtask(self, subtask: Subtask, band_name: str):
        subtask_api = SubtaskAPI(self.address)
        subtask_info = self._subtask_info[subtask.subtask_id]
        subtask_info.result = SubtaskResult(subtask_id=subtask.subtask_id,
                                            session_id=subtask.session_id,
                                            task_id=subtask.task_id,
                                            status=SubtaskStatus.pending)
        batch_quota_req = quota_ref = slot_manager_ref = None

        try:
            quota_ref = await self._get_band_quota_ref(band_name)
            slot_manager_ref = await self._get_slot_manager_ref(band_name)

            yield self._prepare_input_data(subtask)

            input_sizes = await self._collect_input_sizes(subtask, subtask_info.supervisor_address)
            loop = asyncio.get_running_loop()
            _store_size, calc_size = yield loop.run_in_executor(
                self._size_pool, self._estimate_sizes, subtask, input_sizes)
            if subtask_info.cancelling:
                raise asyncio.CancelledError

            batch_quota_req = self._build_quota_request(
                subtask.subtask_id, calc_size, input_sizes)

            quota_aiotask = asyncio.create_task(
                quota_ref.request_batch_quota(batch_quota_req))
            yield quota_aiotask
            if subtask_info.cancelling:
                raise asyncio.CancelledError

            subtask_info.result = yield self._retry_run_subtask(subtask, band_name, subtask_api)
        except asyncio.CancelledError as ex:
            subtask_info.result.status = SubtaskStatus.cancelled
            subtask_info.result.progress = 1.0
            raise ex
        except:  # noqa: E722  # pylint: disable=bare-except
            logger.exception('Failed to run subtask %s on band %s', subtask.subtask_id, band_name)
            _, exc, tb = sys.exc_info()
            subtask_info.result.status = SubtaskStatus.errored
            subtask_info.result.progress = 1.0
            subtask_info.result.error = exc
            subtask_info.result.traceback = tb
        finally:
            if batch_quota_req:
                await quota_ref.release_quotas(list(batch_quota_req.keys()))

            self._subtask_info.pop(subtask.subtask_id, None)
            if self._global_slot_ref is not None:
                yield (
                    # make sure slot is released before marking tasks as finished
                    self._global_slot_ref.release_subtask_slots(
                        (self.address, band_name), subtask.session_id, subtask.subtask_id),
                    # make sure new slot usages are uploaded in time
                    slot_manager_ref.upload_slot_usages(periodical=False)
                )
                logger.debug('Slot released for band %s after subtask %s',
                             band_name, subtask.subtask_id)

            task_api = await self._get_task_api(subtask_info.supervisor_address,
                                                subtask.session_id)
            yield task_api.set_subtask_result(subtask_info.result)

    async def _retry_run_subtask(self, subtask: Subtask, band_name: str, subtask_api: SubtaskAPI):
        slot_manager_ref = await self._get_slot_manager_ref(band_name)
        subtask_info = self._subtask_info[subtask.subtask_id]
        assert subtask_info.max_runs > 0

        class SlotContext:
            def __init__(self):
                self._slot_id = None

            @property
            def slot_id(self):
                return self._slot_id

            async def _cancel_subtask(self):
                try:
                    await asyncio.wait_for(
                            asyncio.shield(subtask_api.cancel_subtask_in_slot(
                                    band_name, self._slot_id)),
                            subtask_info.kill_timeout)
                except asyncio.TimeoutError:
                    return True
                return False

            async def __aenter__(self):
                # TODO(fyrestone): Replace await with yield to release the actor lock.
                self._slot_id = await slot_manager_ref.acquire_free_slot(
                        (subtask.session_id, subtask.subtask_id))
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                try:
                    if exc_type is None:
                        should_kill_slot = False
                    elif issubclass(exc_type, asyncio.CancelledError):
                        should_kill_slot = await self._cancel_subtask()
                    else:
                        # TODO(fyrestone): Handle slot exception correctly.
                        should_kill_slot = True

                    if should_kill_slot:
                        try:
                            # this may cause the failed slot to restart multiple times.
                            await slot_manager_ref.kill_slot(self._slot_id)
                        except ConnectionError:
                            pass
                        finally:
                            # TODO(fyrestone): Make the slot management more reliable.
                            # currently, the slot will not be freed if we kill slot failed.
                            self._slot_id = None
                finally:
                    if self._slot_id is not None:
                        await slot_manager_ref.release_free_slot(self._slot_id)

        last_ex = None
        for _ in range(subtask_info.num_runs, subtask_info.max_runs):
            subtask_info.num_runs += 1
            try:
                async with SlotContext() as ctx:
                    if subtask_info.cancelling:
                        raise asyncio.CancelledError

                    subtask_info.result.status = SubtaskStatus.running
                    # TODO(fyrestone): Replace await with yield to release the actor lock.
                    return await asyncio.shield(subtask_api.run_subtask_in_slot(
                            band_name, ctx.slot_id, subtask))
            except Exception as ex:
                logger.error('rerun subtask %s due to %s', subtask.subtask_id, ex)
                last_ex = ex
        else:
            if last_ex is not None:
                raise last_ex
            raise RuntimeError(f'The {subtask} should be run at least once.')

    async def run_subtask(self, subtask: Subtask, band_name: str,
                          supervisor_address: str):
        with mo.debug.no_message_trace():
            task = asyncio.create_task(self.ref().internal_run_subtask(subtask, band_name))
        self._subtask_info[subtask.subtask_id] = \
            SubtaskExecutionInfo(task, band_name, supervisor_address,
                                 max_runs=subtask.extra_config.get('subtask_max_runs', 1))
        return task

    async def cancel_subtask(self, subtask_id: str, kill_timeout: int = 5):
        try:
            subtask_info = self._subtask_info[subtask_id]
        except KeyError:
            return

        if not subtask_info.cancelling:
            subtask_info.kill_timeout = kill_timeout
            subtask_info.cancelling = True
            subtask_info.aio_task.cancel()

        try:
            yield subtask_info.aio_task
        except asyncio.CancelledError:
            pass
