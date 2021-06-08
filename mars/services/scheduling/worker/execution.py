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
from ....oscar import ServerClosed
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
        slot_id = batch_quota_req = run_aiotask = None
        quota_ref = slot_manager = None

        try:
            quota_ref = await self._get_band_quota_ref(band_name)
            slot_manager = await self._get_slot_manager_ref(band_name)

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

            slot_task = asyncio.create_task(slot_manager.acquire_free_slot(
                (subtask.session_id, subtask.subtask_id)))
            slot_id = subtask_info.slot_id = yield slot_task
            if subtask_info.cancelling:
                raise asyncio.CancelledError

            subtask_info.result.status = SubtaskStatus.running
            run_aiotask = asyncio.create_task(
                subtask_api.run_subtask_in_slot(band_name, slot_id, subtask))
            yield asyncio.shield(run_aiotask)
            if subtask_info.cancelling:
                raise asyncio.CancelledError
        except asyncio.CancelledError as ex:
            if slot_id is not None:
                cancel_task = asyncio.create_task(subtask_api.cancel_subtask_in_slot(band_name, slot_id))
                try:
                    yield asyncio.wait_for(asyncio.shield(cancel_task),
                                           subtask_info.kill_timeout)
                except asyncio.TimeoutError:
                    slot_manager_ref = await self._get_slot_manager_ref(subtask_info.band_name)
                    yield slot_manager_ref.kill_slot(slot_id)
                    # since slot is restored, it is not occupied by now
                    slot_id = None

                    try:
                        yield run_aiotask
                    except ServerClosed:
                        pass
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
            if subtask_info.result.status == SubtaskStatus.running:
                subtask_info.result = yield run_aiotask

            self._subtask_info.pop(subtask.subtask_id, None)
            if self._global_slot_ref is not None:
                # make sure slot is released before marking tasks as finished
                yield self._global_slot_ref.release_subtask_slots(
                    (self.address, band_name), subtask.session_id, subtask.subtask_id)
                logger.debug('Slot released for band %s after subtask %s',
                             band_name, subtask.subtask_id)

            task_api = await self._get_task_api(subtask_info.supervisor_address,
                                                subtask.session_id)
            yield task_api.set_subtask_result(subtask_info.result)

            if batch_quota_req:
                await quota_ref.release_quotas(list(batch_quota_req.keys()))
            if slot_id is not None:
                await slot_manager.release_free_slot(slot_id)

    async def run_subtask(self, subtask: Subtask, band_name: str,
                          supervisor_address: str):
        task = asyncio.create_task(self.ref().internal_run_subtask(subtask, band_name))
        self._subtask_info[subtask.subtask_id] = \
            SubtaskExecutionInfo(task, band_name, supervisor_address)
        return task

    async def cancel_subtask(self, subtask_id: str, kill_timeout: int = 5):
        subtask_info = self._subtask_info[subtask_id]
        if not subtask_info.cancelling:
            subtask_info.kill_timeout = kill_timeout
            subtask_info.cancelling = True

            subtask_info.aio_task.cancel()

        try:
            yield subtask_info.aio_task
        except asyncio.CancelledError:
            pass
