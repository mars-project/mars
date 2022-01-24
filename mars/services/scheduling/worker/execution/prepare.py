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
import operator
from collections import defaultdict
from typing import Dict, Union

from ..... import oscar as mo
from .....core.graph import DAG
from .....core.operand import Fetch, FetchShuffle
from .....lib.aio import alru_cache
from .....storage import StorageLevel
from .....utils import get_chunk_key_to_data_keys
from ....meta import MetaAPI
from ....storage import DataNotExist, StorageAPI
from ....subtask import Subtask
from ..quota import QuotaActor, QuotaInsufficientError
from .core import SubtaskExecutionInfo


class PrepareFastFailed(Exception):
    pass


class SubtaskPreparer:
    _storage_api: StorageAPI

    def __init__(self, address: str):
        self._address = address

    @alru_cache(cache_exceptions=False)
    async def _get_band_quota_ref(self, band: str) -> Union[mo.ActorRef, QuotaActor]:
        return await mo.actor_ref(QuotaActor.gen_uid(band), address=self._address)

    async def _collect_input_sizes(
        self,
        subtask: Subtask,
        supervisor_address: str,
        band_name: str,
        local_only: bool = False,
    ):
        graph = subtask.chunk_graph
        key_to_sizes = dict()

        if local_only and any(
            n.key for n in graph.iter_indep() if isinstance(n.op, FetchShuffle)
        ):
            raise DataNotExist

        fetch_keys = list(
            set(n.key for n in graph.iter_indep() if isinstance(n.op, Fetch))
        )
        if not fetch_keys:
            return key_to_sizes

        storage_api = await StorageAPI.create(
            subtask.session_id, address=self._address, band_name=band_name
        )
        data_infos = await storage_api.get_infos.batch(
            *(storage_api.get_infos.delay(k, error="ignore") for k in fetch_keys)
        )

        # compute memory quota size. when data located in shared memory, the cost
        # should be differences between deserialized memory cost and serialized cost,
        # otherwise we should take deserialized memory cost
        for key, infos in zip(fetch_keys, data_infos):
            if not infos:
                continue
            level = functools.reduce(operator.or_, (info.level for info in infos))
            if level & StorageLevel.MEMORY:
                mem_cost = max(0, infos[0].memory_size - infos[0].store_size)
            else:
                mem_cost = infos[0].memory_size
            key_to_sizes[key] = (infos[0].store_size, mem_cost)

        non_local_keys = list(set(fetch_keys) - set(key_to_sizes.keys()))
        if non_local_keys and local_only:
            raise DataNotExist

        if non_local_keys:
            meta_api = await MetaAPI.create(
                subtask.session_id, address=supervisor_address
            )
            fetch_metas = await meta_api.get_chunk_meta.batch(
                *(
                    meta_api.get_chunk_meta.delay(
                        k, fields=["memory_size", "store_size"]
                    )
                    for k in non_local_keys
                )
            )
            for key, meta in zip(non_local_keys, fetch_metas):
                key_to_sizes[key] = (meta["store_size"], meta["memory_size"])

        return key_to_sizes

    @classmethod
    def _estimate_sizes(cls, subtask: Subtask, input_sizes: Dict):
        size_context = dict(input_sizes.items())
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
        total_memory_cost = 0
        max_memory_cost = sum(calc_size for _, calc_size in size_context.values())
        while key_stack:
            key = key_stack.pop()
            op = key_to_ops[key][0]

            if not isinstance(op, Fetch):
                op.estimate_size(size_context, op)

            calc_cost = sum(size_context[out.key][1] for out in op.outputs)
            total_memory_cost += calc_cost
            max_memory_cost = max(total_memory_cost, max_memory_cost)

            if not isinstance(op, Fetch):
                # when calculation result is stored, memory cost of calculation
                #  can be replaced with result memory cost
                result_cost = sum(size_context[out.key][0] for out in op.outputs)
                total_memory_cost += result_cost - calc_cost

            visited_op_keys.add(key)

            for succ_op_key in op_key_graph.iter_successors(key):
                pred_ref_count[succ_op_key] -= 1
                if pred_ref_count[succ_op_key] == 0:
                    key_stack.append(succ_op_key)

            for pred_op_key in op_key_graph.iter_predecessors(key):
                succ_ref_count[pred_op_key] -= 1
                if succ_ref_count[pred_op_key] == 0:
                    pred_op = key_to_ops[pred_op_key][0]
                    # when clearing fetches, subtract memory size, otherwise subtract store size
                    account_idx = 1 if isinstance(pred_op, Fetch) else 0
                    pop_result_cost = sum(
                        size_context.pop(out.key, (0, 0))[account_idx]
                        for out in key_to_ops[pred_op_key][0].outputs
                    )
                    total_memory_cost -= pop_result_cost
        return sum(t[0] for t in size_context.values()), max_memory_cost

    async def _prepare_input_data(self, subtask: Subtask, band_name: str):
        queries = []
        shuffle_queries = []
        storage_api = await StorageAPI.create(
            subtask.session_id, address=self._address, band_name=band_name
        )
        pure_dep_keys = set()
        chunk_key_to_data_keys = get_chunk_key_to_data_keys(subtask.chunk_graph)
        for n in subtask.chunk_graph:
            pure_dep_keys.update(
                inp.key
                for inp, pure_dep in zip(n.inputs, n.op.pure_depends)
                if pure_dep
            )
        for chunk in subtask.chunk_graph:
            if chunk.op.gpu:  # pragma: no cover
                to_fetch_band = band_name
            else:
                to_fetch_band = "numa-0"
            if isinstance(chunk.op, Fetch):
                queries.append(
                    storage_api.fetch.delay(chunk.key, band_name=to_fetch_band)
                )
            elif isinstance(chunk.op, FetchShuffle):
                for key in chunk_key_to_data_keys[chunk.key]:
                    shuffle_queries.append(
                        storage_api.fetch.delay(
                            key, band_name=to_fetch_band, error="ignore"
                        )
                    )
        if queries:
            await storage_api.fetch.batch(*queries)
        if shuffle_queries:
            # TODO(hks): The batch method doesn't accept different error arguments,
            #  combine them when it can.
            await storage_api.fetch.batch(*shuffle_queries)

    async def run(self, subtask_info: SubtaskExecutionInfo, fail_fast: bool = False):
        batch_quota_req = None
        quota_ref = await self._get_band_quota_ref(subtask_info.band_name)
        try:
            subtask = subtask_info.subtask
            try:
                input_sizes = await self._collect_input_sizes(
                    subtask,
                    subtask_info.supervisor_address,
                    subtask_info.band_name,
                    local_only=fail_fast,
                )
            except DataNotExist:
                raise PrepareFastFailed from None

            _store_size, calc_size = await asyncio.to_thread(
                self._estimate_sizes, subtask, input_sizes
            )

            try:
                insufficient_quota = "raise" if fail_fast else "enqueue"
                batch_quota_req = {(subtask.session_id, subtask.subtask_id): calc_size}
                await quota_ref.request_batch_quota(
                    batch_quota_req, insufficient=insufficient_quota
                )
                subtask_info.quota_request = batch_quota_req
            except QuotaInsufficientError:
                raise PrepareFastFailed from None

            await self._prepare_input_data(subtask_info.subtask, subtask_info.band_name)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            if batch_quota_req is not None:
                await quota_ref.release_quotas(tuple(batch_quota_req.keys()))
            raise
