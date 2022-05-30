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
import os

import pytest

from .... import oscar as mo
from ....storage import StorageLevel
from ... import start_services, stop_services, NodeRole
from .. import ClusterAPI, WorkerSlotInfo, QuotaInfo, StorageInfo, DiskInfo


@pytest.fixture
async def actor_pools():
    async def start_pool():
        pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(start_pool(), start_pool())
    try:
        yield sv_pool, worker_pool
    finally:
        await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.asyncio
async def test_cluster_service(actor_pools):
    sv_pool, worker_pool = actor_pools

    config = {
        "services": ["cluster"],
        "cluster": {
            "backend": "fixed",
            "lookup_address": sv_pool.external_address,
        },
    }
    await start_services(NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(NodeRole.WORKER, config, address=worker_pool.external_address)

    sv_api = await ClusterAPI.create(sv_pool.external_address)
    worker_api = await ClusterAPI.create(worker_pool.external_address)

    await worker_api.set_band_quota_info(
        "numa-0", QuotaInfo(quota_size=1024, allocated_size=100, hold_size=100)
    )
    await worker_api.set_band_slot_infos(
        "numa-0",
        [
            WorkerSlotInfo(
                slot_id=0,
                session_id="test_session",
                subtask_id="test_subtask",
                processor_usage=1.0,
            )
        ],
    )
    await worker_api.set_band_storage_info(
        "numa-0",
        StorageInfo(storage_level=StorageLevel.MEMORY, total_size=1024, used_size=512),
    )
    curdir = os.path.dirname(os.path.abspath(__file__))
    await worker_api.set_node_disk_info([DiskInfo(path=curdir)])
    await asyncio.sleep(1.5)

    assert (
        next(iter(await sv_api.get_nodes_info(role=NodeRole.SUPERVISOR)))
        == sv_pool.external_address
    )
    worker_infos = await sv_api.get_nodes_info(role=NodeRole.WORKER, detail=True)
    assert worker_pool.external_address in worker_infos

    info_details = worker_infos[worker_pool.external_address]["detail"]
    assert len(info_details["disk"]["partitions"]) > 0
    assert len(info_details["slot"]) > 0
    assert len(info_details["quota"]) > 0
    assert len(info_details["storage"]) > 0

    await stop_services(NodeRole.WORKER, config, address=worker_pool.external_address)
    await stop_services(NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
