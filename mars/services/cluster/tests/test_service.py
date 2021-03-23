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

import pytest

import mars.oscar as mo
from mars.services import start_services, NodeRole
from mars.services.cluster import ClusterAPI


@pytest.fixture
async def actor_pools():
    async def start_pool():
        pool = await mo.create_actor_pool('127.0.0.1', n_process=0)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(
        start_pool(), start_pool()
    )
    yield sv_pool, worker_pool
    await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.asyncio
async def test_cluster_service(actor_pools):
    sv_pool, worker_pool = actor_pools

    config = {
        "services": ["cluster"],
        "cluster": {
            "backend": "fixed",
            "master_address": sv_pool.external_address,
        }
    }
    await start_services(
        NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(
        NodeRole.WORKER, config, address=worker_pool.external_address)

    api = await ClusterAPI.create(sv_pool.external_address)
    assert next(iter(await api.get_nodes_info(role=NodeRole.SUPERVISOR))) \
           == sv_pool.external_address
    assert next(iter(await api.get_nodes_info(role=NodeRole.WORKER))) \
           == worker_pool.external_address
