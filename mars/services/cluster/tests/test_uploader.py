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
from mars.services import NodeRole
from mars.services.cluster.locator import SupervisorLocatorActor
from mars.services.cluster.supervisor.node_info import NodeInfoCollectorActor
from mars.services.cluster.uploader import NodeInfoUploaderActor


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)
    await pool.start()
    yield pool
    await pool.stop()


@pytest.mark.asyncio
async def test_uploader(actor_pool):
    pool_addr = actor_pool.external_address
    await mo.create_actor(
        SupervisorLocatorActor, 'fixed', pool_addr,
        uid=SupervisorLocatorActor.default_uid(), address=pool_addr)
    collector_ref = await mo.create_actor(
        NodeInfoCollectorActor, timeout=0.5, check_interval=0.1,
        uid=NodeInfoCollectorActor.default_uid(), address=pool_addr
    )
    uploader_ref = await mo.create_actor(
        NodeInfoUploaderActor, role=NodeRole.WORKER, interval=0.1,
        uid=NodeInfoUploaderActor.default_uid(), address=pool_addr
    )

    await uploader_ref.set_state_value.tell('custom_state', {'key': 'val'})
    await asyncio.sleep(0.2)

    # test empty result
    result = await collector_ref.get_nodes_info(role=NodeRole.WORKER)
    assert pool_addr in result
    assert all(result[pool_addr].get(k) is None
               for k in ('env', 'resource', 'state'))

    result = await collector_ref.get_nodes_info(
        role=NodeRole.WORKER, env=True, resource=True, state=True)
    assert pool_addr in result
    assert all(result[pool_addr].get(k) is not None
               for k in ('env', 'resource', 'state'))
    assert result[pool_addr]['state']['custom_state']

    watch_task = asyncio.create_task(collector_ref.watch_nodes(NodeRole.WORKER))

    await uploader_ref.destroy()
    assert not await watch_task

    await collector_ref.destroy()
