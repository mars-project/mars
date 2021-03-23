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
from mars.services.cluster.api import MockClusterAPI


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)
    await pool.start()
    yield pool
    await pool.stop()


class TestActor(mo.Actor):
    pass


@pytest.mark.asyncio
async def test_api(actor_pool):
    pool_addr = actor_pool.external_address
    api = await MockClusterAPI.create(pool_addr, upload_interval=0.1)

    assert await api.get_supervisors() == [pool_addr]

    assert pool_addr in await api.get_supervisors_by_keys(['test_mock'], False)

    await mo.create_actor(TestActor, uid=TestActor.default_uid(), address=pool_addr)
    assert (await api.get_supervisor_refs([TestActor.default_uid()]))[0].address == pool_addr

    await api.set_state_value('custom_key', {'key': 'value'})
    await asyncio.sleep(0.2)
    nodes_info = await api.get_nodes_info(nodes=[pool_addr], state=True)
    assert pool_addr in nodes_info
    assert 'custom_key' in nodes_info[pool_addr]['state']

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.get_supervisors(watch=True), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.get_supervisor_refs(
            [TestActor.default_uid()], watch=True), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.watch_nodes(NodeRole.WORKER), timeout=0.1)
