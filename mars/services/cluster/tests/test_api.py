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
from mars.services.cluster.api import MockClusterAPI, WebClusterAPI
from mars.services.cluster.api.web import web_handlers
from mars.services.web.supervisor import start as start_web
from mars.utils import get_next_port


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
    await asyncio.sleep(0.1)
    nodes_info = await api.get_nodes_info(nodes=[pool_addr], state=True)
    assert pool_addr in nodes_info
    assert 'custom_key' in nodes_info[pool_addr]['state']

    await api.set_band_resource('numa-0', {'custom_usage': 0.1})
    await asyncio.sleep(0.1)
    nodes_info = await api.get_nodes_info(nodes=[pool_addr], resource=True)
    assert pool_addr in nodes_info
    assert 'custom_usage' in nodes_info[pool_addr]['resource']['numa-0']

    bands = await api.get_all_bands()
    assert (pool_addr, 'numa-0') in bands

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.get_supervisors(watch=True), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.get_supervisor_refs(
            [TestActor.default_uid()], watch=True), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.watch_nodes(NodeRole.WORKER), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(api.get_all_bands(watch=True), timeout=0.1)


@pytest.mark.asyncio
async def test_web_api(actor_pool):
    pool_addr = actor_pool.external_address
    api = await MockClusterAPI.create(pool_addr, upload_interval=0.1)

    web_config = {
        'web': {
            'host': '127.0.0.1',
            'port': get_next_port(),
            'web_handlers': web_handlers,
        }
    }
    await start_web(web_config, pool_addr)

    web_api = WebClusterAPI(f'http://127.0.0.1:{web_config["web"]["port"]}')
    assert await web_api.get_supervisors() == []

    await api.set_state_value('custom_key', {'key': 'value'})
    await asyncio.sleep(0.1)
    nodes_info = await web_api.get_nodes_info(nodes=[pool_addr], state=True)
    assert pool_addr in nodes_info
    assert 'custom_key' in nodes_info[pool_addr]['state']

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(web_api.watch_nodes(NodeRole.WORKER), timeout=0.1)
