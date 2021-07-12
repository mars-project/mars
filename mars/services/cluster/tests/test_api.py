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

import pytest

import mars.oscar as mo
from mars._version import __version__ as mars_version
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
    __test__ = False


async def wait_async_gen(async_gen):
    async for _ in async_gen:
        pass


@pytest.mark.asyncio
async def test_api(actor_pool):
    pool_addr = actor_pool.external_address
    api = await MockClusterAPI.create(pool_addr, upload_interval=0.1)

    assert await api.get_supervisors() == [pool_addr]

    assert pool_addr in await api.get_supervisors_by_keys(['test_mock'])

    await mo.create_actor(TestActor, uid=TestActor.default_uid(), address=pool_addr)
    assert (await api.get_supervisor_refs([TestActor.default_uid()]))[0].address == pool_addr

    bands = await api.get_all_bands()
    assert (pool_addr, 'numa-0') in bands

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            api.watch_supervisors()), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            api.watch_supervisor_refs([TestActor.default_uid()])), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            api.watch_nodes(NodeRole.WORKER)), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            api.watch_all_bands()), timeout=0.1)


@pytest.mark.asyncio
async def test_web_api(actor_pool):
    pool_addr = actor_pool.external_address
    await MockClusterAPI.create(pool_addr, upload_interval=0.1)

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

    assert len(await web_api.get_all_bands()) > 0

    assert await web_api.get_mars_versions() == [mars_version]

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            web_api.watch_supervisors()), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            web_api.watch_nodes(NodeRole.WORKER)), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(
            web_api.watch_all_bands()), timeout=0.1)
