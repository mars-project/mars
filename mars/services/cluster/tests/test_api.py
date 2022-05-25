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

from .... import oscar as mo
from ....utils import get_next_port
from ... import NodeRole
from ...web.supervisor import WebSupervisorService
from ..api import ClusterAPI, MockClusterAPI, WebClusterAPI
from ..api.web import web_handlers
from ..core import NodeStatus


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool


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

    assert pool_addr in await api.get_supervisors_by_keys(["test_mock"])

    await mo.create_actor(TestActor, uid=TestActor.default_uid(), address=pool_addr)
    assert (await api.get_supervisor_refs([TestActor.default_uid()]))[
        0
    ].address == pool_addr

    bands = await api.get_all_bands()
    assert (pool_addr, "numa-0") in bands

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(api.watch_supervisors()), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            wait_async_gen(api.watch_supervisor_refs([TestActor.default_uid()])),
            timeout=0.1,
        )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            wait_async_gen(
                api.watch_nodes(NodeRole.WORKER, statuses={NodeStatus.READY})
            ),
            timeout=0.1,
        )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            wait_async_gen(api.watch_all_bands(statuses={NodeStatus.READY})),
            timeout=0.1,
        )
    with pytest.raises(NotImplementedError):
        await api.request_worker(timeout=1)
    with pytest.raises(NotImplementedError):
        await api.release_worker("127.0.0.1:1234")

    await api.set_node_status(pool_addr, NodeRole.WORKER, NodeStatus.STOPPING)
    assert {} == await api.get_all_bands()
    assert {} == await api.get_nodes_info(role=NodeRole.WORKER)
    bands = await api.get_all_bands(exclude_statuses={NodeStatus.STOPPED})
    assert (pool_addr, "numa-0") in bands
    assert pool_addr in await api.get_nodes_info(
        role=NodeRole.WORKER, exclude_statuses={NodeStatus.STOPPED}
    )

    await MockClusterAPI.cleanup(pool_addr)


@pytest.mark.asyncio
async def test_web_api(actor_pool):
    pool_addr = actor_pool.external_address
    await MockClusterAPI.create(pool_addr, upload_interval=0.1)

    web_config = {
        "web": {
            "host": "127.0.0.1",
            "port": get_next_port(),
            "web_handlers": web_handlers,
        }
    }
    web_service = WebSupervisorService(web_config, pool_addr)
    await web_service.start()

    web_api = WebClusterAPI(f'http://127.0.0.1:{web_config["web"]["port"]}')
    assert await web_api.get_supervisors() == [pool_addr]

    assert len(await web_api.get_all_bands(statuses={NodeStatus.READY})) > 0
    nodes = await web_api.get_nodes_info(
        role=NodeRole.WORKER, statuses={NodeStatus.READY}
    )
    assert len(nodes) > 0

    from .... import __version__ as mars_version

    assert await web_api.get_mars_versions() == [mars_version]

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(web_api.watch_supervisors()), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            wait_async_gen(web_api.watch_nodes(NodeRole.WORKER)), timeout=0.1
        )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_async_gen(web_api.watch_all_bands()), timeout=0.1)

    proc_info = await web_api.get_node_pool_configs(pool_addr)
    assert len(proc_info) > 0
    stacks = await web_api.get_node_thread_stacks(pool_addr)
    assert len(stacks) > 0

    await MockClusterAPI.cleanup(pool_addr)


@pytest.mark.asyncio
async def test_no_supervisor(actor_pool):
    pool_addr = actor_pool.external_address

    from ..supervisor.locator import SupervisorPeerLocatorActor
    from ..uploader import NodeInfoUploaderActor

    await mo.create_actor(
        SupervisorPeerLocatorActor,
        "fixed",
        [],
        uid=SupervisorPeerLocatorActor.default_uid(),
        address=pool_addr,
    )
    await mo.create_actor(
        NodeInfoUploaderActor,
        NodeRole.WORKER,
        interval=1,
        band_to_resource=None,
        use_gpu=False,
        uid=NodeInfoUploaderActor.default_uid(),
        address=pool_addr,
    )
    api = await ClusterAPI.create(address=pool_addr)
    with pytest.raises(mo.ActorNotExist):
        await api.get_supervisor_refs(["KEY"])
