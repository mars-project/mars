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
from ... import NodeRole
from ..supervisor.locator import SupervisorPeerLocatorActor
from ..supervisor.node_info import NodeInfoCollectorActor
from ..uploader import NodeInfoUploaderActor


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_uploader(actor_pool):
    pool_addr = actor_pool.external_address
    await mo.create_actor(
        SupervisorPeerLocatorActor,
        "fixed",
        pool_addr,
        uid=SupervisorPeerLocatorActor.default_uid(),
        address=pool_addr,
    )
    node_info_ref = await mo.create_actor(
        NodeInfoCollectorActor,
        timeout=0.5,
        check_interval=0.1,
        uid=NodeInfoCollectorActor.default_uid(),
        address=pool_addr,
    )
    uploader_ref = await mo.create_actor(
        NodeInfoUploaderActor,
        role=NodeRole.WORKER,
        interval=0.1,
        uid=NodeInfoUploaderActor.default_uid(),
        address=pool_addr,
    )
    wait_ready_task = asyncio.create_task(uploader_ref.wait_node_ready())
    await uploader_ref.mark_node_ready()
    await asyncio.wait_for(wait_ready_task, timeout=0.1)

    # test empty result
    result = await node_info_ref.get_nodes_info(role=NodeRole.WORKER)
    assert pool_addr in result
    assert all(result[pool_addr].get(k) is None for k in ("env", "resource", "detail"))

    result = await node_info_ref.get_nodes_info(
        role=NodeRole.WORKER, env=True, resource=True, detail=True
    )
    assert pool_addr in result
    assert all(
        result[pool_addr].get(k) is not None for k in ("env", "resource", "detail")
    )

    async def watcher():
        version = None
        while True:
            version, infos = await node_info_ref.watch_nodes(
                NodeRole.WORKER, version=version
            )
            if not infos:
                break

    watch_task = asyncio.create_task(watcher())

    await uploader_ref.destroy()
    assert not await asyncio.wait_for(watch_task, timeout=5)

    await node_info_ref.destroy()
