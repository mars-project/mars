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
import tempfile
from typing import List

import pytest

from .... import oscar as mo
from ....utils import Timer
from ....tests.core import flaky
from ..core import NodeRole, NodeStatus
from ..supervisor.locator import SupervisorPeerLocatorActor
from ..supervisor.node_info import NodeInfoCollectorActor
from ..tests import backend
from ..worker.locator import WorkerSupervisorLocatorActor

del backend


class MockNodeInfoCollectorActor(mo.Actor):
    def __init__(self):
        self._node_infos = dict()
        self._version = 0

    def set_all_node_infos(self, node_infos):
        self._node_infos = node_infos

    def get_nodes_info(self, *args, **kwargs):
        return self._node_infos

    async def watch_nodes(self, *args, version=None, **kwargs):
        await asyncio.sleep(0.5)
        self._version += 1
        return self._version, self._node_infos

    def put_starting_nodes(self, nodes: List[str], role: NodeRole):
        for node in nodes:
            self._node_infos[node] = NodeStatus.STARTING


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        await mo.create_actor(
            MockNodeInfoCollectorActor,
            uid=NodeInfoCollectorActor.default_uid(),
            address=pool.external_address,
        )
        yield pool


@pytest.mark.asyncio
async def test_fixed_locator(actor_pool):
    addresses = ["1.2.3.4:1234", "1.2.3.4:1235", "1.2.3.4:1236", "1.2.3.4:1237"]
    locator_ref = await mo.create_actor(
        SupervisorPeerLocatorActor,
        "fixed",
        ",".join(addresses),
        address=actor_pool.external_address,
    )

    assert await locator_ref.get_supervisor("mock_name") in addresses

    dbl_addrs = await locator_ref.get_supervisor("mock_name", 2)
    assert len(dbl_addrs) == 2
    assert all(addr in addresses for addr in dbl_addrs)

    with Timer() as timer:
        await locator_ref.wait_all_supervisors_ready()
    assert timer.duration < 0.1

    await mo.destroy_actor(locator_ref)


@pytest.fixture
def temp_address_file():
    with tempfile.TemporaryDirectory(prefix="mars-test") as dir_name:
        yield os.path.join(dir_name, "addresses")


@flaky(max_runs=3)
@pytest.mark.asyncio
async def test_supervisor_peer_locator(actor_pool, temp_address_file):
    addresses = ["1.2.3.4:1234", "1.2.3.4:1235", "1.2.3.4:1236", "1.2.3.4:1237"]
    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(addresses))

    locator_ref = await mo.create_actor(
        SupervisorPeerLocatorActor,
        "test",
        temp_address_file,
        uid=SupervisorPeerLocatorActor.default_uid(),
        address=actor_pool.external_address,
    )

    # test starting nodes filled
    info_ref = await mo.actor_ref(
        uid=NodeInfoCollectorActor.default_uid(), address=actor_pool.external_address
    )
    assert set(await info_ref.get_nodes_info()) == set(addresses)

    # test watch nodes changes
    version, result = await asyncio.wait_for(
        locator_ref.watch_supervisors_by_keys(["mock_name"]),
        timeout=30,
    )
    assert result[0] in addresses

    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(addresses[2:]))

    version, result = await asyncio.wait_for(
        locator_ref.watch_supervisors_by_keys(["mock_name"], version=version),
        timeout=30,
    )
    assert result[0] in addresses[2:]

    # test wait all supervisors ready
    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(f"{a},{idx % 2}" for idx, a in enumerate(addresses)))

    async def delay_read_fun():
        await asyncio.sleep(0.2)
        with open(temp_address_file, "w") as file_obj:
            file_obj.write(
                "\n".join(f"{a},{(idx + 1) % 2}" for idx, a in enumerate(addresses))
            )
        await asyncio.sleep(0.5)
        with open(temp_address_file, "w") as file_obj:
            file_obj.write("\n".join(addresses))

    asyncio.create_task(delay_read_fun())

    with Timer() as timer:
        await asyncio.wait_for(locator_ref.wait_all_supervisors_ready(), timeout=30)
    assert timer.duration > 0.4

    await mo.destroy_actor(locator_ref)


@flaky(max_runs=3)
@pytest.mark.asyncio
async def test_worker_supervisor_locator(actor_pool, temp_address_file):
    addresses = [actor_pool.external_address]
    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(addresses))

    locator_ref = await mo.create_actor(
        WorkerSupervisorLocatorActor,
        "test",
        temp_address_file,
        uid=WorkerSupervisorLocatorActor.default_uid(),
        address=actor_pool.external_address,
    )

    info_ref = await mo.actor_ref(
        uid=NodeInfoCollectorActor.default_uid(), address=actor_pool.external_address
    )
    await info_ref.set_all_node_infos({actor_pool.external_address: NodeStatus.READY})

    # test watch nodes changes
    supervisors = await locator_ref.get_supervisors(filter_ready=False)
    assert supervisors == addresses
    version, result = await asyncio.wait_for(
        locator_ref.watch_supervisors_by_keys(["mock_name"]),
        timeout=30,
    )
    assert result[0] in addresses

    # test watch without NodeInfoCollectorActor
    await info_ref.destroy()

    addresses = ["localhost:1234", "localhost:1235"]
    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(addresses))
    version, result = await asyncio.wait_for(
        locator_ref.watch_supervisors_by_keys(["mock_name"], version=version),
        timeout=30,
    )
    assert result[0] in addresses

    # test watch when NodeInfoCollectorActor is created again
    info_ref = await mo.create_actor(
        MockNodeInfoCollectorActor,
        uid=NodeInfoCollectorActor.default_uid(),
        address=actor_pool.external_address,
    )
    await info_ref.set_all_node_infos({actor_pool.external_address: NodeStatus.READY})

    addresses = [actor_pool.external_address]
    with open(temp_address_file, "w") as file_obj:
        file_obj.write("\n".join(addresses))

    version, result = await asyncio.wait_for(
        locator_ref.watch_supervisors_by_keys(["mock_name"], version=version),
        timeout=30,
    )
    assert result[0] in addresses
