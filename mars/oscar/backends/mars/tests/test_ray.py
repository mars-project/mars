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
import sys
import time
try:
    import mock
except ImportError:
    from unittest import mock

import pytest
import os

from mars.utils import get_next_port
from mars.oscar import Actor, kill_actor
from mars.oscar.context import get_context
from mars.oscar.backends.mars import create_actor_pool
from mars.oscar.backends.mars.allocate_strategy import \
    AddressSpecified, IdleLabel, MainPool, RandomSubPool, ProcessIndex
from mars.oscar.backends.mars.config import ActorPoolConfig
from mars.oscar.backends.mars.message import new_message_id, \
    CreateActorMessage, DestroyActorMessage, HasActorMessage, \
    ActorRefMessage, SendMessage, TellMessage, ControlMessage, \
    CancelMessage, ControlMessageType, MessageType
from mars.oscar.backends.mars.pool import SubActorPool, MainActorPool
from mars.oscar.backends.mars.ray import create_cluster, NodeResourceSpec, pg_bundle_to_address,\
    address_to_placement_info, RayMainPool
from mars.oscar.backends.mars.router import Router
from mars.oscar.errors import NoIdleSlot, ActorNotExist, ServerClosed
from mars.oscar.utils import create_actor_ref
from mars.oscar.utils import create_actor_ref
import mars.oscar as mo
from mars.tests.core import ray


class DummyActor(mo.Actor):
    def __init__(self, index):
        super().__init__()
        self._index = index

    def getppid(self):
        return os.getppid()

    def index(self):
        return self._index


@pytest.fixture
def ray_cluster():
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    num_nodes = 3
    for i in range(num_nodes):
        remote_nodes.append(cluster.add_node(num_cpus=10))
        if len(remote_nodes) == 1:
            ray.init(address=cluster.address)
    yield

    ray.shutdown()


@pytest.mark.asyncio
async def test_ray_main_pool(ray_cluster):
    pg_name = "test_ray_main_pool"
    n_process = 3
    pg = ray.util.placement_group(name=pg_name,
                                  bundles=[NodeResourceSpec(n_process).to_bundle()],
                                  strategy="SPREAD")
    ray.get(pg.ready())
    address = pg_bundle_to_address(pg_name, 0, process_index=0)
    print(f"address {address}")
    # Hold actor_handle to avoid actor being freed.
    actor_handle = ray.remote(RayMainPool).options(
        name=address, placement_group=pg, placement_group_bundle_index=0).remote()
    ray.get(actor_handle.start.remote(address, n_process))
    actor_ref = await mo.create_actor(DummyActor, 0, address=address)
    assert await actor_ref.index() == 0


@pytest.mark.asyncio
async def test_ray_cluster(ray_cluster):
    cluster_manager = create_cluster("test_ray_cluster", [NodeResourceSpec(3), NodeResourceSpec(3)])
    mo.setup_cluster(cluster_manager.address_to_resources())
    print(f"cluster address and resources {cluster_manager.address_to_resources()}")

    # for index, addr in enumerate(cluster_manager.addresses()):
    #     actor_ref = await mo.create_actor(DummyActor, index, address=addr)
    #     actor_refs.append(actor_ref)

    actor_refs_group = []
    ppids = []
    for index, addresses in enumerate(cluster_manager.get_all_nodes_info()):
        new_refs = []
        ppids = set()
        actor_refs_group.append(new_refs)
        for addr in addresses[1:]:
            actor_ref = await mo.create_actor(DummyActor, index, address=addr)
            new_refs.append(new_refs)
            ppid = await actor_ref.getppid()
            ppids.add(ppid)
    assert len(ppids) == 1

    # results = []
    # actor_ref = await ctx.create_actor(
    #     TestActor, address=pool.external_address,
    #     allocate_strategy=ProcessIndex(1))

if __name__ == '__main__':
    next(ray_cluster())
    import asyncio
    asyncio.run(test_ray_cluster())