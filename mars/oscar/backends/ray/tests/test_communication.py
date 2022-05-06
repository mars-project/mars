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
import inspect

import pytest

from .....tests.core import require_ray
from .....utils import ensure_coverage, lazy_import
from ....core import ActorRef
from ....errors import ServerClosed
from ...communication.base import ChannelType
from ...message import SendMessage
from ..communication import ChannelID, Channel, RayServer, RayClient, msg_to_simple_str

ray = lazy_import("ray")


class ServerActor:
    def __new__(cls, *args, **kwargs):
        ensure_coverage()
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, address):
        self.address = address
        self.server = None

    async def start(self):
        RayServer.set_ray_actor_started()
        self.server = await RayServer.create(
            {"address": self.address, "handle_channel": self.on_new_channel}
        )

    async def on_new_channel(self, channel: Channel):
        while True:
            try:
                message = await channel.recv()
                await channel.send(message)
            except EOFError:
                # no data to read, check channel
                await channel.close()
                return
            await asyncio.sleep(0.1)

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        return await self.server.__on_ray_recv__(channel_id, message)

    async def server(self, method_name, *args, **kwargs):
        result = getattr(self.server, method_name)(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = await result
        return result


class ServerCallActor(ServerActor):
    def __init__(self, address):
        super().__init__(address)

    async def check(self, dest_address, x):
        client = await RayClient.connect(dest_address, self.address)
        await client.send(x)
        return await client.recv() == x


@require_ray
@pytest.mark.asyncio
async def test_driver_to_actor_channel(ray_start_regular):
    dest_address = "ray://test_cluster/0/0"
    server_actor = (
        ray.remote(ServerActor).options(name=dest_address).remote(dest_address)
    )
    await server_actor.start.remote()
    client = await RayClient.connect(dest_address, None)
    assert client.channel_type == ChannelType.ray
    for i in range(10):
        await client.send(i)
        assert await client.recv() == i
    await server_actor.server.remote("stop")
    with pytest.raises(ServerClosed):
        await client.send(1)
        await client.recv()


@require_ray
@pytest.mark.asyncio
async def test_actor_to_actor_channel(ray_start_regular):
    server1_address, server2_address = (
        "ray://test_cluster/0/0",
        "ray://test_cluster/0/1",
    )
    server_actor1 = (
        ray.remote(ServerCallActor)
        .options(name=server1_address)
        .remote(server1_address)
    )
    server_actor2 = (
        ray.remote(ServerCallActor)
        .options(name=server2_address)
        .remote(server2_address)
    )
    await server_actor1.start.remote()
    await server_actor2.start.remote()
    for client in [
        await RayClient.connect(addr, None)
        for addr in [server1_address, server2_address]
    ]:
        for i in range(10):
            await client.send(i)
            assert await client.recv() == i
    for i in range(10):
        assert await server_actor1.check.remote(server2_address, i)
        assert await server_actor2.check.remote(server1_address, i)


@require_ray
@pytest.mark.asyncio
async def test_msg_to_simple_str(ray_start_regular):
    assert msg_to_simple_str(1) == "1"
    assert msg_to_simple_str(True) == "True"
    assert msg_to_simple_str("a") == "a"
    assert msg_to_simple_str([1, 2]) == "List<1, 2...2>"
    assert msg_to_simple_str({1, 2}) == "Set<1, 2...2>"
    assert msg_to_simple_str((1, 2.0, False)) == "Tuple<1, 2.0, False...3>"
    assert msg_to_simple_str({"a": [1, 2]}) == "Dict<k=a, v=List<1, 2...2>...1>"
    assert (
        msg_to_simple_str(
            SendMessage(
                message_id=b"abc", actor_ref=ActorRef("addr", b"id"), content="abc"
            )
        )
        == "SendMessage(actor_ref=ActorRef(uid=b'id', address='addr'), content=abc)"
    )
