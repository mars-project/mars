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
import concurrent.futures as futures
from enum import Enum
from collections import namedtuple
import itertools
from typing import Any, Callable, Coroutine, Dict, Type
from urllib.parse import urlparse

from .....utils import implements, classproperty
from .base import Channel, ChannelType, Server, Client
from .core import register_client, register_server
from .errors import ChannelClosed
from .....utils import lazy_import

ray = lazy_import("ray")

DEFAULT_DUMMY_RAY_ADDRESS = 'ray://0'


class MessageType(Enum):
    SEND = 0
    RECV = 1


ChannelID = namedtuple("ChannelID", ["local_address", "dest_address", "channel_index"])


class RayChannel(Channel):
    """
    Channel for communications between ray actors.
    """
    __slots__ = '_in_queue', '_out_queue', '_closed'

    name = 'ray'
    _channel_index_gen = itertools.count()

    def __init__(self,
                 local_address: str = None,
                 dest_address: str = None,
                 channel_index: int = None,
                 channel_id: ChannelID = None,
                 compression=None):
        super().__init__(local_address=local_address,
                         dest_address=dest_address,
                         compression=compression)
        self._channel_index = channel_index or next(RayChannel.next_channel_index())
        self._channel_id = channel_id or ChannelID(local_address, dest_address, self._channel_index)
        # ray actor should be created with the address as the name.
        self.peer_actor: 'ray.actor.ActorHandle' = ray.get_actor(dest_address)
        self._queue = asyncio.Queue()
        self._closed = asyncio.Event()

    @classmethod
    def next_channel_index(cls):
        return next(cls._channel_index_gen)

    @property
    def channel_id(self) -> ChannelID:
        return self._channel_id

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return ChannelType.ray

    @implements(Channel.send)
    async def send(self, message: Any):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot send message')
        await self.peer_actor.__on_ray_recv__.remote(self.channel_id, message)

    @implements(Channel.recv)
    async def recv(self):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot write message')
        try:
            return await self._queue.get()
        except RuntimeError:
            if self._closed.is_set():
                pass

    @implements(Channel.close)
    async def close(self):
        self._closed.set()

    @property
    @implements(Channel.closed)
    def closed(self) -> bool:
        return self._closed.is_set()

    async def __on_ray_recv__(self, message):
        await self._queue.put(message)


@register_server
class RayServer(Server):
    __slots__ = '_closed',

    _instance: "RayServer" = None
    scheme = 'ray'

    def __init__(self, channel_handler: Callable[[Channel], Coroutine] = None):
        super().__init__(DEFAULT_DUMMY_RAY_ADDRESS, channel_handler)
        self._closed = asyncio.Event()
        self.channels: Dict[ChannelID, RayChannel] = dict()

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return RayClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.ray

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "RayServer":
        config = config.copy()
        address = config.pop('address', DEFAULT_DUMMY_RAY_ADDRESS)
        handle_channel = config.pop('handle_channel')
        if urlparse(address).scheme != RayServer.scheme:  # pragma: no cover
            raise ValueError(f'Address for RayServer '
                             f'should be starts with "ray://", '
                             f'got {address}')
        if config:  # pragma: no cover
            raise TypeError(f'Creating RayServer got unexpected '
                            f'arguments: {",".join(config)}')

        server = RayServer.get_instance()
        if not server:
            server = RayServer._instance = RayServer(address, handle_channel)
        return server

    @implements(Server.start)
    async def start(self):
        # nothing needs to do for ray server
        pass

    @implements(Server.join)
    async def join(self, timeout=None):
        wait_coro = self._closed.wait()
        try:
            await asyncio.wait_for(wait_coro, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):
            pass

    @implements(Server.on_connected)
    async def on_connected(self, *args, **kwargs):
        channel = args[0]
        assert isinstance(channel, RayChannel)
        if kwargs:  # pragma: no cover
            raise TypeError(f'{type(self).__name__} got unexpected '
                            f'arguments: {",".join(kwargs)}')
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._closed.set()

    @property
    @implements(Server.stopped)
    def stopped(self) -> bool:
        return self._closed.is_set()

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        channel = self.channels.get(channel_id, None)
        if not channel:
            peer_local_address, peer_dest_address, peer_channel_index = channel_id
            channel = RayChannel(peer_dest_address, peer_local_address, peer_channel_index, channel_id)
            self.channels[channel_id] = channel
            await self.on_connected(channel)
        await channel.__on_ray_recv__(message)

    def register_channel(self, channel: RayChannel):
        self.channels[channel.channel_id] = channel


@register_client
class RayClient(Client):
    __slots__ = '_task',

    scheme = RayServer.scheme

    def __init__(self,
                 local_address: str,
                 dest_address: str,
                 channel: Channel):
        super().__init__(local_address,
                         dest_address,
                         channel)

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        if urlparse(dest_address).scheme != RayServer.scheme:  # pragma: no cover
            raise ValueError(f'Destination address should start with "ray://" '
                             f'for RayClient, got {dest_address}')
        server = RayServer.get_instance()
        if server is None:  # pragma: no cover
            raise RuntimeError('RayServer needs to be created '
                               'first before RayClient')
        client_channel = RayChannel(local_address, dest_address)
        # The RayServer will push message to this channel's queue after it received
        # the message from the `dest_address` actor.
        server.register_channel(client_channel)
        client = RayClient(local_address, dest_address, client_channel)
        return client

    @implements(Client.close)
    async def close(self):
        await super().close()
