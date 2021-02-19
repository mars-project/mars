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
from typing import Any, Callable, Coroutine, Dict, Type

from .....utils import implements, classproperty
from .base import Channel, ChannelType, Server, Client
from .errors import ChannelClosed

DUMMY_ADDRESS = 'dummy://'


class DummyChannel(Channel):
    """
    Channel for communications in same process.
    """
    __slots__ = '_queue', '_closed'

    name = 'dummy'

    def __init__(self,
                queue: asyncio.Queue = None,
                local_address: str = None,
                dest_address: str = None,
                compression=None):
        super().__init__(local_address=local_address,
                         dest_address=dest_address,
                         compression=compression)
        if queue is None:
            queue = asyncio.Queue()
        self._queue = queue
        self._closed = asyncio.Event()

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return ChannelType.dummy

    @implements(Channel.send)
    async def send(self, message: Any):
        if self._closed.is_set():
            raise ChannelClosed('Channel already closed, cannot send message')
        # put message directly into queue
        await self._queue.put(message)

    @implements(Channel.recv)
    async def recv(self):
        if self._closed.is_set():
            raise ChannelClosed('Channel already closed, cannot write message')
        return await self._queue.get()

    @implements(Channel.close)
    async def close(self):
        self._closed.set()

    @property
    @implements(Channel.closed)
    def closed(self) -> bool:
        return self._closed.is_set()


class DummyServer(Server):
    __slots__ = '_closed',

    _instance = None

    def __init__(self,
                 address: str,
                 handle_channel: Callable[[Channel], Coroutine] = None):
        super().__init__(address, handle_channel)
        self._closed = asyncio.Event()

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return DummyClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.dummy

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "DummyServer":
        # DummyServer is singleton
        if DummyServer._instance is not None:
            return DummyServer._instance

        handle_channel = config.pop('handle_channel')
        server = DummyServer(DUMMY_ADDRESS, handle_channel)
        DummyServer._instance = server
        return server

    @implements(Server.start)
    async def start(self):
        # nothing needs to do for dummy server
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
        assert isinstance(channel, DummyChannel)
        if kwargs:  # pragma: no cover
            raise TypeError(f'{type(self).__name__} got unexpected '
                            f'arguments: {",".join(kwargs)}')
        await self.handle_channel(channel)

    @implements(Server.shutdown)
    async def shutdown(self):
        self._closed.set()

    @implements(Server.is_shutdown)
    def is_shutdown(self) -> bool:
        return self._closed.is_set()


class DummyClient(Client):
    __slots__ = ()

    _instance = None

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        if DummyClient._instance is not None:
            # DummyClient is singleton
            return DummyClient._instance

        if dest_address != DUMMY_ADDRESS:  # pragma: no cover
            raise ValueError(f'Destination address has to be "dummy://" '
                             f'for DummyClient, got {dest_address}')
        server = DummyServer.get_instance()
        if server is None:  # pragma: no cover
            raise RuntimeError('DummyServer needs to be created '
                               'first before DummyClient')

        channel = DummyChannel()
        conn_coro = server.on_connected(channel)
        asyncio.create_task(conn_coro)
        client = DummyClient(local_address, dest_address, channel)
        DummyClient._instance = client
        return client
