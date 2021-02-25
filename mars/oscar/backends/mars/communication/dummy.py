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
from .core import register_client, register_server
from .errors import ChannelClosed

DUMMY_ADDRESS = 'dummy://'


class DummyChannel(Channel):
    """
    Channel for communications in same process.
    """
    __slots__ = '_in_queue', '_out_queue', '_closed'

    name = 'dummy'

    def __init__(self,
                 in_queue: asyncio.Queue,
                 out_queue: asyncio.Queue,
                 local_address: str = None,
                 dest_address: str = None,
                 compression=None):
        super().__init__(local_address=local_address,
                         dest_address=dest_address,
                         compression=compression)
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._closed = asyncio.Event()

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return ChannelType.local

    @implements(Channel.send)
    async def send(self, message: Any):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot send message')
        # put message directly into queue
        await self._out_queue.put(message)

    @implements(Channel.recv)
    async def recv(self):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot write message')
        try:
            return await self._in_queue.get()
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


@register_server
class DummyServer(Server):
    __slots__ = '_closed',

    _instance = None
    scheme = 'dummy'

    def __init__(self,
                 address: str,
                 channel_handler: Callable[[Channel], Coroutine] = None):
        super().__init__(address, channel_handler)
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
        return ChannelType.local

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "DummyServer":
        config = config.copy()
        address = config.pop('address', DUMMY_ADDRESS)
        handle_channel = config.pop('handle_channel')
        if address != DUMMY_ADDRESS:  # pragma: no cover
            raise ValueError(f'Address for DummyServer '
                             f'should be {DUMMY_ADDRESS}, '
                             f'got {address}')
        if config:  # pragma: no cover
            raise TypeError(f'Creating DummyServer got unexpected '
                            f'arguments: {",".join(config)}')

        # DummyServer is singleton
        if DummyServer._instance is not None:
            return DummyServer._instance

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
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._closed.set()
        DummyServer._instance = None

    @property
    @implements(Server.stopped)
    def stopped(self) -> bool:
        return self._closed.is_set()


@register_client
class DummyClient(Client):
    __slots__ = '_task',

    scheme = DummyServer.scheme

    def __init__(self,
                 local_address: str,
                 dest_address: str,
                 channel: Channel):
        super().__init__(local_address,
                         dest_address,
                         channel)
        self._task = None

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        if dest_address != DUMMY_ADDRESS:  # pragma: no cover
            raise ValueError(f'Destination address has to be "dummy://" '
                             f'for DummyClient, got {dest_address}')
        server = DummyServer.get_instance()
        if server is None:  # pragma: no cover
            raise RuntimeError('DummyServer needs to be created '
                               'first before DummyClient')

        q1, q2 = asyncio.Queue(), asyncio.Queue()
        client_channel = DummyChannel(q1, q2)
        server_channel = DummyChannel(q2, q1)

        conn_coro = server.on_connected(server_channel)
        task = asyncio.create_task(conn_coro)
        client = DummyClient(local_address, dest_address, client_channel)
        client._task = task
        return client

    @implements(Client.close)
    async def close(self):
        await super().close()
        DummyClient._instance = None
        self._task.cancel()
        self._task = None
