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
import concurrent.futures as futures
import weakref
from typing import Any, Callable, Coroutine, Dict, Type
from urllib.parse import urlparse

from ....utils import implements, classproperty, abc_type_require_weakref_slot
from ...errors import ServerClosed
from .base import Channel, ChannelType, Server, Client
from .core import register_client, register_server
from .errors import ChannelClosed

DEFAULT_DUMMY_ADDRESS = 'dummy://0'


class DummyChannel(Channel):
    """
    Channel for communications in same process.
    """
    __slots__ = '_in_queue', '_out_queue', '_closed'

    name = 'dummy'

    def __init__(self,
                 in_queue: asyncio.Queue,
                 out_queue: asyncio.Queue,
                 closed: asyncio.Event,
                 local_address: str = None,
                 dest_address: str = None,
                 compression=None):
        super().__init__(local_address=local_address,
                         dest_address=dest_address,
                         compression=compression)
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._closed = closed

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
    __slots__ = ('_closed', '_channels', '_tasks') + ('__weakref__',) \
        if abc_type_require_weakref_slot else tuple()

    _address_to_instances: Dict[str, "DummyServer"] = weakref.WeakValueDictionary()
    scheme = 'dummy'

    def __init__(self,
                 address: str,
                 channel_handler: Callable[[Channel], Coroutine] = None):
        super().__init__(address, channel_handler)
        self._closed = asyncio.Event()
        self._channels = []
        self._tasks = []

    @classmethod
    def get_instance(cls, address: str):
        return cls._address_to_instances[address]

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
        address = config.pop('address', DEFAULT_DUMMY_ADDRESS)
        handle_channel = config.pop('handle_channel')
        if urlparse(address).scheme != DummyServer.scheme:  # pragma: no cover
            raise ValueError(f'Address for DummyServer '
                             f'should be starts with "dummy://", '
                             f'got {address}')
        if config:  # pragma: no cover
            raise TypeError(f'Creating DummyServer got unexpected '
                            f'arguments: {",".join(config)}')
        try:
            server = DummyServer.get_instance(address)
            if server.stopped:
                raise KeyError('server closed')
        except KeyError:
            server = DummyServer(address, handle_channel)
            DummyServer._address_to_instances[address] = server
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
        if self._closed.is_set():  # pragma: no cover
            raise ServerClosed('Dummy server already closed')

        channel = args[0]
        assert isinstance(channel, DummyChannel)
        if kwargs:  # pragma: no cover
            raise TypeError(f'{type(self).__name__} got unexpected '
                            f'arguments: {",".join(kwargs)}')
        self._channels.append(channel)
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._closed.set()
        _ = [t.cancel() for t in self._tasks]
        await asyncio.gather(
            *(channel.close() for channel in self._channels))

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

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        if urlparse(dest_address).scheme != DummyServer.scheme:  # pragma: no cover
            raise ValueError(f'Destination address should start with "dummy://" '
                             f'for DummyClient, got {dest_address}')
        server = DummyServer.get_instance(dest_address)
        if server is None:  # pragma: no cover
            raise RuntimeError('DummyServer needs to be created '
                               'first before DummyClient')
        if server.stopped:  # pragma: no cover
            raise ConnectionError('Dummy server closed')

        q1, q2 = asyncio.Queue(), asyncio.Queue()
        closed = asyncio.Event()
        client_channel = DummyChannel(q1, q2, closed,
                                      local_address=local_address)
        server_channel = DummyChannel(q2, q1, closed,
                                      dest_address=local_address)

        conn_coro = server.on_connected(server_channel)
        task = asyncio.create_task(conn_coro)
        client = DummyClient(local_address, dest_address, client_channel)
        client._task = task
        server._tasks.append(task)
        return client

    @implements(Client.close)
    async def close(self):
        await super().close()
        self._task.cancel()
        self._task = None
