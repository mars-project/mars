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
import itertools
import logging
from abc import ABC
from collections import namedtuple
from typing import Any, Callable, Coroutine, Dict, Type
from urllib.parse import urlparse

from ....serialization import serialize, deserialize
from ....utils import implements, classproperty
from ....utils import lazy_import
from ...debug import debug_async_timeout
from ...errors import ServerClosed
from ..communication.base import Channel, ChannelType, Server, Client
from ..communication.core import register_client, register_server
from ..communication.errors import ChannelClosed

ray = lazy_import("ray")
logger = logging.getLogger(__name__)

ChannelID = namedtuple("ChannelID", ["local_address", "client_id", "channel_index", "dest_address"])


class RayChannelException(Exception):

    def __init__(self, exc_type, exc_value: BaseException, exc_traceback):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback


class RayChannelBase(Channel, ABC):
    """
    Channel for communications between ray processes.
    """
    __slots__ = '_channel_index', '_channel_id', '_in_queue', '_closed'

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
        self._channel_index = channel_index or next(self._channel_index_gen)
        self._channel_id = channel_id or ChannelID(
            local_address, _gen_client_id(), self._channel_index, dest_address)
        self._in_queue = asyncio.Queue()
        self._closed = asyncio.Event()

    @property
    def channel_id(self) -> ChannelID:
        return self._channel_id

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return ChannelType.ray

    @implements(Channel.close)
    async def close(self):
        self._closed.set()

    @property
    @implements(Channel.closed)
    def closed(self) -> bool:
        return self._closed.is_set()


class RayClientChannel(RayChannelBase):
    """
    A channel from ray driver/actor to ray actor. Use ray call reply for client channel recv.
    """
    __slots__ = '_peer_actor',

    def __init__(self,
                 dest_address: str = None,
                 channel_index: int = None,
                 channel_id: ChannelID = None,
                 compression=None):
        super().__init__(None, dest_address, channel_index, channel_id, compression)
        # ray actor should be created with the address as the name.
        self._peer_actor: 'ray.actor.ActorHandle' = ray.get_actor(dest_address)

    @implements(Channel.send)
    async def send(self, message: Any):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot send message')
        # Put ray object ref to queue
        await self._in_queue.put((message, self._peer_actor.__on_ray_recv__.remote(
            self.channel_id, serialize(message))))

    @implements(Channel.recv)
    async def recv(self):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot recv message')
        try:
            # Wait on ray object ref
            message, object_ref = await self._in_queue.get()
            with debug_async_timeout('ray_object_retrieval_timeout', 'Client sent message is %s', message):
                result = await object_ref
            if isinstance(result, RayChannelException):
                raise result.exc_value.with_traceback(result.exc_traceback)
            return deserialize(*result)
        except ray.exceptions.RayActorError:
            if not self._closed.is_set():
                # raise a EOFError as the SocketChannel does
                raise EOFError('Server may be closed')
        except (RuntimeError, ServerClosed) as e:  # pragma: no cover
            if not self._closed.is_set():
                raise e


class RayServerChannel(RayChannelBase):
    """
    A channel from ray actor to ray driver/actor. Since ray actor can't call ray driver,
    we use ray call reply for server channel send. Note that there can't be multiple
    channel message sends for one received message, or else it will be taken as next
    message's reply.
    """
    __slots__ = '_out_queue', '_msg_recv_counter', '_msg_sent_counter'

    def __init__(self,
                 local_address: str = None,
                 channel_index: int = None,
                 channel_id: ChannelID = None,
                 compression=None):
        super().__init__(local_address, None, channel_index, channel_id, compression)
        self._out_queue = asyncio.Queue()
        self._msg_recv_counter = 0
        self._msg_sent_counter = 0

    @implements(Channel.send)
    async def send(self, message: Any):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot send message')
        # Current process is ray actor, we use ray call reply to send message to ray driver/actor.
        # Not that we can only send once for every read message in channel, otherwise
        # it will be taken as other message's reply.
        await self._out_queue.put(serialize(message))
        self._msg_sent_counter += 1
        assert self._msg_sent_counter <= self._msg_recv_counter, \
            "RayServerChannel channel doesn't support send multiple replies for one message."

    @implements(Channel.recv)
    async def recv(self):
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed, cannot write message')
        try:
            return deserialize(*(await self._in_queue.get()))
        except RuntimeError:  # pragma: no cover
            if not self._closed.is_set():
                raise

    async def __on_ray_recv__(self, message):
        """This method will be invoked when current process is a ray actor rather than a ray driver"""
        self._msg_recv_counter += 1
        await self._in_queue.put(message)
        # Avoid hang when channel is closed after `self._out_queue.get()` is awaited.
        done, _ = await asyncio.wait([self._out_queue.get(), self._closed.wait()],
                                     return_when=asyncio.FIRST_COMPLETED)
        if self._closed.is_set():  # pragma: no cover
            raise ChannelClosed('Channel already closed')
        if done:
            return await done.pop()


@register_server
class RayServer(Server):
    __slots__ = '_closed', '_channels', '_tasks'

    scheme = 'ray'
    _server_instance = None
    _ray_actor_started = False

    def __init__(self, address, channel_handler: Callable[[Channel], Coroutine] = None):
        super().__init__(address, channel_handler)
        self._closed = asyncio.Event()
        self._channels: Dict[ChannelID, RayServerChannel] = dict()
        self._tasks: Dict[ChannelID, asyncio.Task] = dict()

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return RayClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.ray

    @classmethod
    def set_ray_actor_started(cls):
        cls._ray_actor_started = True

    @classmethod
    def is_ray_actor_started(cls):
        return cls._ray_actor_started

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "RayServer":
        if not RayServer.is_ray_actor_started():
            logger.warning('Current process is not a ray actor, the ray server '
                           'will not receive messages from clients.')
        assert RayServer._server_instance is None
        config = config.copy()
        address = config.pop('address')
        handle_channel = config.pop('handle_channel')
        if urlparse(address).scheme != RayServer.scheme:  # pragma: no cover
            raise ValueError(f'Address for RayServer '
                             f'should be starts with "ray://", '
                             f'got {address}')
        if config:  # pragma: no cover
            raise TypeError(f'Creating RayServer got unexpected '
                            f'arguments: {",".join(config)}')
        server = RayServer(address, handle_channel)
        RayServer._server_instance = server
        return server

    @classmethod
    def get_instance(cls):
        return cls._server_instance

    @classmethod
    def clear(cls):
        cls._server_instance = None
        cls._ray_actor_started = False

    @implements(Server.start)
    async def start(self):
        # nothing needs to do for ray server
        pass

    @implements(Server.join)
    async def join(self, timeout=None):
        wait_coro = self._closed.wait()
        try:
            await asyncio.wait_for(wait_coro, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):  # pragma: no cover
            pass

    @implements(Server.on_connected)
    async def on_connected(self, *args, **kwargs):
        channel = args[0]
        assert isinstance(channel, RayServerChannel)
        if kwargs:  # pragma: no cover
            raise TypeError(f'{type(self).__name__} got unexpected '
                            f'arguments: {",".join(kwargs)}')
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._closed.set()
        for task in self._tasks.values():
            task.cancel()
        self._tasks = dict()
        for channel in self._channels.values():
            await channel.close()
        self._channels = dict()
        self.clear()

    @property
    @implements(Server.stopped)
    def stopped(self) -> bool:
        return self._closed.is_set()

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        if self.stopped:
            raise ServerClosed(f'Remote server {self.address} closed, but got message {deserialize(*message)} '
                               f'from channel {channel_id}')
        channel = self._channels.get(channel_id)
        if not channel:
            _, _, peer_channel_index, peer_dest_address = channel_id
            channel = RayServerChannel(peer_dest_address, peer_channel_index, channel_id)
            self._channels[channel_id] = channel
            self._tasks[channel_id] = asyncio.create_task(self.on_connected(channel))
        return await channel.__on_ray_recv__(message)


@register_client
class RayClient(Client):
    __slots__ = ()

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
        client_channel = RayClientChannel(dest_address)
        client = RayClient(local_address, dest_address, client_channel)
        return client

    @implements(Client.close)
    async def close(self):
        await super().close()


def _gen_client_id():
    import uuid
    return uuid.uuid4().hex
