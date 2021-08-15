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
import os
import socket
import sys
import tempfile
from abc import ABCMeta
from asyncio import StreamReader, StreamWriter, AbstractServer
from functools import lru_cache
from hashlib import md5
from typing import Any, Dict, Callable, Coroutine, Type
from urllib.parse import urlparse

from ....serialization import AioSerializer, AioDeserializer, deserialize
from ....utils import implements, to_binary, classproperty
from .base import Channel, ChannelType, Server, Client
from .core import register_client, register_server
from .utils import read_buffers, write_buffers

_is_windows: bool = sys.platform.startswith('win')


class SocketChannel(Channel):
    __slots__ = 'reader', 'writer', '_channel_type', '_send_lock', '_recv_lock'

    name = 'socket'

    def __init__(self,
                 reader: StreamReader,
                 writer: StreamWriter,
                 local_address: str = None,
                 dest_address: str = None,
                 compression: int = None,
                 channel_type: ChannelType = None):
        super().__init__(local_address=local_address,
                         dest_address=dest_address,
                         compression=compression)
        self.reader = reader
        self.writer = writer
        self._channel_type = channel_type

        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()

    @property
    @implements(Channel.type)
    def type(self) -> ChannelType:
        return self._channel_type

    @implements(Channel.send)
    async def send(self, message: Any):
        # get buffers
        compress = self.compression or 0
        serializer = AioSerializer(message, compress=compress)
        buffers = await serializer.run()

        # write buffers
        write_buffers(self.writer, buffers)
        async with self._send_lock:
            # add lock, or when parallel send,
            # assertion error may be raised
            await self.writer.drain()

    @implements(Channel.recv)
    async def recv(self):
        deserializer = AioDeserializer(self.reader)
        async with self._recv_lock:
            header = await deserializer.get_header()
            buffers = await read_buffers(header, self.reader)
        return deserialize(header, buffers)

    @implements(Channel.close)
    async def close(self):
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except ConnectionResetError:  # pragma: no cover
            pass

    @property
    @implements(Channel.closed)
    def closed(self):
        return self.writer.is_closing()


class _BaseSocketServer(Server, metaclass=ABCMeta):
    __slots__ = '_aio_server', '_channels'

    def __init__(self,
                 address: str,
                 aio_server: AbstractServer,
                 channel_handler: Callable[[Channel], Coroutine] = None):
        super().__init__(address, channel_handler)
        # asyncio.Server
        self._aio_server = aio_server
        self._channels = []

    @implements(Server.start)
    async def start(self):
        await self._aio_server.start_serving()

    @implements(Server.join)
    async def join(self, timeout=None):
        if timeout is None:
            await self._aio_server.serve_forever()
        else:
            future = asyncio.create_task(self._aio_server.serve_forever())
            try:
                await asyncio.wait_for(future, timeout=timeout)
            except (futures.TimeoutError, asyncio.TimeoutError):
                future.cancel()

    @implements(Server.on_connected)
    async def on_connected(self, *args, **kwargs):
        reader, writer = args
        local_address = kwargs.pop('local_address', None)
        dest_address = kwargs.pop('dest_address', None)
        if kwargs:  # pragma: no cover
            raise TypeError(f'{type(self).__name__} got unexpected '
                            f'arguments: {",".join(kwargs)}')
        channel = SocketChannel(reader, writer,
                                local_address=local_address,
                                dest_address=dest_address,
                                channel_type=self.channel_type)
        self._channels.append(channel)
        # handle over channel to some handlers
        await self.channel_handler(channel)

    @implements(Server.stop)
    async def stop(self):
        self._aio_server.close()
        await self._aio_server.wait_closed()
        # close all channels
        await asyncio.gather(
            *(channel.close() for channel in self._channels
              if not channel.closed))

    @property
    @implements(Server.stopped)
    def stopped(self) -> bool:
        return not self._aio_server.is_serving()


@register_server
class SocketServer(_BaseSocketServer):
    __slots__ = 'host', 'port'

    scheme = None

    def __init__(self,
                 host: str,
                 port: int,
                 aio_server: AbstractServer,
                 channel_handler: Callable[[Channel], Coroutine] = None):
        address = f'{host}:{port}'
        super().__init__(address, aio_server,
                         channel_handler=channel_handler)
        self.host = host
        self.port = port

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return SocketClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.remote

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "Server":
        config = config.copy()
        if 'address' in config:
            address = config.pop('address')
            host, port = address.split(':', 1)
            port = int(port)
        else:
            host = config.pop('host')
            port = int(config.pop('port'))
        handle_channel = config.pop('handle_channel')
        if 'start_serving' not in config:
            config['start_serving'] = False

        async def handle_connection(reader, writer):
            # create a channel when client connected
            return await server.on_connected(reader, writer,
                                             local_address=server.address)

        aio_server = await asyncio.start_server(
            handle_connection, host=host, port=port, **config)

        if _is_windows:
            for sock in aio_server.sockets:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

        server = SocketServer(host, port, aio_server,
                              channel_handler=handle_channel)
        return server


@register_client
class SocketClient(Client):
    __slots__ = ()

    scheme = SocketServer.scheme

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        host, port = dest_address.split(':', 1)
        port = int(port)
        (reader, writer) = await asyncio.open_connection(
            host=host, port=port, **kwargs)
        channel = SocketChannel(reader, writer,
                                local_address=local_address,
                                dest_address=dest_address)
        return SocketClient(local_address, dest_address, channel)


TEMPDIR = tempfile.gettempdir()


@lru_cache(100)
def _gen_unix_socket_default_path(process_index):
    return f'{TEMPDIR}/mars/' \
           f'{md5(to_binary(str(process_index))).hexdigest()}'  # nosec


@register_server
class UnixSocketServer(_BaseSocketServer):
    __slots__ = 'process_index', 'path'

    scheme = 'unixsocket'

    def __init__(self,
                 process_index: int,
                 aio_server: AbstractServer,
                 path: str,
                 channel_handler: Callable[[Channel], Coroutine] = None):
        address = f'{self.scheme}:///{process_index}'
        super().__init__(address, aio_server,
                         channel_handler=channel_handler)
        self.process_index = process_index
        self.path = path

    @classproperty
    @implements(Server.client_type)
    def client_type(self) -> Type["Client"]:
        return UnixSocketClient

    @property
    @implements(Server.channel_type)
    def channel_type(self) -> ChannelType:
        return ChannelType.ipc

    @staticmethod
    @implements(Server.create)
    async def create(config: Dict) -> "Server":
        config = config.copy()
        if 'address' in config:
            process_index = int(urlparse(config.pop('address')).path.lstrip('/'))
        else:
            process_index = config.pop('process_index')
        handle_channel = config.pop('handle_channel')
        path = config.pop('path', _gen_unix_socket_default_path(process_index))

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if 'start_serving' not in config:
            config['start_serving'] = False

        async def handle_connection(reader, writer):
            # create a channel when client connected
            return await server.on_connected(reader, writer,
                                             local_address=server.address)

        aio_server = await asyncio.start_unix_server(
            handle_connection, path=path, **config)

        for sock in aio_server.sockets:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

        server = UnixSocketServer(process_index, aio_server, path,
                                  channel_handler=handle_channel)
        return server

    @implements(Server.stop)
    async def stop(self):
        await super().stop()
        try:
            os.remove(self.path)
        except OSError:  # pragma: no cover
            pass


@register_client
class UnixSocketClient(Client):
    __slots__ = ()

    scheme = UnixSocketServer.scheme

    @staticmethod
    @lru_cache(100)
    def _get_process_index(addr):
        return int(urlparse(addr).path.lstrip('/'))

    @staticmethod
    @implements(Client.connect)
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        process_index = UnixSocketClient._get_process_index(dest_address)
        path = kwargs.pop('path',
                          _gen_unix_socket_default_path(process_index))
        try:
            (reader, writer) = await asyncio.open_unix_connection(path, **kwargs)
        except FileNotFoundError:
            raise ConnectionRefusedError('Cannot connect unix socket '
                                         'due to file not exists')
        channel = SocketChannel(reader, writer,
                                local_address=local_address,
                                dest_address=dest_address)
        return UnixSocketClient(local_address, dest_address, channel)
