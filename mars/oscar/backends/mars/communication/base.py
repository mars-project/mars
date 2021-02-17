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

from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Coroutine, Type

from .....utils import classproperty


class Channel(ABC):
    """
    Channel is used to do data exchange between server and client.
    """
    __slots__ = 'local_address', 'dest_address', 'compression'

    name = None

    def __init__(self,
                 local_address: str = None,
                 dest_address: str = None,
                 compression=None):
        self.local_address = local_address
        self.dest_address = dest_address
        self.compression = compression

    @abstractmethod
    async def send(self, message: Any):
        """
        Send data to dest.

        Parameters
        ----------
        message:
            data that sent to dest.
        """

    @abstractmethod
    async def recv(self):
        """
        Receive data that sent from dest.
        """

    @abstractmethod
    async def close(self):
        """
        Close channel.
        """

    @property
    @abstractmethod
    def closed(self) -> bool:
        """
        This channel is closed or not.

        Returns
        -------
        closed:
            If the channel is closed.
        """

    @property
    def info(self) -> Dict:
        return {
            'name': self.name,
            'compression': self.compression,
            'local_address': self.local_address,
            'dest_address': self.dest_address
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class Server(ABC):
    __slots__ = 'address', 'handle_channel'

    scheme = None

    def __init__(self,
                 address: str,
                 handle_channel: Callable[[Channel], Coroutine] = None):
        self.address = address
        self.handle_channel = handle_channel

    @classproperty
    @abstractmethod
    def client_type(self) -> Type["Client"]:
        """
        Return the corresponding client type.

        Returns
        -------
        client_type: type
            client type.
        """

    @staticmethod
    @abstractmethod
    async def create(config: Dict) -> "Server":
        """
        Create a server instance according to configuration.

        Parameters
        ----------
        config: dict
            configuration about creating a channel.

        Returns
        -------
        server: Server
            a server that wating for connections from clients.
        """

    @abstractmethod
    async def start(self):
        """
        Used for listening to port or similar stuff.
        """

    @abstractmethod
    async def join(self, timeout=None):
        """
        Wait forever until timeout.
        """

    @abstractmethod
    async def on_connected(self, *args, **kwargs):
        """
        Return a channel when new client connected.

        Returns
        -------
        channel: Channel
            channel for communication
        """

    @abstractmethod
    async def shutdown(self):
        """
        Shutdown the server.
        """

    @property
    @abstractmethod
    def is_shutdown(self) -> bool:
        """
        If this server is shutdown or not.

        Returns
        -------
        if_shutdown: bool
           This server is shutdown or not.
        """

    def info(self) -> Dict:
        return {
            'name': self.scheme,
            'address': self.address
        }

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


class Client(ABC):
    __slots__ = 'local_address', 'dest_address', 'channel'

    def __init__(self,
                 local_address: str,
                 dest_address: str,
                 channel: Channel):
        self.local_address = local_address
        self.dest_address = dest_address
        self.channel = channel

    @staticmethod
    @abstractmethod
    async def connect(dest_address: str,
                      local_address: str = None,
                      **kwargs) -> "Client":
        """
        Create a client that is able to connect to some server.

        Parameters
        ----------
        dest_address: str
            Destination server address that to connect to.
        local_address: str
            local address.

        Returns
        -------
        client: Client
            Client that holds a channel to communicate.
        """

    async def close(self):
        """
        Close connection.
        """
        await self.channel.close()

    @property
    def closed(self) -> bool:
        """
        This client is closed or not.

        Returns
        -------
        closed: bool
            If the client is closed.
        """
        return self.channel.closed

    @property
    def info(self) -> Dict:
        return {
            'local_address': self.local_address,
            'dest_address': self.dest_address,
            'channel_name': self.channel.name
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
