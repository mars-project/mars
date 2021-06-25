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
from typing import Dict, Union

from ..errors import ServerClosed
from .communication import Client
from .message import _MessageBase, ResultMessage, ErrorMessage, \
    DeserializeMessageFailed
from .router import Router


result_message_type = Union[ResultMessage, ErrorMessage]


class ActorCaller:
    __slots__ = '_client_to_message_futures', '_clients'

    def __init__(self):
        self._client_to_message_futures: \
            Dict[Client, Dict[bytes, asyncio.Future]] = dict()
        self._clients: Dict[Client, asyncio.Task] = dict()

    async def get_client(self,
                         router: Router,
                         dest_address: str) -> Client:
        client = await router.get_client(dest_address,
                                         from_who=self)
        if client not in self._clients:
            self._clients[client] = asyncio.create_task(self._listen(client))
            self._client_to_message_futures[client] = dict()
        return client

    async def _listen(self, client: Client):
        while not client.closed:
            try:
                try:
                    message: _MessageBase = await client.recv()
                except (EOFError, ConnectionError, BrokenPipeError):
                    # remote server closed, close client and raise ServerClosed
                    try:
                        await client.close()
                    except (ConnectionError, BrokenPipeError):
                        # close failed, ignore it
                        pass
                    raise ServerClosed(f'Remote server {client.dest_address} closed') from None
                future = self._client_to_message_futures[client].pop(message.message_id)
                future.set_result(message)
            except DeserializeMessageFailed as e:  # pragma: no cover
                message_id = e.message_id
                future = self._client_to_message_futures[client].pop(message_id)
                future.set_exception(e)
            except Exception as e:  # noqa: E722  # pylint: disable=bare-except
                message_futures = self._client_to_message_futures.get(client)
                self._client_to_message_futures[client] = dict()
                for future in message_futures.values():
                    future.set_exception(e)

    async def call(self,
                   router: Router,
                   dest_address: str,
                   message: _MessageBase,
                   wait: bool = True) \
            -> Union[ResultMessage, ErrorMessage, asyncio.Future]:
        client = await self.get_client(router, dest_address)
        loop = asyncio.get_running_loop()
        wait_response = loop.create_future()
        self._client_to_message_futures[client][message.message_id] = wait_response

        try:
            await client.send(message)
        except ConnectionError:
            try:
                await client.close()
            except ConnectionError:
                # close failed, ignore it
                pass
            raise ServerClosed(f'Remote server {client.dest_address} closed')

        if not wait:
            return wait_response
        else:
            return await wait_response

    async def stop(self):
        try:
            await asyncio.gather(*[client.close() for client in self._clients])
        except (ConnectionError, ServerClosed):
            pass
        self.cancel_tasks()

    def cancel_tasks(self):
        # cancel listening for all clients
        _ = [task.cancel() for task in self._clients.values()]
