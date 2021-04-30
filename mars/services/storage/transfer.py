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
from dataclasses import dataclass
from typing import Union

from ... import oscar as mo
from ...config import options
from ...storage import StorageLevel
from ...utils import dataslots, extensible
from .core import StorageManagerActor, StorageHandlerActor


@dataslots
@dataclass
class SendMessage:
    data: bytes
    session_id: str
    data_key: str
    level: StorageLevel
    is_eof: bool


class SenderManagerActor(mo.Actor):
    def __init__(self):
        self._key_to_send_task = dict()

    async def __post_create__(self):
        self._storage_manager_ref: Union[mo.ActorRef, StorageManagerActor] = \
            await mo.actor_ref(self.address, StorageManagerActor.default_uid())
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    @staticmethod
    async def send_part(receiver_ref: Union[mo.ActorRef, "ReceiverActor"],
                        message: SendMessage):
        await receiver_ref.receive_part_data(message)

    @extensible
    async def send_data(self,
                        session_id: str,
                        data_key: str,
                        address: str,
                        level: StorageLevel,
                        block_size: int = None):
        block_size = block_size or options.worker.transfer_block_size
        receiver_ref = await mo.actor_ref(
            address=address, uid=ReceiverActor.default_uid())
        info = await self._storage_manager_ref.get_data_info(session_id, data_key)
        store_size = info.store_size
        yield receiver_ref.open_writer(session_id, data_key,
                                       store_size, level)
        async with await self._storage_handler.open_reader(
                session_id, data_key) as reader:
            while True:
                part_data = await reader.read(block_size)
                is_eof = len(part_data) < block_size
                message = SendMessage(part_data, session_id, data_key, level, is_eof)
                send_task = asyncio.create_task(self.send_part(receiver_ref, message))
                self._key_to_send_task[(session_id, data_key)] = send_task
                try:
                    yield send_task
                except asyncio.CancelledError:
                    await receiver_ref.clean_cancelled_transfer(
                        session_id, data_key)
                if is_eof:
                    break

    async def cancel(self,
                     session_id: str,
                     data_key: str):
        task = self._key_to_send_task[(session_id, data_key)]
        task.cancel()


class ReceiverActor(mo.Actor):
    def __init__(self):
        self._key_to_writer = dict()

    async def __post_create__(self):
        self._storage_manager_ref: Union[mo.ActorRef, StorageManagerActor] = \
            await mo.actor_ref(self.address, StorageManagerActor.default_uid())
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    async def open_writer(self,
                          session_id: str,
                          data_key: str,
                          data_size: int,
                          level: StorageLevel):
        await self._storage_manager_ref.allocate_size(data_size, level)
        writer = await self._storage_handler.open_writer(session_id, data_key,
                                                         data_size, level)
        self._key_to_writer[(session_id, data_key)] = writer

    async def clean_cancelled_transfer(self,
                                       session_id: str,
                                       data_key: str):
        writer = self._key_to_writer[(session_id, data_key)]
        await writer.close(done=False)

    async def receive_part_data(self, message: SendMessage):
        writer = self._key_to_writer[(message.session_id,
                                      message.data_key)]
        yield writer.write(message.data)
        if message.is_eof:
            yield writer.close()
