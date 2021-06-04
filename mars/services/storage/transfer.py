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
from typing import Union, Any

from ... import oscar as mo
from ...serialization.serializables import Serializable, BoolField,\
    StringField, ReferenceField, AnyField
from ...storage import StorageLevel
from ...utils import extensible
from .core import StorageManagerActor, StorageHandlerActor

DEFAULT_TRANSFER_BLOCK_SIZE = 5 * 1024 ** 2


class TransferMessage(Serializable):
    data: Any = AnyField('data')
    session_id: str = StringField('session_id')
    data_key: str = AnyField('data_key')
    level: StorageLevel = ReferenceField('level', StorageLevel)
    is_eof: bool = BoolField('is_eof')

    def __init__(self,
                 data: Any = None,
                 session_id: str = None,
                 data_key: Union[str, tuple] = None,
                 level: StorageLevel = None,
                 is_eof: bool = None):
        super().__init__(data=data,
                         session_id=session_id,
                         data_key=data_key,
                         level=level,
                         is_eof=is_eof)


class SenderManagerActor(mo.Actor):
    def __init__(self,
                 transfer_block_size: int = None):
        self._transfer_block_size = transfer_block_size or DEFAULT_TRANSFER_BLOCK_SIZE

    async def __post_create__(self):
        self._storage_manager_ref: Union[mo.ActorRef, StorageManagerActor] = \
            await mo.actor_ref(self.address, StorageManagerActor.default_uid())
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    @staticmethod
    async def get_receiver_ref(address: str):
        return await mo.actor_ref(
            address=address, uid=ReceiverManagerActor.default_uid())

    @extensible
    async def send_data(self,
                        session_id: str,
                        data_key: str,
                        address: str,
                        level: StorageLevel,
                        block_size: int = None):
        block_size = block_size or self._transfer_block_size
        receiver_ref = await self.get_receiver_ref(address)
        info = await self._storage_manager_ref.get_data_info(session_id, data_key)
        store_size = info.store_size
        await receiver_ref.open_writer(session_id, data_key,
                                       store_size, level)

        sent_size = 0
        async with await self._storage_handler.open_reader(
                session_id, data_key) as reader:
            while True:
                part_data = await reader.read(block_size)
                is_eof = sent_size + len(part_data) >= store_size
                message = TransferMessage(part_data, session_id, data_key, level, is_eof)
                send_task = asyncio.create_task(receiver_ref.receive_part_data(message))
                yield send_task
                sent_size += len(part_data)
                if is_eof:
                    break


class ReceiverManagerActor(mo.Actor):
    def __init__(self):
        self._key_to_writer_info = dict()

    async def __post_create__(self):
        self._storage_manager_ref: Union[mo.ActorRef, StorageManagerActor] = \
            await mo.actor_ref(self.address, StorageManagerActor.default_uid())
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    async def create_writer(self,
                            session_id: str,
                            data_key: str,
                            data_size: int,
                            level: StorageLevel):
        writer = await self._storage_handler.open_writer(session_id, data_key,
                                                         data_size, level)
        self._key_to_writer_info[(session_id, data_key)] = (writer, data_size, level)

    async def open_writer(self,
                          session_id: str,
                          data_key: str,
                          data_size: int,
                          level: StorageLevel):
        create_task = asyncio.create_task(
            self.create_writer(session_id, data_key, data_size, level))
        try:
            await create_task
        except asyncio.CancelledError:
            create_task.cancel()
            raise

    async def do_write(self, message: TransferMessage):
        writer, _, _ = self._key_to_writer_info[
            (message.session_id, message.data_key)]
        await writer.write(message.data)
        if message.is_eof:
            await writer.close()

    async def receive_part_data(self, message: TransferMessage):

        try:
            yield self.do_write(message)
        except asyncio.CancelledError:
            writer, data_size, level = self._key_to_writer_info[
                (message.session_id, message.data_key)]
            await self._storage_manager_ref.release_quota(
                data_size, level)
            await self._storage_handler.delete(
                message.session_id, message.data_key, error='ignore')
            await writer.clean_up()
            self._key_to_writer_info.pop((
                message.session_id, message.data_key))
