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
import logging
from typing import Union, Any, List

from ... import oscar as mo
from ...serialization.serializables import Serializable, BoolField,\
    StringField, ReferenceField, AnyField
from ...storage import StorageLevel
from ...utils import extensible
from .core import DataManagerActor
from .handler import StorageHandlerActor

DEFAULT_TRANSFER_BLOCK_SIZE = 5 * 1024 ** 2


logger = logging.getLogger(__name__)


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
        self._data_manager_ref: Union[mo.ActorRef, DataManagerActor] = \
            await mo.actor_ref(self.address, DataManagerActor.default_uid())
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    @staticmethod
    async def get_receiver_ref(address: str):
        return await mo.actor_ref(
            address=address, uid=ReceiverManagerActor.default_uid())

    async def _send_data(self,
                         receiver_ref: Union[mo.ActorRef],
                         session_id: str,
                         data_key: str,
                         level: StorageLevel,
                         block_size: int
                         ):
        sent_size = 0
        async with await self._storage_handler.open_reader(
                session_id, data_key) as reader:
            while True:
                part_data = await reader.read(block_size)
                # Notes on [How to decide whether the reader reaches EOF?]
                #
                # In some storage backend, e.g., the reported memory usage (i.e., the
                # `store_size`) may not same with the byte size that need to be transferred
                # when moving to a remote worker. Thus, we think the reader reaches EOF
                # when a `read` request returns nothing, rather than comparing the `sent_size`
                # and the `store_size`.
                #
                is_eof = not part_data  # can be non-empty bytes, empty bytes and None
                message = TransferMessage(part_data, session_id, data_key, level, is_eof)
                send_task = asyncio.create_task(receiver_ref.receive_part_data(message))
                await send_task
                sent_size += len(part_data)
                if is_eof:
                    break

    @extensible
    async def send_batch_data(self,
                              session_id: str,
                              data_keys: List[str],
                              address: str,
                              level: StorageLevel,
                              block_size: int = None):
        logger.debug('Begin to send data (%s, %s) to %s', session_id, data_keys, address)
        block_size = block_size or self._transfer_block_size
        receiver_ref: Union[ReceiverManagerActor, mo.ActorRef] = await self.get_receiver_ref(address)
        get_infos = []
        for data_key in data_keys:
            get_infos.append(self._data_manager_ref.get_data_info.delay(session_id, data_key))
        infos = await self._data_manager_ref.get_data_info.batch(*get_infos)
        data_sizes = [info.store_size for info in infos]
        await receiver_ref.open_writers(session_id, data_keys, data_sizes, level)

        send_tasks = []
        for data_key, info in zip(data_keys, infos):
            send_task = asyncio.create_task(
                self._send_data(receiver_ref, session_id, data_key,
                                level, block_size))
            send_tasks.append(send_task)
        await asyncio.gather(*send_tasks)

        logger.debug('Finish sending data (%s, %s) to %s', session_id, data_keys, address)


class ReceiverManagerActor(mo.Actor):
    def __init__(self, quota_refs):
        self._key_to_writer_info = dict()
        self._quota_refs = quota_refs

    async def __post_create__(self):
        self._storage_handler: Union[mo.ActorRef, StorageHandlerActor] = \
            await mo.actor_ref(self.address, StorageHandlerActor.default_uid())

    async def create_writer(self,
                            session_id: str,
                            data_key: str,
                            data_size: int,
                            level: StorageLevel):
        writer = await self._storage_handler.open_writer(session_id, data_key,
                                                         data_size, level, request_quota=False)
        self._key_to_writer_info[(session_id, data_key)] = (writer, data_size, level)

    async def open_writers(self,
                           session_id: str,
                           data_keys: List[str],
                           data_sizes: List[int],
                           level: StorageLevel):
        await self._storage_handler.request_quota_with_spill(level, sum(data_sizes))
        tasks = []
        for data_key, data_size in zip(data_keys, data_sizes):
            create_task = asyncio.create_task(
                self.create_writer(session_id, data_key, data_size, level))
            tasks.append(create_task)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            [t.cancel() for t in tasks]
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
            await self._quota_refs[level].release_quota(data_size)
            await self._storage_handler.delete(
                message.session_id, message.data_key, error='ignore')
            await writer.clean_up()
            self._key_to_writer_info.pop((
                message.session_id, message.data_key))
