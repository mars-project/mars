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
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, AnyField, ListField
from ...storage import StorageLevel
from ...utils import extensible
from .core import DataManagerActor
from .handler import StorageHandlerActor

DEFAULT_TRANSFER_BLOCK_SIZE = 4 * 1024 ** 2


logger = logging.getLogger(__name__)


class TransferMessage(Serializable):
    data: Any = AnyField('data')
    session_id: str = StringField('session_id')
    data_keys: List[str] = ListField('data_keys')
    level: StorageLevel = ReferenceField('level', StorageLevel)
    eof_marks: List[bool] = ListField('eof_marks')

    def __init__(self,
                 data: List = None,
                 session_id: str = None,
                 data_keys: List[Union[str, tuple]] = None,
                 level: StorageLevel = None,
                 eof_marks: List[bool] = None):
        super().__init__(data=data,
                         session_id=session_id,
                         data_keys=data_keys,
                         level=level,
                         eof_marks=eof_marks)


class SenderManagerActor(mo.StatelessActor):
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
                         data_keys: List[str],
                         level: StorageLevel,
                         block_size: int
                         ):

        class BufferedSender:
            def __init__(self):
                self._buffers = []
                self._send_keys = []
                self._eof_marks = []

            async def flush(self):
                if self._buffers:
                    transfer_message = TransferMessage(
                        self._buffers, session_id, self._send_keys,
                        level, self._eof_marks)
                    await receiver_ref.receive_part_data(transfer_message)
                self._buffers = []
                self._send_keys = []
                self._eof_marks = []

            async def send(self, buffer, eof_mark, key):
                self._eof_marks.append(eof_mark)
                self._buffers.append(buffer)
                self._send_keys.append(key)
                if sum(len(b) for b in self._buffers) >= block_size:
                    await self.flush()

        sender = BufferedSender()
        for data_key in data_keys:
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
                    await sender.send(part_data, is_eof, data_key)
                    if is_eof:
                        break
        await sender.flush()

    @extensible
    async def send_batch_data(self,
                              session_id: str,
                              data_keys: List[str],
                              address: str,
                              level: StorageLevel,
                              block_size: int = None,
                              error: str = 'raise'):
        logger.debug('Begin to send data (%s, %s) to %s', session_id, data_keys, address)
        block_size = block_size or self._transfer_block_size
        receiver_ref: Union[ReceiverManagerActor, mo.ActorRef] = await self.get_receiver_ref(address)
        get_infos = []
        for data_key in data_keys:
            get_infos.append(self._data_manager_ref.get_data_info.delay(session_id, data_key, error))
        infos = await self._data_manager_ref.get_data_info.batch(*get_infos)
        filtered = [(data_info, data_key) for data_info, data_key in
                    zip(infos, data_keys) if data_info is not None]
        if filtered:
            infos, data_keys = zip(*filtered)
        else:
            infos, data_keys = [], []
        data_sizes = [info.store_size for info in infos]
        await receiver_ref.open_writers(session_id, data_keys, data_sizes, level)

        send_task = asyncio.create_task(
            self._send_data(receiver_ref, session_id, data_keys, level, block_size))
        await asyncio.gather(send_task)

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
            await self._quota_refs[level].release_quota(sum(data_sizes))
            _ = [task.cancel() for task in tasks]
            raise

    async def do_write(self, message: TransferMessage):
        for data, data_key, is_eof in zip(message.data,
                                          message.data_keys,
                                          message.eof_marks):
            writer, _, _ = self._key_to_writer_info[
                (message.session_id, data_key)]
            await writer.write(data)
            if is_eof:
                await writer.close()

    async def receive_part_data(self, message: TransferMessage):
        try:
            yield self.do_write(message)
        except asyncio.CancelledError:
            for data_key in message.data_keys:
                if (message.session_id, data_key) in self._key_to_writer_info:
                    writer, data_size, level = self._key_to_writer_info[
                        (message.session_id, data_key)]
                    await self._quota_refs[level].release_quota(data_size)
                    await self._storage_handler.delete(
                        message.session_id, data_key, error='ignore')
                    await writer.clean_up()
                    self._key_to_writer_info.pop((
                        message.session_id, data_key))
