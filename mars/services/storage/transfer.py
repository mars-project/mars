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
from typing import Dict, Union, Any, List

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, AnyField, ListField
from ...storage import StorageLevel
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
                 band_name: str = 'numa-0',
                 transfer_block_size: int = None,
                 storage_handler_ref: Union[mo.ActorRef, StorageHandlerActor] = None):
        self._band_name = band_name
        self._storage_handler = storage_handler_ref
        self._transfer_block_size = transfer_block_size or DEFAULT_TRANSFER_BLOCK_SIZE

    @classmethod
    def gen_uid(cls, band_name: str):
        return f'sender_manager_{band_name}'

    async def __post_create__(self):
        self._data_manager_ref: Union[mo.ActorRef, DataManagerActor] = \
            await mo.actor_ref(self.address, DataManagerActor.default_uid())
        if self._storage_handler is None:  # for test
            self._storage_handler = await mo.actor_ref(
                self.address, StorageHandlerActor.gen_uid('numa-0'))

    @staticmethod
    @alru_cache
    async def get_receiver_ref(address: str, band_name: str):
        return await mo.actor_ref(
            address=address, uid=ReceiverManagerActor.gen_uid(band_name))

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
        open_reader_tasks = []
        for data_key in data_keys:
            open_reader_tasks.append(
                self._storage_handler.open_reader.delay(session_id, data_key))
        readers = await self._storage_handler.open_reader.batch(*open_reader_tasks)

        for data_key, reader in zip(data_keys, readers):
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

    @mo.extensible
    async def send_batch_data(self,
                              session_id: str,
                              data_keys: List[str],
                              address: str,
                              level: StorageLevel,
                              band_name: str = 'numa-0',
                              block_size: int = None,
                              error: str = 'raise'):
        logger.debug('Begin to send data (%s, %s) to %s', session_id, data_keys, address)
        block_size = block_size or self._transfer_block_size
        receiver_ref: Union[ReceiverManagerActor, mo.ActorRef] = \
            await self.get_receiver_ref(address, band_name)
        get_infos = []
        for data_key in data_keys:
            get_infos.append(self._data_manager_ref.get_data_info.delay(
                session_id, data_key, self._band_name, error))
        infos = await self._data_manager_ref.get_data_info.batch(*get_infos)
        filtered = [(data_info, data_key) for data_info, data_key in
                    zip(infos, data_keys) if data_info is not None]
        if filtered:
            infos, data_keys = zip(*filtered)
        else:  # pragma: no cover
            infos, data_keys = [], []
        data_sizes = [info.store_size for info in infos]
        if level is None:
            level = infos[0].level
        await receiver_ref.open_writers(session_id, data_keys, data_sizes, level)

        await self._send_data(receiver_ref, session_id, data_keys, level, block_size)
        logger.debug('Finish sending data (%s, %s) to %s', session_id, data_keys, address)


class ReceiverManagerActor(mo.Actor):
    def __init__(self,
                 quota_refs: Dict,
                 storage_handler_ref: Union[mo.ActorRef, StorageHandlerActor] = None):
        self._key_to_writer_info = dict()
        self._quota_refs = quota_refs
        self._storage_handler = storage_handler_ref

    async def __post_create__(self):
        if self._storage_handler is None: # for test
            self._storage_handler = await mo.actor_ref(
                self.address, StorageHandlerActor.gen_uid('numa-0'))

    @classmethod
    def gen_uid(cls, band_name: str):
        return f'sender_receiver_{band_name}'

    async def create_writers(self,
                             session_id: str,
                             data_keys: List[str],
                             data_sizes: List[int],
                             level: StorageLevel):
        tasks = []
        for data_key, data_size in zip(data_keys, data_sizes):
            tasks.append(self._storage_handler.open_writer.delay(
                session_id, data_key, data_size, level, request_quota=False))
        writers = await self._storage_handler.open_writer.batch(*tasks)
        for data_key, data_size, writer in zip(data_keys, data_sizes, writers):
            self._key_to_writer_info[(session_id, data_key)] = (writer, data_size, level)

    async def open_writers(self,
                           session_id: str,
                           data_keys: List[str],
                           data_sizes: List[int],
                           level: StorageLevel):
        await self._storage_handler.request_quota_with_spill(level, sum(data_sizes))
        future = asyncio.create_task(
            self.create_writers(session_id, data_keys, data_sizes, level))
        try:
            await future
        except asyncio.CancelledError:
            await self._quota_refs[level].release_quota(sum(data_sizes))
            future.cancel()
            raise

    async def do_write(self, message: TransferMessage):
        # close may be a high cost operation, use create_task
        close_tasks = []
        for data, data_key, is_eof in zip(message.data,
                                          message.data_keys,
                                          message.eof_marks):
            writer, _, _ = self._key_to_writer_info[
                (message.session_id, data_key)]
            if data:
                await writer.write(data)
            if is_eof:
                close_tasks.append(asyncio.create_task(writer.close()))
        await asyncio.gather(*close_tasks)

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
