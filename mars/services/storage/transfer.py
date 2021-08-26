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
from dataclasses import dataclass
import logging
from typing import Dict, Union, Any, List

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...serialization.serializables import Serializable, StringField, \
    ReferenceField, AnyField, ListField
from ...storage import StorageLevel
from ...utils import dataslots
from .core import DataManagerActor, WrappedStorageFileObject
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
                 data_manager_ref: Union[mo.ActorRef, DataManagerActor] = None,
                 storage_handler_ref: Union[mo.ActorRef, StorageHandlerActor] = None):
        self._band_name = band_name
        self._data_manager_ref = data_manager_ref
        self._storage_handler = storage_handler_ref
        self._transfer_block_size = transfer_block_size or DEFAULT_TRANSFER_BLOCK_SIZE

    @classmethod
    def gen_uid(cls, band_name: str):
        return f'sender_manager_{band_name}'

    async def __post_create__(self):
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
        pin_tasks = []
        for data_key in data_keys:
            get_infos.append(self._data_manager_ref.get_data_info.delay(
                session_id, data_key, self._band_name, error))
            pin_tasks.append(self._data_manager_ref.pin.delay(
                session_id, data_key, self._band_name))
        await self._data_manager_ref.pin.batch(*pin_tasks)
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
        is_transferring_list = await receiver_ref.open_writers(session_id, data_keys, data_sizes, level)
        to_send_keys = []
        to_wait_keys = []
        for data_key, is_transferring in zip(data_keys, is_transferring_list):
            if is_transferring:
                to_wait_keys.append(data_key)
            else:
                to_send_keys.append(data_key)

        if to_send_keys:
            await self._send_data(receiver_ref, session_id, to_send_keys, level, block_size)
        if to_wait_keys:
            await receiver_ref.wait_transfer_done(session_id, to_wait_keys)
        unpin_tasks = []
        for data_key in data_keys:
            unpin_tasks.append(self._data_manager_ref.unpin.delay(
                session_id, data_key, self._band_name, error='ignore'))
        await self._data_manager_ref.unpin.batch(*unpin_tasks)
        logger.debug('Finish sending data (%s, %s) to %s', session_id, data_keys, address)


@dataslots
@dataclass
class WritingInfo:
    writer: WrappedStorageFileObject
    size: int
    level: StorageLevel
    event: asyncio.Event
    ref_counts: int


class ReceiverManagerActor(mo.StatelessActor):
    def __init__(self,
                 quota_refs: Dict,
                 storage_handler_ref: Union[mo.ActorRef, StorageHandlerActor] = None):
        self._quota_refs = quota_refs
        self._storage_handler = storage_handler_ref
        self._writing_infos: Dict[tuple, WritingInfo] = dict()
        self._lock = asyncio.Lock()

    async def __post_create__(self):
        if self._storage_handler is None:  # for test
            self._storage_handler = await mo.actor_ref(
                self.address, StorageHandlerActor.gen_uid('numa-0'))

    @classmethod
    def gen_uid(cls, band_name: str):
        return f'receiver_manager_{band_name}'

    def _decref_writing_key(self, session_id: str, data_key: str):
        self._writing_infos[(session_id, data_key)].ref_counts -= 1
        if self._writing_infos[(session_id, data_key)].ref_counts == 0:
            del self._writing_infos[(session_id, data_key)]

    async def create_writers(self,
                             session_id: str,
                             data_keys: List[str],
                             data_sizes: List[int],
                             level: StorageLevel):
        tasks = dict()
        data_key_to_size = dict()
        being_processed = []
        for data_key, data_size in zip(data_keys, data_sizes):
            data_key_to_size[data_key] = data_size
            if (session_id, data_key) not in self._writing_infos:
                being_processed.append(False)
                tasks[data_key] = self._storage_handler.open_writer.delay(
                    session_id, data_key, data_size, level, request_quota=False)
            else:
                being_processed.append(True)
                self._writing_infos[(session_id, data_key)].ref_counts += 1
        if tasks:
            writers = await self._storage_handler.open_writer.batch(*tuple(tasks.values()))
            for data_key, writer in zip(tasks, writers):
                self._writing_infos[
                    (session_id, data_key)] = WritingInfo(
                    writer, data_key_to_size[data_key], level, asyncio.Event(), 1)
        return being_processed

    async def open_writers(self,
                           session_id: str,
                           data_keys: List[str],
                           data_sizes: List[int],
                           level: StorageLevel):
        async with self._lock:
            await self._storage_handler.request_quota_with_spill(level, sum(data_sizes))
            future = asyncio.create_task(
                self.create_writers(session_id, data_keys, data_sizes, level))
            try:
                return await future
            except asyncio.CancelledError:
                await self._quota_refs[level].release_quota(sum(data_sizes))
                future.cancel()
                raise

    async def do_write(self, message: TransferMessage):
        # close may be a high cost operation, use create_task
        close_tasks = []
        finished_keys = []
        session_id = message.session_id
        for data, data_key, is_eof in zip(message.data,
                                          message.data_keys,
                                          message.eof_marks):
            writer = self._writing_infos[(session_id, data_key)].writer
            if data:
                await writer.write(data)
            if is_eof:
                close_tasks.append(asyncio.create_task(writer.close()))
                finished_keys.append(data_key)
        await asyncio.gather(*close_tasks)
        async with self._lock:
            for data_key in finished_keys:
                event = self._writing_infos[(session_id, data_key)].event
                event.set()
                self._decref_writing_key(session_id, data_key)

    async def receive_part_data(self, message: TransferMessage):
        write_task = asyncio.create_task(self.do_write(message))
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            session_id = message.session_id
            async with self._lock:
                for data_key in message.data_keys:
                    if (message.session_id, data_key) in self._writing_infos:
                        if self._writing_infos[(message.session_id, data_key)].ref_counts == 1:
                            info = self._writing_infos[(message.session_id, data_key)]
                            await self._quota_refs[info.level].release_quota(info.size)
                            await self._storage_handler.delete(
                                message.session_id, data_key, error='ignore')
                            await info.writer.clean_up()
                            info.event.set()
                            self._decref_writing_key(session_id, data_key)
                            write_task.cancel()
                            await write_task
            raise

    async def wait_transfer_done(self, session_id, data_keys):
        await asyncio.gather(*[self._writing_infos[(session_id, key)].event.wait()
                               for key in data_keys])
        async with self._lock:
            for data_key in data_keys:
                self._decref_writing_key(session_id, data_key)
