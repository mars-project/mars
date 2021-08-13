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
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Union, Tuple

from ... import oscar as mo
from ...storage import StorageLevel
from .core import DataManagerActor
from .errors import NoDataToSpill
from .handler import StorageHandlerActor

logger = logging.getLogger(__name__)

DEFAULT_SPILL_BLOCK_SIZE = 128 * 1024


class SpillStrategy(ABC):
    @abstractmethod
    def record_put_info(self, key, data_size: int):
        """
        Record the data key and data size when putting into storage
        """

    @abstractmethod
    def record_delete_info(self, key):
        """
        Record who is removed from storage
        """

    @abstractmethod
    def get_spill_keys(self, size: int) -> Tuple[List, List]:
        """
        Return sizes and keys for spilling according to spill size
        """


class FIFOStrategy(SpillStrategy):
    def __init__(self, level: StorageLevel):
        self._level = level
        self._data_sizes = dict()
        self._pinned_keys = defaultdict(int)
        self._spilling_keys = set()

    def record_put_info(self, key, data_size: int):
        self._data_sizes[key] = data_size

    def record_delete_info(self, key):
        self._data_sizes.pop(key)
        if key in self._spilling_keys:
            self._spilling_keys.remove(key)

    def pin_data(self, key):
        self._pinned_keys[key] += 1

    def unpin_data(self, key):
        if key not in self._pinned_keys:
            return
        self._pinned_keys[key] -= 1
        if self._pinned_keys[key] <= 0:
            del self._pinned_keys[key]

    def get_spillable_size(self):
        total_size = 0
        for data_key, data_size in self._data_sizes.items():
            if data_key not in self._pinned_keys and \
                    data_key not in self._spilling_keys:
                total_size += data_size
        return total_size

    def get_spill_keys(self, size: int) -> Tuple[List, List]:
        spill_sizes = []
        spill_keys = []
        spill_size = 0
        for data_key, data_size in self._data_sizes.items():
            if spill_size >= size:
                break
            if data_key in self._pinned_keys:
                continue
            if data_key in self._spilling_keys:
                continue
            spill_sizes.append(data_size)
            spill_keys.append(data_key)
            spill_size += data_size

        if spill_size < size:  # pragma: no cover
            pinned_sizes = dict((k, self._data_sizes[k]) for k in self._pinned_keys)
            spilling_keys = dict((k, self._data_sizes[k]) for k in self._spilling_keys)
            logger.debug('No data can be spilled for level: %s, pinned keys: %s,'
                         ' spilling keys: %s', self._level, pinned_sizes, spilling_keys)
            raise NoDataToSpill(f'No data can be spilled for level: {self._level}')
        self._spilling_keys.update(set(spill_keys))
        return spill_sizes, spill_keys


class SpillManagerActor(mo.StatelessActor):
    """
    The actor to handle the race condition when NoDataToSpill happens.
    There are two situations when spill raises `NoDataToSpill`,
    one is that space is allocated while objects are not put into storage,
    another is some objects are pinned that can not be spilled,
    so we create an asyncio event if not have enough objects to spill,
    when put or unpin happens, we will notify and check spillable size,
    if size is enough for spilling, call event.set() to wake up spilling task.
    """
    def __init__(self, level: StorageLevel):
        self._level = level
        self._event = None
        self._lock = asyncio.Lock()

    @classmethod
    def gen_uid(cls, band_name: str, level: StorageLevel):
        return f'spill_manager_{band_name}_{level}'

    def has_spill_task(self):
        return self._event is not None

    def notify_spillable_space(self, spillable_size: int, quota_left: int):
        event = self._event
        if event is None:
            return
        logger.debug('Notify to check if has space for spilling')
        if spillable_size + quota_left > event.size:
            logger.debug('Check pass, wake up spill task, spill bytes is %s',
                         event.size - quota_left)
            event.size = event.size - quota_left
            event.set()

    async def wait_for_space(self, size: int):
        # make sure only one spilling task is waiting the event
        async with self._lock:
            self._event = event = asyncio.Event()
            event.size = size
            await self._event.wait()
            size = self._event.size
            self._event = None
            return size


async def spill(request_size: int,
                level: StorageLevel,
                band_name: str,
                data_manager: Union[mo.ActorRef, DataManagerActor],
                storage_handler: Union[mo.ActorRef, StorageHandlerActor],
                block_size=None,
                multiplier=1.1):
    logger.debug('%s is full, need to spill %s bytes, '
                 'multiplier is %s', level, request_size, multiplier)
    request_size *= multiplier
    block_size = block_size or DEFAULT_SPILL_BLOCK_SIZE
    spill_level = level.spill_level()
    spill_sizes, spill_keys = await data_manager.get_spill_keys(
        level, band_name, request_size)
    logger.debug('Decide to spill %s bytes, '
                 'data keys are %s', sum(spill_sizes), spill_keys)

    for (session_id, key), size in zip(spill_keys, spill_sizes):
        reader = await storage_handler.open_reader(session_id, key)
        writer = await storage_handler.open_writer(
            session_id, key, size, spill_level)
        async with reader:
            async with writer:
                while True:
                    block_data = await reader.read(block_size)
                    if not block_data:
                        break
                    else:
                        await writer.write(block_data)
        try:
            await storage_handler.delete_object(
                session_id, key, size, reader.object_id, level)
        except KeyError:  # pragma: no cover
            # workaround for the case that the object
            # has been deleted during spill
            logger.debug('Data %s %s is deleted during spill', session_id, key)
            await storage_handler.delete(session_id, key, error='ignore')
    logger.debug('Spill finishes, release %s bytes of %s', sum(spill_sizes), level)
