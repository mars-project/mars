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

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union

from ... import oscar as mo
from ...storage import StorageLevel
from .core import StorageManagerActor, StorageHandlerActor
from .errors import NoDataToSpill

logger = logging.getLogger(__name__)

DEFAULT_SPILL_BLOCK_SIZE = 128 * 1024


class SpillStrategy(ABC):
    @abstractmethod
    def put(self, key, data_size: int):
        """
        Put element
        """

    @abstractmethod
    def delete(self, key):
        """
        Delete element
        """

    @abstractmethod
    def get_spill_keys(self, size: int):
        """
        Return keys for spilling according to spill size
        """


class BaseStrategy(SpillStrategy):
    def __init__(self, level: StorageLevel):
        self._level = level
        self._data_sizes = dict()
        self._pinned_keys = defaultdict(int)

    def put(self, key, data_size: int):
        self._data_sizes[key] = data_size

    def delete(self, key):
        self._data_sizes.pop(key)

    def pin_data(self, key):
        self._pinned_keys[key] += 1

    def unpin_data(self, key):
        self._pinned_keys[key] -= 1
        if self._pinned_keys[key] <= 0:
            del self._pinned_keys[key]

    def get_spill_keys(self, size: int):
        spill_sizes = []
        spill_keys = []
        spill_size = 0
        for data_key, data_size in self._data_sizes.items():
            if data_key in self._pinned_keys:
                continue
            spill_sizes.append(data_size)
            spill_keys.append(data_key)
            spill_size += data_size
            if spill_size > size:
                break
        if spill_size < size:
            raise NoDataToSpill(f'No data can be spilled for level: {self._level},'
                                f'pinned keys: {self._pinned_keys}')
        return spill_sizes, spill_keys


async def spill(request_size: int,
                level: StorageLevel,
                storage_manager: Union[mo.ActorRef, StorageManagerActor],
                storage_handler: Union[mo.ActorRef, StorageHandlerActor],
                block_size=None,
                multiplier=1.2):
    await storage_manager.lock_quota(level)
    logger.debug(f'{level} is full, need to spill {request_size} bytes, '
                 f'multiplier is {multiplier}')
    request_size *= multiplier
    block_size = block_size or DEFAULT_SPILL_BLOCK_SIZE
    spill_level = level.spill_level()
    spill_sizes, spill_keys = await storage_manager.get_spill_keys(
        level, request_size)
    logger.debug(f'Decide to spill {sum(spill_sizes)} bytes, '
                 f'data keys are {spill_keys}')

    try:
        await storage_manager.request_quota(sum(spill_sizes), spill_level)
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
            await storage_handler.delete_object(
                session_id, key, reader.object_id, level)
    finally:
        await storage_manager.unlock_quota(level)

    await storage_manager.release_quota(sum(spill_sizes), level)
    logger.debug(f'Spill finishes, release {sum(spill_sizes)} bytes of {level}')
