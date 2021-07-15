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
import enum
import functools
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Awaitable, Callable, Dict, Optional, \
    Set, Tuple, TypeVar

from ...serialization.serializables import Serializable, Float64Field, \
    Int32Field, Int64Field, StringField
from ...storage import StorageLevel
from ..core import NodeRole


class NodeStatus(enum.Enum):
    STARTING = 0
    READY = 1
    DEGENERATED = 2
    STOPPING = 3
    STOPPED = -1


@dataclass
class NodeInfo:
    role: NodeRole
    status: NodeStatus = NodeStatus.READY
    update_time: float = field(default_factory=time.time)
    env: Dict = field(default_factory=dict)
    resource: Dict = field(default_factory=dict)
    detail: Dict = field(default_factory=dict)


class WatchNotifier:
    _events: Set[asyncio.Event]

    def __init__(self):
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._version = 0

    async def watch(self, version: Optional[int] = None):
        if version != self._version:
            return self._version
        await self._event.wait()
        return self._version

    async def notify(self):
        async with self._lock:
            self._version += 1
            self._event.set()
            self._event = asyncio.Event()


RetType = TypeVar('RetType')


def watch_method(
        func: Callable[..., Awaitable[Tuple[int, RetType]]]
) -> Callable[..., AsyncGenerator[RetType, None]]:
    @functools.wraps(func)
    async def wrapped(*args, **kwargs):
        if 'version' in kwargs:
            yield await func(*args, **kwargs)
            return

        kwargs['version'] = None
        while True:
            version, val = await func(*args, **kwargs)
            kwargs['version'] = version
            yield val

    return wrapped


class WorkerSlotInfo(Serializable):
    slot_id: int = Int32Field('slot_id')
    session_id: str = StringField('session_id')
    subtask_id: str = StringField('subtask_id')
    processor_usage: float = Float64Field('processor_usage')


class QuotaInfo(Serializable):
    quota_size: int = Int64Field('quota_size')
    allocated_size: int = Int64Field('allocated_size')
    hold_size: int = Int64Field('hold_size')


class StorageInfo(Serializable):
    storage_level: StorageLevel = Int32Field('storage_level',
                                             on_serialize=lambda x: x.value,
                                             on_deserialize=StorageLevel)
    total_size: int = Int64Field('total_size')
    used_size: int = Int64Field('used_size')
    pinned_size: int = Int64Field('pinned_size', default=None)


class DiskInfo(Serializable):
    path: str = StringField('path')
    limit_size: int = Int64Field('limit_size', default=None)
