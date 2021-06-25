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

import functools
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Type, Union

from ..utils import dataslots
from .core import StorageFileObject

_storage_backends = dict()


def register_storage_backend(backend: Type["StorageBackend"]):
    _storage_backends[backend.name] = backend
    return backend


def get_storage_backend(backend_name) -> Type["StorageBackend"]:
    return _storage_backends[backend_name]


_ComparableLevel = Union[int, "StorageLevel"]


class StorageLevel(Enum):
    GPU = 1 << 0
    MEMORY = 1 << 1
    DISK = 1 << 2
    REMOTE = 1 << 3

    def __and__(self, other: _ComparableLevel):
        other_value = getattr(other, 'value', other)
        return self.value & other_value

    __rand__ = __and__

    def __or__(self, other: _ComparableLevel):
        other_value = getattr(other, 'value', other)
        return self.value | other_value

    __ror__ = __or__

    def __lt__(self, other: _ComparableLevel):
        other_value = getattr(other, 'value', other)
        return self.value < other_value

    def __gt__(self, other: _ComparableLevel):
        other_value = getattr(other, 'value', other)
        return self.value > other_value

    def spill_level(self):
        if self == StorageLevel.GPU:
            return StorageLevel.MEMORY
        elif self == StorageLevel.MEMORY:
            return StorageLevel.DISK
        else:  # pragma: no cover
            raise ValueError(f"Level {self} doesn't have spill level")

    @staticmethod
    def from_str(s: str):
        level_mapping = StorageLevel.__members__
        level_strings = [ss.strip() for ss in s.upper().split('|')]
        levels = []
        for ls in level_strings:
            if ls not in level_mapping:  # pragma: no cover
                raise ValueError(f'Unknown level {ls}')
            levels.append(level_mapping[ls])
        return functools.reduce(operator.or_, levels)


@dataslots
@dataclass
class ObjectInfo:
    size: int = None
    device: int = None
    object_id: Any = None


class StorageBackend(ABC):
    name = None

    @classmethod
    @abstractmethod
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        """
        Setup environments, for example, start plasma store for plasma backend.

        Parameters
        ----------
        kwargs : kwargs
            Kwargs for setup.

        Returns
        -------
        Tuple of two dicts
            Dicts for initialization and teardown.
        """

    @staticmethod
    async def teardown(**kwargs):
        """
        Clean up the environments.

        Parameters
        ----------
        kwargs : kwargs
             Parameters for clean up.
        """

    @property
    def size(self) -> Union[int, None]:
        """
        The total size of storage.

        Returns
        -------
        Size: int
            Total size of storage.
        """
        return None

    @property
    @abstractmethod
    def level(self) -> StorageLevel:
        """
        Level of current storage backend.

        Returns
        -------
        Level: StorageLevel
            storage level.
        """

    @abstractmethod
    async def get(self, object_id, **kwargs) -> object:
        """
        Get object by key. For some backends, `columns` or `slice` can pass to get part of data.

        Parameters
        ----------
        object_id : object id
            Object id to get.

        kwargs:
            Additional keyword arguments

        Returns
        -------
        Python object
        """

    @abstractmethod
    async def put(self, obj, importance: int = 0) -> ObjectInfo:
        """
        Put object into storage with object_id.

        Parameters
        ----------
        obj : python object
            Object to put.

        importance: int
             The priority to spill when storage is full

        Returns
        -------
        ObjectInfo
            object information including size, raw_size, device
        """

    @abstractmethod
    async def delete(self, object_id):
        """
        Delete object from storage by object_id.

        Parameters
        ----------
        object_id
            object id
        """

    @abstractmethod
    async def object_info(self, object_id) -> ObjectInfo:
        """
        Get information about stored object.

        Parameters
        ----------
        object_id
            object id

        Returns
        -------
        ObjectInfo
            Object info including size, device and etc.
        """

    @abstractmethod
    async def open_writer(self, size=None) -> StorageFileObject:
        """
        Return a file-like object for writing.

        Parameters
        ----------
        size: int
            Maximum size in bytes

        Returns
        -------
        fileobj: StorageFileObject
        """

    @abstractmethod
    async def open_reader(self, object_id) -> StorageFileObject:
        """
        Return a file-like object for reading.

        Parameters
        ----------
        object_id
            Object id

        Returns
        -------
        fileobj: StorageFileObject
        """

    async def list(self) -> List:
        """
        List all stored objects in storage.

        Returns
        -------
        List of objects
        """

    async def fetch(self, object_id):
        """
        Fetch object to current worker.

        Parameters
        ----------
        object_id
            Object id.
        """

    async def pin(self, object_id):
        """
        Pin the data to prevent the data being released or spilled.

        Parameters
        ----------
        object_id
            object id
        """

    async def unpin(self, object_id):
        """
        Unpin the data, allow storage to release the data.

        Parameters
        ----------
        object_id
            object id
        """
