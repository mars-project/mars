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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple

from .core import StorageFileObject


class StorageLevel(Enum):
    GPU = 1 << 0
    MEMORY = 1 << 1
    DISK = 1 << 2
    REMOTE = 1 << 3

    def __and__(self, other: "StorageLevel"):
        return self.value | other.value


class ObjectInfo:
    __slots__ = 'size', 'device', 'object_id'

    def __init__(self,
                 size: int = None,
                 device: int = None,
                 object_id: Any = None):
        self.size = size
        self.device = device
        self.object_id = object_id


class StorageBackend(ABC):
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

    async def prefetch(self, object_id):
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
