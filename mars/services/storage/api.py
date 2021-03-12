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
from typing import List

from ...storage.base import ObjectInfo, StorageLevel, StorageFileObject


class StorageAPI(ABC):
    def __init__(self, session_id):
        self._session_id = session_id

    @abstractmethod
    async def get(self, data_key: str, **kwargs):
        """
        Get object by data key.

        Parameters
        ----------
        data_key: str
            date key to get.

        Returns
        -------
            object
        """

    @abstractmethod
    async def put(self, data_key: str,
                  obj: object,
                  level: StorageLevel = StorageLevel.MEMORY) -> ObjectInfo:
        """
        Put object into storage.

        Parameters
        ----------
        data_key: str
            data key to put.
        obj: object
            object to put.
        level: StorageLevel
            the storage level to put into, MEMORY as default

        Returns
        -------
        object information: ObjectInfo
            the put object information
        """

    @abstractmethod
    async def delete(self, data_key: str):
        """
        Delete object.

        Parameters
        ----------
        data_key: str
            object key to delete

        """

    @abstractmethod
    async def prefetch(self, data_key: str,
                       level: StorageLevel = StorageLevel.MEMORY):
        """
        Fetch object to current worker.

        Parameters
        ----------
        data_key: str
            data key to fetch to current worker
        level: StorageLevel
            the storage level to put into, MEMORY as default

        """

    async def allocate(self, size: int,
                       level: StorageLevel) -> bool:
        """
        Allocate size for storing, called in put and prefetch.
        It will send a quota request to main process.

        Parameters
        ----------
        size: int
            the size to allocate
        level: StorageLevel
            the storage level to allocate

        Returns
        -------
            return True if request is accepted, False when rejected.
        """

    @abstractmethod
    async def pin(self, data_key: str):
        """
        Pin the data to prevent the data being released or spilled.

        Parameters
        ----------
        data_key: str
            data key to pin

        """

    @abstractmethod
    async def unpin(self, data_key: str):
        """
        Unpin the data, allow storage to release the data.

        Parameters
        ----------
        data_key: str
            data key to unpin

        """

    @abstractmethod
    async def open_reader(self, data_key: str) -> StorageFileObject:
        """
        Return a file-like object for reading.

        Parameters
        ----------
        data_key: str
            data key

        Returns
        -------
            return a file-like object.
        """

    @abstractmethod
    async def open_writer(self,
                          data_key: str,
                          size: int,
                          level: StorageLevel) -> StorageFileObject:
        """
        Return a file-like object for writing data.

        Parameters
        ----------
        data_key: str
            data key
        size: int
            the total size of data
        level: StorageLevel
            the storage level to write

        Returns
        -------
            return a file-like object.
        """

    @abstractmethod
    async def list(self, level: StorageLevel) -> List:
        """
        List all stored objects in storage.

        Parameters
        ----------
        level: StorageLevel
            the storage level to list all objects

        Returns
        -------
            list of data keys
        """
