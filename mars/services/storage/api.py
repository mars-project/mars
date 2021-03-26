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

from typing import Any, List, Type, TypeVar

from ... import oscar as mo
from ...storage.base import StorageLevel, StorageFileObject
from ...utils import extensible
from .core import StorageHandlerActor, StorageManagerActor, DataInfo

APIType = TypeVar('APIType', bound='StorageAPI')


class StorageAPI:
    def __init__(self,
                 address: str,
                 session_id: str):
        self._address = address
        self._session_id = session_id

    async def _init(self):
        self._storage_handler_ref = await mo.actor_ref(
            self._address, StorageHandlerActor.default_uid())
        self._storage_manager_ref = await mo.actor_ref(
            self._address, StorageManagerActor.default_uid())

    @classmethod
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str,
                     **kwargs) -> APIType:
        """
        Create storage API.

        Parameters
        ----------
        session_id: str
            session id

        address: str
            worker address

        Returns
        -------
        storage_api
            Storage api.
        """
        if kwargs:  # pragma: no cover
            raise TypeError(f'Got unexpected arguments: {",".join(kwargs)}')
        api = StorageAPI(address, session_id)
        await api._init()
        return api

    @extensible
    async def get(self, data_key: str, conditions: List = None) -> Any:
        """
        Get object by data key.

        Parameters
        ----------
        data_key: str
            date key to get.

        conditions: List
            Index conditions to pushdown

        Returns
        -------
            object
        """
        return await self._storage_handler_ref.get(
            self._session_id, data_key, conditions)

    @extensible
    async def put(self, data_key: str,
                  obj: object,
                  level: StorageLevel = StorageLevel.MEMORY) -> DataInfo:
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
        return await self._storage_handler_ref.put(
            self._session_id, data_key, obj, level
        )

    @extensible
    async def get_infos(self, data_key: str) -> List[DataInfo]:
        """
        Get data information items for specific data key

        Parameters
        ----------
        data_key

        Returns
        -------
        out
            List of information for specified key
        """
        return await self._storage_manager_ref.get_data_infos(
            self._session_id, data_key
        )

    @extensible
    async def delete(self, data_key: str):
        """
        Delete object.

        Parameters
        ----------
        data_key: str
            object key to delete
        """
        await self._storage_handler_ref.delete(
            self._session_id, data_key)

    @extensible
    async def prefetch(self,
                       data_key: str,
                       level: StorageLevel = StorageLevel.MEMORY):
        """
        Fetch object from remote worker ot load object from disk.

        Parameters
        ----------
        data_key: str
            data key to fetch to current worker with specific level
        level: StorageLevel
            the storage level to put into, MEMORY as default

        """
        await self._storage_manager_ref.prefetch(
            self._session_id, data_key, level)

    @extensible
    async def unpin(self, data_key: str):
        """
        Unpin the data, allow storage to release the data.

        Parameters
        ----------
        data_key: str
            data key to unpin

        """
        await self._storage_manager_ref.unpin(data_key)

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
        return await self._storage_handler_ref.open_reader(
            self._session_id, data_key)

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
        return await self._storage_handler_ref.open_writer(
            self._session_id, data_key, size, level
        )

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
        return await self._storage_handler_ref.list(level=level)


class MockStorageApi(StorageAPI):
    @classmethod
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str,
                     **kwargs) -> APIType:
        from .core import StorageManagerActor

        storage_configs = kwargs.get('storage_configs')
        await mo.create_actor(StorageManagerActor,
                              storage_configs,
                              uid=StorageManagerActor.default_uid(),
                              address=address)
        return await super().create(address=address, session_id=session_id)
