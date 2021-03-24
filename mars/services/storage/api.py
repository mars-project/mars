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

from typing import List

from ... import oscar as mo
from ...storage.base import ObjectInfo, StorageLevel, StorageFileObject
from ...utils import calc_data_size, extensible
from .core import DataManagerActor, StorageHandlerActor, \
    StorageManagerActor, DataInfo


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
        self._data_manager_ref = await mo.actor_ref(
            self._address, DataManagerActor.default_uid())

    @classmethod
    async def create(cls,
                     session_id: str,
                     address: str,
                     **kwargs):
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
    async def get(self, data_key: str, conditions: List = None):
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
        data_info = await self._data_manager_ref.get_info(self._session_id, data_key)
        return await self._storage_handler_ref.get(
            data_info.object_id, data_info.level, conditions)

    @extensible
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
        size = calc_data_size(obj)
        await self._storage_manager_ref.allocate_size(size, level=level)
        object_info = await self._storage_handler_ref.put(obj, level)
        if object_info.size is not None and size != object_info.size:
            await self._storage_manager_ref.update_quota(
                object_info.size - size, level=level)
        data_info = DataInfo(object_info.object_id, level, size)
        return await self._data_manager_ref.put(
            self._session_id, data_key, data_info)

    @extensible
    async def delete(self, data_key: str):
        """
        Delete object.

        Parameters
        ----------
        data_key: str
            object key to delete
        """
        infos = await self._data_manager_ref.get_infos(self._session_id, data_key)
        for info in infos or []:
            level = info.level
            await self._data_manager_ref.delete(self._session_id, data_key, level)
            await self._storage_handler_ref.delete(info.object_id, level)
            await self._storage_manager_ref.release_size(info.size, level)

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
        data_info = await self._data_manager_ref.get_info(
            self._session_id, data_key)
        return await self._storage_handler_ref.open_reader(
            data_info.object_id, data_info.level)

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
        await self._storage_manager_ref.allocate_size(size, level=level)
        writer = await self._storage_handler_ref.open_writer(size, level)
        data_info = DataInfo(writer.object_id, level, size)
        await self._data_manager_ref.put(
            self._session_id, data_key, data_info)
        return writer

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
    async def create(cls,
                     session_id: str,
                     address: str,
                     **kwargs):
        from .core import StorageManagerActor, DataManagerActor

        storage_configs = kwargs.get('storage_configs')
        await mo.create_actor(DataManagerActor,
                              uid=DataManagerActor.default_uid(),
                              address=address)
        await mo.create_actor(StorageManagerActor,
                              storage_configs,
                              uid=StorageManagerActor.default_uid(),
                              address=address)
        return await super().create(address=address, session_id=session_id)
