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

from abc import ABC
from typing import List

from ... import oscar as mo
from ...storage import get_storage_backend
from ...storage.base import ObjectInfo, StorageLevel, StorageFileObject
from ...utils import calc_data_size
from .core import KeyMapperActor, StorageManagerActor


class StorageAPI(ABC):
    def __init__(self,
                 address: str,
                 session_id: str):
        self._address = address
        self._session_id = session_id

    async def _init(self):
        self._storage_manager_ref = await mo.actor_ref(
            self._address, StorageManagerActor.default_uid())
        self._key_mapper_ref = await mo.actor_ref(
            self._address, KeyMapperActor.default_uid())

        client_init_params = await self._storage_manager_ref.get_client_params()
        clients = dict()
        for backend, params in client_init_params.items():
            storage_cls = get_storage_backend(backend)
            client = storage_cls(**params)
            if client.level & StorageLevel.MEMORY:
                clients[StorageLevel.MEMORY] = client
            if client.level & StorageLevel.DISK:
                clients[StorageLevel.DISK] = client
            if client.level & StorageLevel.GPU:
                clients[StorageLevel.GPU] = client
        self._clients = clients

    @classmethod
    async def create(cls,
                     session_id: str,
                     address: str):
        api = StorageAPI(address, session_id)
        await api._init()
        return api

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
        object_id, level = await self._key_mapper_ref.get_lowest(self._session_id, data_key)
        return await self._clients[level].get(object_id, conditions=conditions)

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
        await self._allocate(size, level=level)
        object_info = await self._clients[level].put(obj)
        await self._key_mapper_ref.put(
            self._session_id, data_key, object_info.object_id, level=level)
        return object_info

    async def delete(self, data_key: str):
        """
        Delete object.

        Parameters
        ----------
        data_key: str
            object key to delete
        """
        infos = await self._key_mapper_ref.get(self._session_id, data_key)
        for info in infos or []:
            level = info[1]
            await self._key_mapper_ref.delete(self._session_id, data_key, level)
            await self._clients[level].delete(info[0])

    async def prefetch(self, data_key: str,
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
        infos = await self._key_mapper_ref.get(self._session_id, data_key)
        if infos is None:
            # fetch data from remote
            raise NotImplementedError
        else:
            infos = sorted(infos, key=lambda x: x[0])
            object_id, min_level = infos[0]
            if min_level > level:
                object_info = await self._clients[min_level].object_info(object_id)
                size = object_info.size
                await self._allocate(size, level=level)
                await self._storage_manager_ref.copy_to(data_key, level)
                await self._storage_manager_ref.pin(object_id)

    async def _allocate(self,
                        size: int,
                        level: StorageLevel):
        """
        Allocate size for storing, called in put and prefetch.
        It will send a quota request to main process.

        Parameters
        ----------
        size: int
            the size to allocate
        level: StorageLevel
            the storage level to allocate
        """
        has_space, spill_size = await self._storage_manager_ref.allocate_size(size, level)
        if not has_space:
            await self._storage_manager_ref.do_spill(spill_size, level=level)
            await self._allocate(size, level=level)

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
        object_id, level = await self._key_mapper_ref.get_lowest(self._session_id, data_key)
        reader = await self._clients[level].open_reader(object_id)
        return reader

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
        await self._allocate(size, level=level)
        writer = await self._clients[level].open_writer(size)
        await self._key_mapper_ref.put(self._session_id, data_key, writer.object_id, level=level)
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
        return await self._clients[level].list()


class MockStorageApi(StorageAPI):
    @classmethod
    async def create(cls,
                     address: str,
                     session_id: str,
                     **kwargs):
        from .core import StorageManagerActor, KeyMapperActor

        storage_configs = kwargs.get('storage_configs')
        await mo.create_actor(KeyMapperActor,
                              uid=KeyMapperActor.default_uid(),
                              address=address)
        await mo.create_actor(StorageManagerActor,
                              storage_configs,
                              uid=StorageManagerActor.default_uid(),
                              address=address)
        return await super().create(address=address, session_id=session_id)
