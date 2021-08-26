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

import sys
from typing import Any, List, Type, TypeVar, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....storage.base import StorageLevel, StorageFileObject
from ...cluster import StorageInfo
from ..core import StorageManagerActor, DataManagerActor, \
    DataInfo, WrappedStorageFileObject
from ..handler import StorageHandlerActor
from .core import AbstractStorageAPI

_is_windows = sys.platform.lower().startswith('win')
APIType = TypeVar('APIType', bound='StorageAPI')


class StorageAPI(AbstractStorageAPI):
    _storage_handler_ref: Union[StorageHandlerActor, mo.ActorRef]
    _data_manager_ref: Union[DataManagerActor, mo.ActorRef]

    def __init__(self,
                 address: str,
                 session_id: str,
                 band_name: str):
        self._address = address
        self._session_id = session_id
        self._band_name = band_name

    async def _init(self):
        self._storage_handler_ref = await mo.actor_ref(
            self._address, StorageHandlerActor.gen_uid(self._band_name))
        self._data_manager_ref = await mo.actor_ref(
            self._address, DataManagerActor.default_uid())

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str,
                     band_name: str = 'numa-0',
                     **kwargs) -> APIType:
        """
        Create storage API.

        Parameters
        ----------
        session_id: str
            session id

        address: str
            worker address

        band_name: str
            name of band, default as 'numa-0'

        Returns
        -------
        storage_api
            Storage api.
        """
        if kwargs:  # pragma: no cover
            raise TypeError(f'Got unexpected arguments: {",".join(kwargs)}')
        api = StorageAPI(address, session_id, band_name)
        await api._init()
        return api

    @mo.extensible
    async def get(self,
                  data_key: str,
                  conditions: List = None,
                  error: str = 'raise') -> Any:
        return await self._storage_handler_ref.get(
            self._session_id, data_key, conditions, error)

    @get.batch
    async def batch_get(self, args_list, kwargs_list):
        gets = []
        for args, kwargs in zip(args_list, kwargs_list):
            gets.append(
                self._storage_handler_ref.get.delay(
                    self._session_id, *args, **kwargs)
            )
        return await self._storage_handler_ref.get.batch(*gets)

    @mo.extensible
    async def put(self, data_key: str,
                  obj: object,
                  level: StorageLevel = None) -> DataInfo:
        return await self._storage_handler_ref.put(
            self._session_id, data_key, obj, level
        )

    @put.batch
    async def batch_put(self, args_list, kwargs_list):
        puts = []
        for args, kwargs in zip(args_list, kwargs_list):
            puts.append(
                self._storage_handler_ref.put.delay(
                    self._session_id, *args, **kwargs)
            )
        return await self._storage_handler_ref.put.batch(*puts)

    @mo.extensible
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
        return await self._data_manager_ref.get_data_infos(
            self._session_id, data_key, self._band_name)

    @mo.extensible
    async def delete(self, data_key: str, error: str = 'raise'):
        """
        Delete object.

        Parameters
        ----------
        data_key: str
            object key to delete
        error: str
            raise or ignore
        """
        await self._storage_handler_ref.delete(
            self._session_id, data_key, error=error)

    @delete.batch
    async def batch_delete(self, args_list, kwargs_list):
        deletes = []
        for args, kwargs in zip(args_list, kwargs_list):
            deletes.append(
                self._storage_handler_ref.delete.delay(
                    self._session_id, *args, **kwargs)
            )
        return await self._storage_handler_ref.delete.batch(*deletes)

    @mo.extensible
    async def fetch(self,
                    data_key: str,
                    level: StorageLevel = None,
                    band_name: str = None,
                    remote_address: str = None,
                    error: str = 'raise'):
        """
        Fetch object from remote worker or load object from disk.

        Parameters
        ----------
        data_key: str
            data key to fetch to current worker with specific level
        level: StorageLevel
            the storage level to put into, MEMORY as default
        band_name: BandType
            put data on specific band
        remote_address:
            remote address that stores the data
        error: str
            raise or ignore
        """
        await self._storage_handler_ref.fetch_batch(
            self._session_id, [data_key], level,
            remote_address, band_name, error)

    @fetch.batch
    async def batch_fetch(self, args_list, kwargs_list):
        extracted_args = []
        data_keys = []
        for args, kwargs in zip(args_list, kwargs_list):
            data_key, level, band_name, dest_address, error = \
                self.fetch.bind(*args, **kwargs)
            if extracted_args:
                assert extracted_args == (level, band_name, dest_address, error)
            extracted_args = (level, band_name, dest_address, error)
            data_keys.append(data_key)
        await self._storage_handler_ref.fetch_batch(
            self._session_id, data_keys, *extracted_args)

    @mo.extensible
    async def unpin(self, data_key: str,
                    error: str = 'raise'):
        """
        Unpin the data, allow storage to release the data.

        Parameters
        ----------
        data_key: str
            data key to unpin
        error: str
            raise or ignore
        """
        await self._storage_handler_ref.unpin(self._session_id,
                                              data_key, error)

    @unpin.batch
    async def batch_unpin(self, args_list, kwargs_list):
        unpins = []
        for args, kwargs in zip(args_list, kwargs_list):
            data_key, error = self.unpin.bind(*args, **kwargs)
            unpins.append(
                self._storage_handler_ref.unpin.delay(
                    self._session_id, data_key, error)
            )
        return await self._storage_handler_ref.unpin.batch(*unpins)

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
                          level: StorageLevel) -> WrappedStorageFileObject:
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
        List all stored data_keys in storage.

        Parameters
        ----------
        level: StorageLevel
            the storage level to list all objects

        Returns
        -------
            list of data keys
        """
        return await self._storage_handler_ref.list(level=level)

    async def get_storage_level_info(self,
                                     level: StorageLevel) -> StorageInfo:
        """
        Get storage level's info.

        Parameters
        ----------
        level : StorageLevel
            Storage level.

        Returns
        -------
        storage_level_info : StorageInfo
        """
        return await self._storage_handler_ref.get_storage_level_info(level)


class MockStorageAPI(StorageAPI):
    @classmethod
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str,
                     **kwargs) -> APIType:
        from ..core import StorageManagerActor

        storage_configs = kwargs.get('storage_configs')
        if not storage_configs:
            if sys.platform == 'darwin':
                plasma_dir = '/tmp'
            else:
                plasma_dir = '/dev/shm'
            plasma_setup_params = dict(
                store_memory=10 * 1024 * 1024,
                plasma_directory=plasma_dir,
                check_dir_size=False)
            if _is_windows:
                storage_configs = {"shared_memory": {}}
            else:
                storage_configs = {
                    "plasma": plasma_setup_params,
                }

        storage_handler_cls = kwargs.pop('storage_handler_cls', StorageHandlerActor)
        await mo.create_actor(StorageManagerActor,
                              storage_configs,
                              storage_handler_cls=storage_handler_cls,
                              uid=StorageManagerActor.default_uid(),
                              address=address)
        return await super().create(address=address, session_id=session_id)

    @classmethod
    async def cleanup(cls: Type[APIType], address: str):
        await mo.destroy_actor(
            await mo.actor_ref(address, StorageManagerActor.default_uid()))
