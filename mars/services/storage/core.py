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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union

from ... import oscar as mo
from ...oscar.backends.mars.allocate_strategy import IdleLabel, NoIdleSlot
from ...storage import StorageLevel, get_storage_backend
from ...storage.base import ObjectInfo, StorageFileObject
from ...utils import calc_data_size, dataslots

logger = logging.getLogger(__name__)


class StorageQuota:
    def __init__(self, total_size: Union[int, None]):
        self._total_size = total_size
        self._used_size = 0

    @property
    def total_size(self):
        return self._total_size

    @property
    def used_size(self):
        return self._used_size

    def update(self, size: int):
        if self._total_size is not None:
            self._total_size += size

    def request(self, size: int) -> bool:
        if self._total_size is None:
            self._used_size += size
            return True
        elif self._used_size + size >= self._total_size:
            return False
        else:
            self._used_size += size
            return True

    def release(self, size: int):
        self._used_size -= size


@dataslots
@dataclass
class DataInfo:
    object_id: object
    level: StorageLevel
    size: int
    band: str = None


class DataManager:
    def __init__(self):
        # mapping key is (session_id, data_key)
        # mapping value is list of DataInfo
        self._mapping = defaultdict(list)

    def put(self,
            session_id: str,
            data_key: str,
            data_info: DataInfo):
        self._mapping[(session_id, data_key)].append(data_info)

    def get_infos(self,
                  session_id: str,
                  data_key: str) -> List[DataInfo]:
        return self._mapping.get((session_id, data_key))

    def get_info(self,
                 session_id: str,
                 data_key: str) -> DataInfo:
        # if the data is stored in multiply levels,
        # return the lowest level info
        infos = sorted(self._mapping.get((session_id, data_key)),
                       key=lambda x: x.level)
        return infos[0]

    def delete(self,
               session_id: str,
               data_key: str,
               level: StorageLevel):
        if (session_id, data_key) in self._mapping:
            infos = self._mapping[(session_id, data_key)]
            rest = [info for info in infos if info.level != level]
            if len(rest) == 0:
                del self._mapping[(session_id, data_key)]
            else:  # pragma: no cover
                self._mapping[(session_id, data_key)] = rest


class StorageHandlerActor(mo.Actor):
    def __init__(self,
                 storage_init_params: Dict,
                 storage_manager_ref: "StorageManagerActor"):
        self._storage_init_params = storage_init_params
        self._storage_manager_ref = storage_manager_ref

    @staticmethod
    def _build_data_info(storage_info: ObjectInfo, level, size):
        # todo handle multiple
        band = 'numa-0' if storage_info.device is not None \
            else f'gpu-{storage_info.device}'
        return DataInfo(storage_info.object_id, level, size, band)

    async def __post_create__(self):
        self._clients = clients = dict()
        for backend, init_params in self._storage_init_params.items():
            storage_cls = get_storage_backend(backend)
            client = storage_cls(**init_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    clients[level] = client

    async def get(self,
                  session_id: str,
                  data_key: str,
                  conditions: List = None):
        data_info = await self._storage_manager_ref.get_data_info(
            session_id, data_key)
        return await self._clients[data_info.level].get(
            data_info.object_id, conditions=conditions)

    async def put(self,
                  session_id: str,
                  data_key: str,
                  obj: object,
                  level: StorageLevel) -> DataInfo:
        size = calc_data_size(obj)
        await self._storage_manager_ref.allocate_size(size, level=level)
        object_info = await self._clients[level].put(obj)
        if object_info.size is not None and size != object_info.size:
            await self._storage_manager_ref.update_quota(
                object_info.size - size, level=level)
        data_info = self._build_data_info(object_info, level, size)
        await self._storage_manager_ref.put_data_info(
            session_id, data_key, data_info)
        return data_info

    async def delete(self,
                     session_id: str,
                     data_key: str):
        infos = await self._storage_manager_ref.get_data_infos(
            session_id, data_key)
        for info in infos or []:
            level = info.level
            await self._storage_manager_ref.delete_data_info(
                session_id, data_key, level)
            await self._clients[level].delete(info.object_id)
            await self._storage_manager_ref.release_size(info.size, level)

    async def open_reader(self,
                          session_id: str,
                          data_key: str) -> StorageFileObject:
        data_info = await self._storage_manager_ref.get_data_info(
            session_id, data_key)
        reader = await self._clients[data_info.level].open_reader(
            data_info.object_id)
        return reader

    async def open_writer(self,
                          session_id: str,
                          data_key: str,
                          size: int,
                          level: StorageLevel) -> StorageFileObject:
        await self._storage_manager_ref.allocate_size(size, level=level)
        writer = await self._clients[level].open_writer(size)
        data_info = DataInfo(writer.object_id, level, size)
        await self._storage_manager_ref.put_data_info(
            session_id, data_key, data_info)
        return writer

    async def list(self, level: StorageLevel) -> List:
        return await self._clients[level].list()


class StorageManagerActor(mo.Actor):
    def __init__(self,
                 storage_configs: Dict,
                 ):
        self._storage_configs = storage_configs
        # params to init and teardown
        self._init_params = dict()
        self._teardown_params = dict()
        # pinned_keys
        self._pinned_keys = []
        # stores the mapping from data key to storage info
        self._data_manager = DataManager()

    async def __post_create__(self):
        # setup storage backend
        quotas = dict()
        for backend, setup_params in self._storage_configs.items():
            client = await self._setup_storage(backend, setup_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    quotas[level] = StorageQuota(client.size)

        # create handler actors for every process
        strategy = IdleLabel(None, 'StorageHandler')
        while True:
            try:
                await mo.create_actor(StorageHandlerActor,
                                      self._init_params,
                                      self.ref(),
                                      uid=StorageHandlerActor.default_uid(),
                                      address=self.address,
                                      allocate_strategy=strategy)
            except NoIdleSlot:
                break

        self._quotas = quotas

    async def __pre_destroy__(self):
        for backend, teardown_params in self._teardown_params.items():
            backend_cls = get_storage_backend(backend)
            await backend_cls.teardown(**teardown_params)

    async def _setup_storage(self,
                             storage_backend: str,
                             storage_config: Dict):
        backend = get_storage_backend(storage_backend)
        init_params, teardown_params = await backend.setup(**storage_config)
        client = backend(**init_params)
        self._init_params[storage_backend] = init_params
        self._teardown_params[storage_backend] = teardown_params
        return client

    def get_client_params(self):
        return self._init_params

    def allocate_size(self,
                      size: int,
                      level: StorageLevel):
        if self._quotas[level].request(size):
            return
        else:  # pragma: no cover
            raise NotImplementedError

    async def prefetch(self,
                       session_id: str,
                       data_key: str,
                       level: StorageLevel):
        infos = self._data_manager.get_infos(session_id, data_key)
        infos = sorted(infos, key=lambda x: x.level)
        self.pin(infos[0].object_id)

    def update_quota(self,
                     size: int,
                     level: StorageLevel):
        self._quotas[level].update(size)

    def release_size(self,
                     size: int,
                     level: StorageLevel
                     ):
        self._quotas[level].release(size)

    def get_data_infos(self,
                       session_id: str,
                       data_key: str) -> List[DataInfo]:
        return self._data_manager.get_infos(session_id, data_key)

    def get_data_info(self,
                      session_id: str,
                      data_key: str) -> DataInfo:
        return self._data_manager.get_info(session_id, data_key)

    def put_data_info(self,
                      session_id: str,
                      data_key: str,
                      data_info: DataInfo):
        self._data_manager.put(session_id, data_key, data_info)

    def delete_data_info(self,
                         session_id: str,
                         data_key: str,
                         level: StorageLevel):
        self._data_manager.delete(session_id, data_key, level)

    def pin(self, object_id):
        self._pinned_keys.append(object_id)

    def unpin(self, object_id):
        self._pinned_keys.remove(object_id)
