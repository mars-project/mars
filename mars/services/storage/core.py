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

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ... import oscar as mo
from ...lib.aio import AioFileObject
from ...oscar import ActorRef
from ...oscar.backends.allocate_strategy import IdleLabel, NoIdleSlot
from ...storage import StorageLevel, get_storage_backend
from ...storage.base import ObjectInfo, StorageBackend
from ...storage.core import StorageFileObject
from ...utils import calc_data_size, dataslots
from ..cluster import ClusterAPI
from ..meta import MetaAPI
from .errors import DataNotExist

logger = logging.getLogger(__name__)


def _build_data_info(storage_info: ObjectInfo, level, size):
    # todo handle multiple
    band = 'numa-0' if storage_info.device is None \
        else f'gpu-{storage_info.device}'
    if storage_info.size is None:
        store_size = size
    else:
        store_size = storage_info.size
    return DataInfo(storage_info.object_id, level, size, store_size, band)


class WrappedStorageFileObject(AioFileObject):
    """
    Wrap to hold ref after write close
    """
    def __init__(self,
                 file: StorageFileObject,
                 level: StorageLevel,
                 size: int,
                 session_id: str,
                 data_key: str,
                 storage_manager: Union[ActorRef, "StorageManagerActor"],
                 storage_handler: StorageBackend
                 ):
        super().__init__(file)
        self._size = size
        self._level = level
        self._session_id = session_id
        self._data_key = data_key
        self._storage_manager = storage_manager
        self._storage_handler = storage_handler

    def __getattr__(self, item):
        return getattr(self._file, item)

    async def clean_up(self):
        self._file.close()

    async def close(self):
        self._file.close()
        if 'w' in self._file._mode:
            object_info = await self._storage_handler.object_info(self._file._object_id)
            data_info = _build_data_info(object_info, self._level, self._size)
            await self._storage_manager.put_data_info(
                self._session_id, self._data_key, data_info, object_info)


class StorageQuota:
    def __init__(self, total_size: Optional[int]):
        self._total_size = total_size
        self._used_size = 0
        self._lock = asyncio.Semaphore()

    @property
    def total_size(self):
        return self._total_size

    @property
    def used_size(self):
        return self._used_size

    async def update(self, size: int):
        async with self._lock:
            if self._total_size is not None:
                self._total_size += size

    async def request(self, size: int) -> bool:
        async with self._lock:
            if self._total_size is None:
                self._used_size += size
                return True
            elif self._used_size + size >= self._total_size:
                return False
            else:
                self._used_size += size
                return True

    async def release(self, size: int):
        async with self._lock:
            self._used_size -= size


@dataslots
@dataclass
class DataInfo:
    object_id: object
    level: StorageLevel
    memory_size: int
    store_size: int
    band: str = None


@dataslots
@dataclass
class InternalDataInfo:
    data_info: DataInfo
    object_info: ObjectInfo


class DataManager:
    def __init__(self):
        # mapping key is (session_id, data_key)
        # mapping value is list of InternalDataInfo
        self._data_key_to_info: Dict[tuple, List[InternalDataInfo]] = defaultdict(list)
        self._data_info_list = dict()
        for level in StorageLevel.__members__.values():
            self._data_info_list[level] = dict()

    def put(self,
            session_id: str,
            data_key: str,
            data_info: DataInfo,
            object_info: ObjectInfo):
        info = InternalDataInfo(data_info, object_info)
        self._data_key_to_info[(session_id, data_key)].append(info)
        self._data_info_list[data_info.level][(session_id, data_key)] = object_info

    def get_infos(self,
                  session_id: str,
                  data_key: str) -> List[DataInfo]:
        if (session_id, data_key) not in self._data_key_to_info:  # pragma: no cover
            raise DataNotExist(f'Data key {session_id, data_key} not exists.')
        return [info.data_info for info in
                self._data_key_to_info.get((session_id, data_key))]

    def get_info(self,
                 session_id: str,
                 data_key: str) -> DataInfo:
        # if the data is stored in multiply levels,
        # return the lowest level info
        if (session_id, data_key) not in self._data_key_to_info:  # pragma: no cover
            raise DataNotExist(f'Data key {session_id, data_key} not exists.')
        infos = sorted(self._data_key_to_info.get((session_id, data_key)),
                       key=lambda x: x.data_info.level)
        return infos[0].data_info

    def delete(self,
               session_id: str,
               data_key: str,
               level: StorageLevel):
        if (session_id, data_key) in self._data_key_to_info:
            self._data_info_list[level].pop((session_id, data_key))
            infos = self._data_key_to_info[(session_id, data_key)]
            rest = [info for info in infos if info.data_info.level != level]
            if len(rest) == 0:
                del self._data_key_to_info[(session_id, data_key)]
            else:  # pragma: no cover
                self._data_key_to_info[(session_id, data_key)] = rest


class StorageHandlerActor(mo.Actor):
    def __init__(self,
                 storage_init_params: Dict,
                 storage_manager_ref: mo.ActorRef):
        self._storage_init_params = storage_init_params
        self._storage_manager_ref = storage_manager_ref

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
        if conditions is None:
            res = yield self._clients[data_info.level].get(
                data_info.object_id)
            raise mo.Return(res)
        else:
            try:
                data = await self._clients[data_info.level].get(
                    data_info.object_id, conditions=conditions)
                raise mo.Return(data)
            except NotImplementedError:
                data = yield self._clients[data_info.level].get(
                    data_info.object_id)
                try:
                    sliced_value = data.iloc[tuple(conditions)]
                except AttributeError:
                    sliced_value = data[tuple(conditions)]
                raise mo.Return(sliced_value)

    async def put(self,
                  session_id: str,
                  data_key: str,
                  obj: object,
                  level: StorageLevel) -> DataInfo:
        size = calc_data_size(obj)
        await self._storage_manager_ref.request_quota(size, level=level)
        object_info = await self._clients[level].put(obj)
        if object_info.size is not None and size != object_info.size:
            await self._storage_manager_ref.update_quota(
                object_info.size - size, level=level)
        data_info = _build_data_info(object_info, level, size)
        await self._storage_manager_ref.put_data_info(
            session_id, data_key, data_info, object_info)
        return data_info

    async def delete(self,
                     session_id: str,
                     data_key: str,
                     error='raise'):
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')
        try:
            infos = await self._storage_manager_ref.get_data_infos(
                session_id, data_key)
        except DataNotExist:
            if error == 'raise':
                raise
            else:
                return
        for info in infos or []:
            level = info.level
            await self._storage_manager_ref.delete_data_info(
                session_id, data_key, level)
            yield self._clients[level].delete(info.object_id)
            await self._storage_manager_ref.release_quota(info.store_size, level)

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
                          level: StorageLevel) -> WrappedStorageFileObject:
        await self._storage_manager_ref.request_quota(size, level=level)
        writer = await self._clients[level].open_writer(size)
        return WrappedStorageFileObject(writer, level, size, session_id, data_key,
                                        self._storage_manager_ref, self._clients[level])

    async def object_info(self,
                          session_id: str,
                          data_key: str,):
        data_info = await self._storage_manager_ref.get_data_info(
            session_id, data_key)
        return await self._clients[data_info.level].object_info(
                    data_info.object_id)

    async def list(self, level: StorageLevel) -> List:
        return await self._clients[level].list()

    async def fetch(self,
                    session_id: str,
                    data_key: str):
        if StorageLevel.REMOTE not in self._clients:
            raise NotImplementedError
        else:  # pragma: no cover
            data_info = await self._storage_manager_ref.fetch_data_info(
                session_id, data_key)
            await self._clients[StorageLevel.REMOTE].fetch(data_info.object_id)


class StorageManagerActor(mo.Actor):
    def __init__(self,
                 storage_configs: Dict,
                 transfer_block_size: int = None
                 ):
        self._storage_configs = storage_configs
        # params to init and teardown
        self._init_params = dict()
        self._teardown_params = dict()
        # pinned_keys
        self._pinned_keys = []
        # stores the mapping from data key to storage info
        self._data_manager = DataManager()
        self._supervisor_address = None

        # transfer config
        self._transfer_block_size = transfer_block_size

    async def __post_create__(self):
        from .transfer import SenderManagerActor, ReceiverManagerActor

        # setup storage backend
        quotas = dict()

        for backend, setup_params in self._storage_configs.items():
            client = await self._setup_storage(backend, setup_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    quotas[level] = StorageQuota(client.size)
        self._quotas = quotas

        # create handler actors for every process
        strategy = IdleLabel(None, 'storage_handler')
        self._storage_handler_refs = []
        while True:
            try:
                handler_ref = await mo.create_actor(
                    StorageHandlerActor, self._init_params,
                    self.ref(), uid=StorageHandlerActor.default_uid(),
                    address=self.address, allocate_strategy=strategy)
                self._storage_handler_refs.append(handler_ref)
            except NoIdleSlot:
                break

        self._storage_handler = await mo.actor_ref(address=self.address,
                                                   uid=StorageHandlerActor.default_uid())

        # create actor for transfer
        while True:
            try:
                sender_strategy = IdleLabel('io', 'sender')
                await mo.create_actor(
                    SenderManagerActor, uid=SenderManagerActor.default_uid(),
                    address=self.address, allocate_strategy=sender_strategy)

                receiver_strategy = IdleLabel('io', 'receiver')
                await mo.create_actor(ReceiverManagerActor, address=self.address,
                                      uid=ReceiverManagerActor.default_uid(),
                                      allocate_strategy=receiver_strategy)
            except NoIdleSlot:
                break

    async def __pre_destroy__(self):
        for backend, teardown_params in self._teardown_params.items():
            backend_cls = get_storage_backend(backend)
            await backend_cls.teardown(**teardown_params)

    async def _setup_storage(self,
                             storage_backend: str,
                             storage_config: Dict):
        backend = get_storage_backend(storage_backend)
        storage_config = storage_config or dict()
        init_params, teardown_params = await backend.setup(**storage_config)
        client = backend(**init_params)
        self._init_params[storage_backend] = init_params
        self._teardown_params[storage_backend] = teardown_params
        return client

    async def _get_meta_api(self, session_id: str):
        if self._supervisor_address is None:
            cluster_api = await ClusterAPI.create(self.address)
            self._supervisor_address = (await cluster_api.get_supervisors())[0]
        return await MetaAPI.create(session_id=session_id,
                                    address=self._supervisor_address)

    def get_client_params(self):
        return self._init_params

    async def request_quota(self,
                            size: int,
                            level: StorageLevel):
        if await self._quotas[level].request(size):
            return
        else:  # pragma: no cover
            raise NotImplementedError('Spill is not supported now')

    async def update_quota(self,
                           size: int,
                           level: StorageLevel):
        await self._quotas[level].update(size)

    async def release_quota(self,
                            size: int,
                            level: StorageLevel
                            ):
        await self._quotas[level].release(size)

    def get_quota(self, level: StorageLevel):
        return self._quotas[level].total_size, self._quotas[level].used_size

    async def fetch(self,
                    session_id: str,
                    data_key: str,
                    level: StorageLevel,
                    address: str,
                    band_name: str):
        from .transfer import SenderManagerActor

        try:
            info = self._data_manager.get_info(session_id, data_key)
            self.pin(info.object_id)
        except DataNotExist:
            # Not exists in local, fetch from remote worker
            try:
                yield self._storage_handler.fetch(session_id, data_key)
            except NotImplementedError:
                meta_api = await self._get_meta_api(session_id)
                if address is None:
                    address = (await meta_api.get_chunk_meta(
                        data_key, fields=['bands']))['bands'][0][0]
                sender_ref = await mo.actor_ref(
                    address=address, uid=SenderManagerActor.default_uid())
                yield sender_ref.send_data(session_id, data_key,
                                           self.address, level)
                await meta_api.add_chunk_bands(data_key, [(address, band_name or 'numa-0')])

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
                      data_info: DataInfo,
                      object_info: Union[ObjectInfo] = None):
        self._data_manager.put(session_id, data_key, data_info, object_info)

    async def fetch_data_info(self,
                              session_id: str,
                              data_key: str) -> DataInfo:
        meta_api = await self._get_meta_api(session_id)
        address = (await meta_api.get_chunk_meta(
            data_key, fields=['bands']))['bands'][0][0]
        remote_manager_ref = await mo.actor_ref(uid=StorageManagerActor.default_uid(),
                                                address=address)
        data_info = yield remote_manager_ref.get_data_info(session_id, data_key)
        self.put_data_info(session_id, data_key, data_info, None)
        raise mo.Return(data_info)

    def delete_data_info(self,
                         session_id: str,
                         data_key: str,
                         level: StorageLevel):
        self._data_manager.delete(session_id, data_key, level)

    def pin(self, object_id):
        self._pinned_keys.append(object_id)

    def unpin(self, object_id):
        self._pinned_keys.remove(object_id)
