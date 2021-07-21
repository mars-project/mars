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
from ...utils import dataslots, extensible
from .errors import DataNotExist, StorageFull

logger = logging.getLogger(__name__)


def build_data_info(storage_info: ObjectInfo, level, size):
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
                 data_manager: Union[ActorRef, "DataManagerActor"],
                 storage_handler: StorageBackend
                 ):
        self._object_id = file.object_id
        super().__init__(file)
        self._size = size
        self._level = level
        self._session_id = session_id
        self._data_key = data_key
        self._data_manager = data_manager
        self._storage_handler = storage_handler

    def __getattr__(self, item):
        return getattr(self._file, item)

    async def clean_up(self):
        self._file.close()

    async def close(self):
        self._file.close()
        if self._object_id is None:
            # for some backends like vineyard,
            # object id is generated after write close
            self._object_id = self._file.object_id
        if 'w' in self._file.mode:
            object_info = await self._storage_handler.object_info(self._object_id)
            data_info = build_data_info(object_info, self._level, self._size)
            await self._data_manager.put_data_info(
                self._session_id, self._data_key, data_info, object_info)


class StorageQuotaActor(mo.Actor):
    def __init__(self,
                 data_manager: Union[mo.ActorRef, "DataManagerActor"],
                 level: StorageLevel,
                 total_size: Optional[Union[int, float]]):
        self._data_manager = data_manager
        self._total_size = total_size if total_size is None else total_size * 0.95
        self._used_size = 0
        self._level = level

    @classmethod
    def gen_uid(cls, level: StorageLevel):
        return f'storage_quota_{level}'

    def update_quota(self, size: int):
        if self._total_size is not None:
            self._used_size += size
        logger.debug('Update %s bytes of %s, used size now is %s',
                     size, self._level, self._used_size)

    def request_quota(self, size: int) -> bool:
        if self._total_size is not None and size > self._total_size:  # pragma: no cover
            raise StorageFull(f'Request size {size} is larger '
                              f'than total size {self._total_size}')
        if self._total_size is None:
            self._used_size += size
            return True
        elif self._used_size + size >= self._total_size:
            logger.debug('Request %s bytes of %s, used size now is %s,'
                         'space is not enough for the request', size, self._level, self._used_size)
            return False
        else:
            self._used_size += size
            logger.debug('Request %s bytes of %s, used size now is %s,'
                         'total size is %s', size, self._level, self._used_size, self._total_size)
            return True

    def release_quota(self, size: int):
        self._used_size -= size
        logger.debug('Release %s bytes of %s, used size now is %s,'
                     'total size is %s', size, self._level, self._used_size, self._total_size)

    def get_quota(self):
        return self._total_size, self._used_size


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


class DataManagerActor(mo.Actor):
    _data_key_to_info: Dict[tuple, List[InternalDataInfo]]

    def __init__(self):
        from .spill import FIFOStrategy

        # mapping key is (session_id, data_key)
        # mapping value is list of InternalDataInfo
        self._data_key_to_info = defaultdict(list)
        self._data_info_list = dict()
        self._spill_strategy = dict()
        # data key may be a tuple in some cases,
        # we record main key to manage their lifecycle
        self._main_key_to_sub_keys = defaultdict(set)
        for level in StorageLevel.__members__.values():
            self._data_info_list[level] = dict()
            self._spill_strategy[level] = FIFOStrategy(level)

    def _get_data_infos(self,
                        session_id: str,
                        data_key: str,
                        error: str) -> Union[List[DataInfo], None]:
        try:
            if (session_id, data_key) not in self._data_key_to_info:
                if (session_id, data_key) in self._main_key_to_sub_keys:
                    infos = []
                    for sub_key in self._main_key_to_sub_keys[(session_id, data_key)]:
                        infos.extend([info.data_info for info in
                                      self._data_key_to_info.get((session_id, sub_key))])
                    return infos
                raise DataNotExist(f'Data key {session_id, data_key} not exists.')
            return [info.data_info for info in
                    self._data_key_to_info.get((session_id, data_key))]
        except DataNotExist:
            if error == 'raise':
                raise
            else:
                return

    @extensible
    def get_data_infos(self,
                       session_id: str,
                       data_key: str,
                       error: str = 'raise') -> List[DataInfo]:
        return self._get_data_infos(session_id, data_key, error)

    def _get_data_info(self,
                       session_id: str,
                       data_key: str,
                       error: str = 'raise') -> Union[DataInfo, None]:
        # if the data is stored in multiply levels,
        # return the lowest level info
        if (session_id, data_key) not in self._data_key_to_info:
            if error == 'raise':
                raise DataNotExist(f'Data key {session_id, data_key} not exists.')
            else:
                return None
        infos = sorted(self._data_key_to_info.get((session_id, data_key)),
                       key=lambda x: x.data_info.level)
        return infos[0].data_info

    @extensible
    def get_data_info(self,
                      session_id: str,
                      data_key: str,
                      error: str = 'raise') -> Union[DataInfo, None]:
        return self._get_data_info(session_id, data_key, error)

    def _put_data_info(self,
                       session_id: str,
                       data_key: str,
                       data_info: DataInfo,
                       object_info: ObjectInfo = None):
        info = InternalDataInfo(data_info, object_info)
        self._data_key_to_info[(session_id, data_key)].append(info)
        self._data_info_list[data_info.level][(session_id, data_key)] = object_info
        if object_info is not None:
            self._spill_strategy[data_info.level].record_put_info(
                (session_id, data_key), data_info.store_size)
        if isinstance(data_key, tuple):
            self._main_key_to_sub_keys[(session_id, data_key[0])].update([data_key])

    @extensible
    def put_data_info(self,
                      session_id: str,
                      data_key: str,
                      data_info: DataInfo,
                      object_info: ObjectInfo = None):
        self._put_data_info(
            session_id, data_key, data_info, object_info=object_info)

    def _delete_data_info(self,
                          session_id: str,
                          data_key: str,
                          level: StorageLevel
                          ):
        if (session_id, data_key) in self._main_key_to_sub_keys:
            to_delete_keys = self._main_key_to_sub_keys[(session_id, data_key)]
        else:
            to_delete_keys = [data_key]
        logger.debug('Begin to delete data keys for level %s '
                     'in data manager: %s', level, to_delete_keys)
        for key in to_delete_keys:
            if (session_id, key) in self._data_key_to_info:
                self._data_info_list[level].pop((session_id, key))
                self._spill_strategy[level].record_delete_info((session_id, key))
                infos = self._data_key_to_info[(session_id, key)]
                rest = [info for info in infos if info.data_info.level != level]
                if len(rest) == 0:
                    del self._data_key_to_info[(session_id, key)]
                else:  # pragma: no cover
                    self._data_key_to_info[(session_id, key)] = rest
        logger.debug('Finish deleting data keys for level %s '
                     'in data manager: %s', level, to_delete_keys)

    @extensible
    def delete_data_info(self,
                         session_id: str,
                         data_key: str,
                         level: StorageLevel):
        self._delete_data_info(session_id, data_key, level)

    def list(self, level: StorageLevel):
        return list(self._data_info_list[level].keys())

    def pin(self, session_id, data_key):
        level = self.get_data_info(session_id, data_key).level
        self._spill_strategy[level].pin_data((session_id, data_key))

    def unpin(self, session_id, data_key, error: str = 'raise'):
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')
        try:
            level = self.get_data_info(session_id, data_key).level
            self._spill_strategy[level].unpin_data((session_id, data_key))
            return level
        except DataNotExist:
            if error == 'raise':
                raise
            else:
                return

    def get_spillable_size(self, level: StorageLevel):
        return self._spill_strategy[level].get_spillable_size()

    async def get_spill_keys(self, level: StorageLevel, size: int):
        if level.spill_level() not in self._spill_strategy:  # pragma: no cover
            raise RuntimeError(f'Spill level of {level} is not configured')
        return self._spill_strategy[level].get_spill_keys(size)


class StorageManagerActor(mo.Actor):
    """
    Storage manager actor, created only on main process, mainly to setup storage backends
    and create all the necessary actors for storage service.
    """
    _data_manager: Union[mo.ActorRef, DataManagerActor]

    def __init__(self,
                 storage_configs: Dict,
                 transfer_block_size: int = None,
                 **kwargs):
        from .handler import StorageHandlerActor

        self._handler_cls = kwargs.pop('storage_handler_cls', StorageHandlerActor)
        self._storage_configs = storage_configs
        # params to init and teardown
        self._init_params = dict()
        self._teardown_params = dict()

        self._supervisor_address = None

        # transfer config
        self._transfer_block_size = transfer_block_size

    async def __post_create__(self):
        from .handler import StorageHandlerActor
        from .spill import SpillManagerActor
        from .transfer import SenderManagerActor, ReceiverManagerActor

        # stores the mapping from data key to storage info
        self._data_manager = await mo.create_actor(
            DataManagerActor,
            uid=DataManagerActor.default_uid(),
            address=self.address)

        # setup storage backend
        quotas = dict()
        spill_managers = dict()
        for backend, setup_params in self._storage_configs.items():
            client = await self._setup_storage(backend, setup_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    logger.debug('Create quota manager for %s,'
                                 ' total size is %s', level, client.size)
                    quotas[level] = await mo.create_actor(
                        StorageQuotaActor, self._data_manager,
                        level, client.size,
                        uid=StorageQuotaActor.gen_uid(level),
                        address=self.address,
                    )
                    spill_managers[level] = await mo.create_actor(
                        SpillManagerActor, level,
                        uid=SpillManagerActor.gen_uid(level),
                        address=self.address)

        # create handler actors for every process
        strategy = IdleLabel(None, 'StorageHandler')
        while True:
            try:
                await mo.create_actor(self._handler_cls,
                                      self._init_params,
                                      self._data_manager,
                                      spill_managers,
                                      quotas,
                                      uid=StorageHandlerActor.default_uid(),
                                      address=self.address,
                                      allocate_strategy=strategy)
            except NoIdleSlot:
                break

        # create actor for transfer
        sender_strategy = IdleLabel('io', 'sender')
        receiver_strategy = IdleLabel('io', 'receiver')
        while True:
            try:
                await mo.create_actor(
                    SenderManagerActor, uid=SenderManagerActor.default_uid(),
                    address=self.address, allocate_strategy=sender_strategy)

                await mo.create_actor(ReceiverManagerActor,
                                      quotas,
                                      address=self.address,
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
        client = backend(address=self.address, **init_params)
        self._init_params[storage_backend] = init_params
        self._teardown_params[storage_backend] = teardown_params
        return client

    def get_client_params(self):
        return self._init_params
