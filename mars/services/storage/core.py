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
from typing import Any, Dict, List, Optional, Union

from ... import oscar as mo
from ...lib.aio import AioFileObject
from ...oscar import ActorRef
from ...oscar.backends.allocate_strategy import IdleLabel, NoIdleSlot
from ...storage import StorageLevel, get_storage_backend
from ...storage.base import ObjectInfo, StorageBackend
from ...storage.core import StorageFileObject
from ...utils import calc_data_size, dataslots, extensible
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
            data_info = _build_data_info(object_info, self._level, self._size)
            await self._data_manager.put_data_info(
                self._session_id, self._data_key, data_info, object_info)


class StorageQuotaActor(mo.Actor):
    def __init__(self,
                 level: StorageLevel,
                 total_size: Optional[Union[int, float]]):
        self._total_size = total_size if total_size is None else total_size * 0.95
        self._used_size = 0
        self._level = level

    @classmethod
    def gen_uid(cls, level: StorageLevel):
        return f'storage_quota_{level}'

    @property
    def total_size(self):
        return self._total_size

    @property
    def used_size(self):
        return self._used_size

    @property
    def level(self):
        return self._level

    def update_quota(self, size: int):
        if self._total_size is not None:
            self._used_size += size

    def request_quota(self, size: int) -> bool:
        logger.debug(f'Request {size} bytes of {self.level}, '
                     f'used size is {self.used_size},'
                     f'total size is {self.total_size}')
        if self._total_size is None:
            self._used_size += size
            return True
        elif self._used_size + size >= self._total_size:
            return False
        else:
            self._used_size += size
            return True

    def release_quota(self, size: int):
        self._used_size -= size
        logger.debug(f'Release {size} bytes of {self.level}, '
                     f'used size now is {self.used_size},'
                     f'total size is {self.total_size}')

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
    def __init__(self):
        from .spill import FIFOStrategy

        # mapping key is (session_id, data_key)
        # mapping value is list of InternalDataInfo
        self._data_key_to_info: Dict[tuple, List[InternalDataInfo]] = defaultdict(list)
        self._data_info_list = dict()
        self._spill_strategy = dict()
        # data key may be a tuple in some cases,
        # we record main key to manage their lifecycle
        self._main_key_to_sub_keys = defaultdict(set)
        for level in StorageLevel.__members__.values():
            self._data_info_list[level] = dict()
            self._spill_strategy[level] = FIFOStrategy(level)

    def put(self,
            session_id: str,
            data_key: str,
            data_info: DataInfo,
            object_info: ObjectInfo):
        info = InternalDataInfo(data_info, object_info)
        self._data_key_to_info[(session_id, data_key)].append(info)
        self._data_info_list[data_info.level][(session_id, data_key)] = object_info
        if object_info is not None:
            self._spill_strategy[data_info.level].record_put_info(
                (session_id, data_key), data_info.store_size)
        if isinstance(data_key, tuple):
            self._main_key_to_sub_keys[(session_id, data_key[0])].update([data_key])

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
        logger.debug(f'Begin to delete data keys for level {level} '
                     f'in data manager: {to_delete_keys}')
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
        logger.debug(f'Finish deleting data keys for level {level} '
                     f'in data manager: {to_delete_keys}')

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
        except DataNotExist:
            if error == 'raise':
                raise
            else:
                return

    def get_spill_keys(self, level, size):
        return self._spill_strategy[level].get_spill_keys(size)


class StorageHandlerActor(mo.Actor):
    _quota_refs: Dict[StorageLevel,  Union["StorageQuotaActor", mo.ActorRef]]
    _data_manager_ref: Union["DataManagerActor", mo.ActorRef]

    def __init__(self,
                 storage_init_params: Dict,
                 data_manager_ref: Union["DataManagerActor", mo.ActorRef],
                 quota_refs: Dict[StorageLevel,
                                  Union["StorageQuotaActor", mo.ActorRef]]):
        self._storage_init_params = storage_init_params
        self._data_manager_ref = data_manager_ref
        self._quota_refs = quota_refs

    async def __post_create__(self):
        self._clients = clients = dict()
        for backend, init_params in self._storage_init_params.items():
            storage_cls = get_storage_backend(backend)
            client = storage_cls(**init_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    clients[level] = client

    async def _get_data(self, data_info, conditions):
        logger.debug(f'Begin to get data {data_info} with conditions {conditions}')
        if conditions is None:
            res = yield self._clients[data_info.level].get(
                data_info.object_id)
        else:
            try:
                res = yield self._clients[data_info.level].get(
                    data_info.object_id, conditions=conditions)
            except NotImplementedError:
                data = yield await self._clients[data_info.level].get(
                    data_info.object_id)
                try:
                    sliced_value = data.iloc[tuple(conditions)]
                except AttributeError:
                    sliced_value = data[tuple(conditions)]
                res = sliced_value
        logger.debug(f'Finish getting data {data_info} with conditions {conditions}')
        raise mo.Return(res)

    @extensible
    async def get(self,
                  session_id: str,
                  data_key: str,
                  conditions: List = None,
                  error: str = 'raise'):
        try:
            data_info = await self._data_manager_ref.get_data_info(
                session_id, data_key)
            data = yield self._get_data(data_info, conditions)
            raise mo.Return(data)
        except DataNotExist:
            if error == 'raise':
                raise

    def _get_data_info(self,
                       session_id: str,
                       data_key: str,
                       conditions: List = None,
                       error: str = 'raise'):
        info = self._data_manager_ref.get_data_info.delay(
            session_id, data_key, error)
        logger.debug(f'Get info of data {session_id}-{data_key}: '
                     f'{info}')
        return info, conditions

    @get.batch
    async def batch_get(self, args_list, kwargs_list):
        infos = []
        conditions_list = []
        for args, kwargs in zip(args_list, kwargs_list):
            info, conditions = self._get_data_info(*args, **kwargs)
            infos.append(info)
            conditions_list.append(conditions)
        data_infos = await self._data_manager_ref.get_data_info.batch(*infos)
        results = []
        for data_info, conditions in zip(data_infos, conditions_list):
            if data_info is None:
                results.append(None)
            else:
                result = yield self._get_data(data_info, conditions)
                results.append(result)
        raise mo.Return(results)

    @extensible
    async def put(self,
                  session_id: str,
                  data_key: str,
                  obj: object,
                  level: StorageLevel) -> DataInfo:
        size = calc_data_size(obj)
        await self._request_quota_with_spill(level, size)
        object_info = await self._clients[level].put(obj)
        data_info = _build_data_info(object_info, level, size)
        await self._data_manager_ref.put_data_info(
            session_id, data_key, data_info, object_info)
        if object_info.size is not None and data_info.memory_size != object_info.size:
            await self._quota_refs[level].update_quota(
                object_info.size - data_info.memory_size)
        return data_info

    @classmethod
    def _get_put_arg(cls,
                     session_id: str,
                     data_key: str,
                     obj: object,
                     level: StorageLevel):
        return session_id, data_key, obj, level, calc_data_size(obj)

    @put.batch
    async def batch_put(self, args_list, kwargs_list):
        objs = []
        data_keys = []
        session_id = None
        level = last_level = None
        sizes = []
        for args, kwargs in zip(args_list, kwargs_list):
            session_id, data_key, obj, level, size = \
                self._get_put_arg(*args, **kwargs)
            if last_level is not None:
                assert last_level == level
            last_level = level
            objs.append(obj)
            data_keys.append(data_key)
            sizes.append(size)

        await self._request_quota_with_spill(level, sum(sizes))

        data_infos = []
        put_infos = []
        for size, data_key, obj in zip(sizes, data_keys, objs):
            logger.debug(f'Begin to put data key {data_key}')
            object_info = await self._clients[level].put(obj)
            data_info = _build_data_info(object_info, level, size)
            data_infos.append(data_info)
            if object_info.size is not None and \
                    data_info.memory_size != object_info.size:
                # we request memory size before putting, when put finishes,
                # update quota to the true store size
                await self._quota_refs[level].update_quota(
                    object_info.size - data_info.memory_size)
            put_infos.append(
                self._data_manager_ref.put_data_info.delay(
                    session_id, data_key, data_info, object_info))
            logger.debug(f'Finish putting data key {data_key}, size is {size}, '
                         f'object_id is {data_info.object_id}')
        await self._data_manager_ref.put_data_info.batch(*put_infos)
        return data_infos

    def _get_data_infos_arg(self,
                            session_id: str,
                            data_key: str,
                            error: str = 'raise'):
        infos = self._data_manager_ref.get_data_infos.delay(
            session_id, data_key, error)
        return infos, session_id, data_key

    async def delete_object(self,
                            session_id: str,
                            data_key: Any,
                            object_id: Any,
                            level: StorageLevel):
        await self._data_manager_ref.delete_data_info(
            session_id, data_key, level)
        await self._clients[level].delete(object_id)

    @extensible
    async def delete(self,
                     session_id: str,
                     data_key: str,
                     error: str = 'raise'):
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')

        infos = await self._data_manager_ref.get_data_infos(
            session_id, data_key, error)
        if not infos:
            return

        for info in infos:
            level = info.level
            await self._data_manager_ref.delete_data_info(
                session_id, data_key, level)
            yield self._clients[level].delete(info.object_id)
            await self._quota_refs[level].release_quota(info.store_size)

    @delete.batch
    async def batch_delete(self, args_list, kwargs_list):
        get_infos = []
        session_id = None
        data_keys = []
        for args, kwargs in zip(args_list, kwargs_list):
            infos, session_id, data_key = self._get_data_infos_arg(*args, **kwargs)
            get_infos.append(infos)
            data_keys.append(data_key)
        infos_list = await self._data_manager_ref.get_data_infos.batch(*get_infos)

        delete_infos = []
        to_removes = []
        level_sizes = defaultdict(lambda: 0)
        for infos, data_key in zip(infos_list, data_keys):
            if not infos:
                # data not exist and error == 'ignore'
                continue
            for info in infos:
                level = info.level
                delete_infos.append(
                    self._data_manager_ref.delete_data_info.delay(
                        session_id, data_key, level))
                to_removes.append((level, info.object_id))
                level_sizes[level] += info.store_size

        if not delete_infos:
            # no data to remove
            return

        await self._data_manager_ref.delete_data_info.batch(*delete_infos)
        logger.debug(f'Begin to delete batch data {to_removes}')
        for level, object_id in to_removes:
            yield self._clients[level].delete(object_id)
        logger.debug(f'Finish deleting batch data {to_removes}')
        for level, size in level_sizes.items():
            await self._quota_refs[level].release_quota(size)

    async def open_reader(self,
                          session_id: str,
                          data_key: str) -> StorageFileObject:
        data_info = await self._data_manager_ref.get_data_info(
            session_id, data_key)
        reader = await self._clients[data_info.level].open_reader(
            data_info.object_id)
        return reader

    async def open_writer(self,
                          session_id: str,
                          data_key: str,
                          size: int,
                          level: StorageLevel) -> WrappedStorageFileObject:
        await self._request_quota_with_spill(level, size)
        writer = await self._clients[level].open_writer(size)
        return WrappedStorageFileObject(writer, level, size, session_id, data_key,
                                        self._data_manager_ref, self._clients[level])

    async def list(self, level: StorageLevel) -> List:
        return await self._clients[level].list()

    async def fetch(self,
                    session_id: str,
                    data_key: str):
        if StorageLevel.REMOTE not in self._clients:
            raise NotImplementedError
        else:
            data_info = await self._data_manager_ref.get_data_info(
                session_id, data_key)
            await self._clients[StorageLevel.REMOTE].fetch(data_info.object_id)

    async def _request_quota_with_spill(self,
                                        level: StorageLevel,
                                        size: int):
        if await self._quota_refs[level].request_quota(size):
            logger.debug(f'Request {size} bytes of {level} finished')
            return
        else:
            await self.spill(level, size)
            await self._quota_refs[level].request_quota(size)
            logger.debug(f'Spill is triggered, request {size} bytes of {level} finished')

    async def spill(self, level: StorageLevel, size: int):
        from .spill import spill

        await spill(size, level, self._data_manager_ref,
                    self, self._quota_refs[level])


class StorageManagerActor(mo.Actor):
    _data_manager: Union[mo.ActorRef, DataManagerActor]
    _storage_handler: Union[mo.ActorRef, StorageHandlerActor]

    def __init__(self,
                 storage_configs: Dict,
                 transfer_block_size: int = None
                 ):
        self._storage_configs = storage_configs
        # params to init and teardown
        self._init_params = dict()
        self._teardown_params = dict()

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
                    logger.debug(f'Create quota manager for {level},'
                                 f' total size is {client.size}')
                    quotas[level] = await mo.create_actor(
                        StorageQuotaActor, level, client.size,
                        uid=StorageQuotaActor.gen_uid(level),
                        address=self.address,
                    )

        self._quotas: Dict[StorageLevel, Union[mo.ActorRef, StorageQuotaActor]] = quotas

        # stores the mapping from data key to storage info
        self._data_manager = await mo.create_actor(
            DataManagerActor,
            uid=DataManagerActor.default_uid(),
            address=self.address)

        # create handler actors for every process
        strategy = IdleLabel(None, 'StorageHandler')
        while True:
            try:
                await mo.create_actor(StorageHandlerActor,
                                      self._init_params,
                                      self._data_manager,
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

        self._storage_handler = await mo.actor_ref(address=self.address,
                                                   uid=StorageHandlerActor.default_uid())

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

    async def fetch(self,
                    session_id: str,
                    data_key: str,
                    level: StorageLevel,
                    address: str,
                    band_name: str,
                    error: str):
        from .transfer import SenderManagerActor

        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')

        try:
            await self._data_manager.get_data_info(session_id, data_key)
            await self.pin(session_id, data_key)
        except DataNotExist:
            # Not exists in local, fetch from remote worker
            try:
                meta_api = await self._get_meta_api(session_id)
                if address is None:
                    # we get meta using main key when fetch shuffle data
                    main_key = data_key[0] if isinstance(data_key, tuple) else data_key
                    address = (await meta_api.get_chunk_meta(
                        main_key, fields=['bands']))['bands'][0][0]
                if address == self.address:
                    return
                logger.debug(f'Begin to fetch data {data_key} from {address}')
                if StorageLevel.REMOTE in self._quotas:
                    remote_manager_ref = await mo.actor_ref(uid=DataManagerActor.default_uid(),
                                                            address=address)
                    data_info = yield remote_manager_ref.get_data_info(session_id, data_key)
                    await self._data_manager.put_data_info(session_id, data_key, data_info, None)
                    yield self._storage_handler.fetch(session_id, data_key)
                else:
                    sender_ref = await mo.actor_ref(
                        address=address, uid=SenderManagerActor.default_uid())
                    yield sender_ref.send_data(session_id, data_key,
                                               self.address, level)
                logger.debug(f'finish fetching data {data_key} from {address}')
                if not isinstance(data_key, tuple):
                    # no need to update meta for shuffle data
                    await meta_api.add_chunk_bands(
                        data_key, [(address, band_name or 'numa-0')])
            except KeyError:
                if error == 'raise':
                    raise DataNotExist(f'Data {session_id, data_key} not exists')

    async def list(self, level: StorageLevel) -> List:
        return await self._data_manager.list(level)

    async def pin(self, session_id, data_key):
        await self._data_manager.pin(session_id, data_key)

    async def unpin(self, session_id, data_key, error):
        await self._data_manager.unpin(session_id, data_key, error)
