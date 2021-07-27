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

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Union

from ... import oscar as mo
from ...storage import StorageLevel, get_storage_backend
from ...storage.core import StorageFileObject
from ...utils import calc_data_size, extensible
from ..cluster import ClusterAPI, StorageInfo
from ..meta import MetaAPI
from .core import StorageQuotaActor, DataManagerActor, DataInfo, \
    build_data_info, WrappedStorageFileObject
from .errors import DataNotExist, NoDataToSpill

logger = logging.getLogger(__name__)


class StorageHandlerActor(mo.Actor):
    """
    Storage handler actor, provide methods like `get`, `put`, etc.
    This actor is stateless and created on worker's sub pools.
    """
    def __init__(self,
                 storage_init_params: Dict,
                 data_manager_ref: Union[DataManagerActor, mo.ActorRef],
                 spill_manager_refs,
                 quota_refs: Dict[StorageLevel,
                                  Union[StorageQuotaActor, mo.ActorRef]]):
        from .spill import SpillManagerActor

        self._storage_init_params = storage_init_params
        self._data_manager_ref = data_manager_ref
        self._spill_manager_refs: \
            Dict[StorageLevel, Union[SpillManagerActor, mo.ActorRef]] = spill_manager_refs
        self._quota_refs = quota_refs
        self._supervisor_address = None

    async def __post_create__(self):
        self._clients = clients = dict()
        for backend, init_params in self._storage_init_params.items():
            logger.debug('Start storage %s with params %s', backend, init_params)
            storage_cls = get_storage_backend(backend)
            client = storage_cls(**init_params)
            for level in StorageLevel.__members__.values():
                if client.level & level:
                    clients[level] = client

    async def _get_data(self, data_info, conditions):
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
        data_info = build_data_info(object_info, level, size)
        await self._data_manager_ref.put_data_info(
            session_id, data_key, data_info, object_info)
        if object_info.size is not None and data_info.memory_size != object_info.size:
            await self._quota_refs[level].update_quota(
                object_info.size - data_info.memory_size)
        await self.notify_spillable_space(level)
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
        quota_delta = 0
        for size, data_key, obj in zip(sizes, data_keys, objs):
            object_info = await self._clients[level].put(obj)
            data_info = build_data_info(object_info, level, size)
            data_infos.append(data_info)
            if object_info.size is not None and \
                    data_info.memory_size != object_info.size:
                # we request memory size before putting, when put finishes,
                # update quota to the true store size
                quota_delta += object_info.size - data_info.memory_size
            put_infos.append(
                self._data_manager_ref.put_data_info.delay(
                    session_id, data_key, data_info, object_info))
            logger.debug('Finish putting data key %s, size is %s, '
                         'object_id is %s', data_key, size, data_info.object_id)
        await self._quota_refs[level].update_quota(quota_delta)
        await self._data_manager_ref.put_data_info.batch(*put_infos)
        await self.notify_spillable_space(level)
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
                            data_size: Union[int, float],
                            object_id: Any,
                            level: StorageLevel):
        await self._data_manager_ref.delete_data_info(
            session_id, data_key, level)
        await self._clients[level].delete(object_id)
        await self._quota_refs[level].release_quota(data_size)

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
        logger.debug('Begin to delete batch data %s', to_removes)
        for level, object_id in to_removes:
            yield self._clients[level].delete(object_id)
        logger.debug('Finish deleting batch data %s', to_removes)
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

    async def _get_meta_api(self, session_id: str):
        if self._supervisor_address is None:
            cluster_api = await ClusterAPI.create(self.address)
            [self._supervisor_address] = await cluster_api.get_supervisors_by_keys([session_id])

        return await MetaAPI.create(session_id=session_id,
                                    address=self._supervisor_address)

    async def _fetch_remote(self,
                            session_id: str,
                            data_key: Union[str, tuple],
                            level: StorageLevel,
                            remote_address: str):
        remote_manager_ref = await mo.actor_ref(uid=DataManagerActor.default_uid(),
                                                address=remote_address)
        data_info = await remote_manager_ref.get_data_info(session_id, data_key)
        await self._data_manager_ref.put_data_info(
            session_id, data_key, data_info, None)
        try:
            await self._clients[level].fetch(data_info.object_id)
        except asyncio.CancelledError:  # pragma: no cover
            await self._data_manager_ref.delete_data_info(
                session_id, data_key, data_info.level)
            raise

    async def _fetch_via_transfer(self,
                                  session_id: str,
                                  data_key: Union[str, tuple],
                                  level: StorageLevel,
                                  remote_address: str):
        from .transfer import SenderManagerActor

        sender_ref = await mo.actor_ref(
            address=remote_address, uid=SenderManagerActor.default_uid())
        await sender_ref.send_data(
            session_id, data_key, self._data_manager_ref.address, level)

    async def fetch(self,
                    session_id: str,
                    data_key: str,
                    level: StorageLevel,
                    address: str,
                    band_name: str,
                    error: str):

        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')

        try:
            await self._data_manager_ref.get_data_info(session_id, data_key)
            await self._data_manager_ref.pin(session_id, data_key)
        except DataNotExist:
            # Not exists in local, fetch from remote worker
            try:
                meta_api = await self._get_meta_api(session_id)
                if address is None:
                    # we get meta using main key when fetch shuffle data
                    main_key = data_key[0] if isinstance(data_key, tuple) else data_key
                    address = (await meta_api.get_chunk_meta(
                        main_key, fields=['bands']))['bands'][0][0]
                logger.debug('Begin to fetch data %s from %s', data_key, address)
                if StorageLevel.REMOTE in self._quota_refs:
                    yield self._fetch_remote(session_id, data_key, level, address)
                else:
                    await self._fetch_via_transfer(session_id, data_key, level, address)
                logger.debug('finish fetching data %s from %s', data_key, address)
                if not isinstance(data_key, tuple):
                    # no need to update meta for shuffle data
                    await meta_api.add_chunk_bands(
                        data_key, [(address, band_name or 'numa-0')])
            except DataNotExist:
                if error == 'raise':  # pragma: no cover
                    raise

    async def _request_quota_with_spill(self,
                                        level: StorageLevel,
                                        size: int):
        if await self._quota_refs[level].request_quota(size):
            return
        else:
            total, used = await self._quota_refs[level].get_quota()
            await self.spill(level, int(used + size - total), size)
            await self._quota_refs[level].request_quota(size)
            logger.debug('Spill is triggered, request %s bytes of %s finished', size, level)

    async def notify_spillable_space(self, level):
        total, used = await self._quota_refs[level].get_quota()
        if total is not None:
            spillable_size = await self._data_manager_ref.get_spillable_size(level)
            await self._spill_manager_refs[level].notify_spillable_space(
                spillable_size, total - used)

    async def spill(self,
                    level: StorageLevel,
                    request_size: int,
                    object_size: int):
        from .spill import spill

        try:
            await spill(request_size, level, self._data_manager_ref, self)
        except NoDataToSpill:
            logger.warning('No data to spill %s bytes, waiting more space', request_size)
            size = await self._spill_manager_refs[level].wait_for_space(object_size)
            await spill(size, level, self._data_manager_ref, self)

    async def list(self, level: StorageLevel) -> List:
        return await self._data_manager_ref.list(level)

    async def unpin(self, session_id: str, data_key: str, error: str):
        level = await self._data_manager_ref.unpin(session_id, data_key, error)
        if level is not None:
            await self.notify_spillable_space(level)

    async def get_storage_level_info(self, level: StorageLevel) -> StorageInfo:
        quota_ref = self._quota_refs[level]
        total_size, used_size = await quota_ref.get_quota()
        return StorageInfo(storage_level=level,
                           total_size=int(total_size) if total_size else total_size,
                           used_size=int(used_size))
