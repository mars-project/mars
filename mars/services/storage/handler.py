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
from ...typing import BandType
from ...utils import calc_data_size, lazy_import
from ..cluster import ClusterAPI, StorageInfo
from ..meta import MetaAPI
from .core import StorageQuotaActor, DataManagerActor, DataInfo, \
    build_data_info, WrappedStorageFileObject
from .errors import DataNotExist, NoDataToSpill

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())

logger = logging.getLogger(__name__)


class StorageHandlerActor(mo.StatelessActor):
    """
    Storage handler actor, provide methods like `get`, `put`, etc.
    This actor is stateless and created on worker's sub pools.
    """
    def __init__(self,
                 storage_init_params: Dict,
                 data_manager_ref: Union[DataManagerActor, mo.ActorRef],
                 spill_manager_refs,
                 quota_refs: Dict[StorageLevel,
                                  Union[StorageQuotaActor, mo.ActorRef]],
                 band_name: str = 'numa-0'):
        from .spill import SpillManagerActor

        self._storage_init_params = storage_init_params
        self._data_manager_ref = data_manager_ref
        self._spill_manager_refs: \
            Dict[StorageLevel, Union[SpillManagerActor, mo.ActorRef]] = spill_manager_refs
        self._quota_refs = quota_refs
        self._band_name = band_name
        self._supervisor_address = None

    @classmethod
    def gen_uid(cls, band_name: str):
        return f'storage_handler_{band_name}'

    @property
    def highest_level(self):
        return min(self._quota_refs)

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

    @mo.extensible
    async def get(self,
                  session_id: str,
                  data_key: str,
                  conditions: List = None,
                  error: str = 'raise'):
        try:
            data_info = await self._data_manager_ref.get_data_info(
                session_id, data_key, self._band_name)
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
            session_id, data_key, self._band_name, error)
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

    def _get_default_level(self, obj):
        obj = obj[0] if isinstance(obj, (list, tuple)) else obj
        if self.highest_level != StorageLevel.GPU:
            return self.highest_level
        else:  # pragma: no cover
            if cudf is not None and isinstance(obj, (cudf.DataFrame, cudf.Series, cudf.Index)):
                return StorageLevel.GPU
            elif cupy is not None and isinstance(obj, cupy.ndarray):
                return StorageLevel.GPU
            else:
                return StorageLevel.MEMORY

    @mo.extensible
    async def put(self,
                  session_id: str,
                  data_key: str,
                  obj: object,
                  level: StorageLevel = None) -> DataInfo:
        if level is None:
            level = self._get_default_level(obj)
        size = await asyncio.to_thread(calc_data_size, obj)
        await self.request_quota_with_spill(level, size)
        object_info = await self._clients[level].put(obj)
        data_info = build_data_info(object_info, level, size, self._band_name)
        await self._data_manager_ref.put_data_info(
            session_id, data_key, data_info, object_info)
        if object_info.size is not None and data_info.memory_size != object_info.size:
            await self._quota_refs[level].update_quota(
                object_info.size - data_info.memory_size)
        await self.notify_spillable_space(level)
        return data_info

    @put.batch
    async def batch_put(self, args_list, kwargs_list):
        objs = []
        data_keys = []
        session_id = None
        level = last_level = None
        sizes = []
        for args, kwargs in zip(args_list, kwargs_list):
            session_id, data_key, obj, level = \
                self.put.bind(*args, **kwargs)
            if level is None:
                level = self._get_default_level(obj)
            size = await asyncio.to_thread(calc_data_size, obj)
            if last_level is not None:
                assert last_level == level
            last_level = level
            objs.append(obj)
            data_keys.append(data_key)
            sizes.append(size)

        await self.request_quota_with_spill(level, sum(sizes))

        data_infos = []
        put_infos = []
        quota_delta = 0
        for size, data_key, obj in zip(sizes, data_keys, objs):
            object_info = await self._clients[level].put(obj)
            data_info = build_data_info(object_info, level, size, self._band_name)
            data_infos.append(data_info)
            if object_info.size is not None and \
                    data_info.memory_size != object_info.size:
                # we request memory size before putting, when put finishes,
                # update quota to the true store size
                quota_delta += object_info.size - data_info.memory_size
            put_infos.append(
                self._data_manager_ref.put_data_info.delay(
                    session_id, data_key, data_info, object_info))
        await self._quota_refs[level].update_quota(quota_delta)
        await self._data_manager_ref.put_data_info.batch(*put_infos)
        await self.notify_spillable_space(level)
        return data_infos

    def _get_data_infos_arg(self,
                            session_id: str,
                            data_key: str,
                            error: str = 'raise'):
        infos = self._data_manager_ref.get_data_infos.delay(
            session_id, data_key, self._band_name, error)
        return infos, session_id, data_key

    async def delete_object(self,
                            session_id: str,
                            data_key: Any,
                            data_size: Union[int, float],
                            object_id: Any,
                            level: StorageLevel):
        await self._data_manager_ref.delete_data_info(
            session_id, data_key, level, self._band_name)
        await self._clients[level].delete(object_id)
        await self._quota_refs[level].release_quota(data_size)

    @mo.extensible
    async def delete(self,
                     session_id: str,
                     data_key: str,
                     error: str = 'raise'):
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')

        infos = await self._data_manager_ref.get_data_infos(
            session_id, data_key, self._band_name, error)
        if not infos:
            return

        for info in infos:
            level = info.level
            await self._data_manager_ref.delete_data_info(
                session_id, data_key, level, self._band_name)
            await self._clients[level].delete(info.object_id)
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
                        session_id, data_key, level, info.band))
                to_removes.append((level, info.object_id))
                level_sizes[level] += info.store_size

        if not delete_infos:
            # no data to remove
            return

        await self._data_manager_ref.delete_data_info.batch(*delete_infos)
        await asyncio.gather(*[self._clients[level].delete(object_id)
                               for level, object_id in to_removes])
        for level, size in level_sizes.items():
            await self._quota_refs[level].release_quota(size)

    @mo.extensible
    async def open_reader(self,
                          session_id: str,
                          data_key: str) -> StorageFileObject:
        data_info = await self._data_manager_ref.get_data_info(
            session_id, data_key, self._band_name)
        reader = await self._clients[data_info.level].open_reader(
            data_info.object_id)
        return reader

    @open_reader.batch
    async def batch_open_readers(self, args_list, kwargs_list):
        get_data_infos = []
        for args, kwargs in zip(args_list, kwargs_list):
            get_data_infos.append(
                self._data_manager_ref.get_data_info.delay(
                    *args, band_name=self._band_name, **kwargs))
        data_infos = await self._data_manager_ref.get_data_info.batch(*get_data_infos)
        return await asyncio.gather(
            *[self._clients[data_info.level].open_reader(data_info.object_id)
            for data_info in data_infos])

    @mo.extensible
    async def open_writer(self,
                          session_id: str,
                          data_key: str,
                          size: int,
                          level: StorageLevel,
                          request_quota=True) -> WrappedStorageFileObject:
        if request_quota:
            await self.request_quota_with_spill(level, size)
        writer = await self._clients[level].open_writer(size)
        return WrappedStorageFileObject(writer, level, size, session_id, data_key,
                                        self._data_manager_ref, self._clients[level])

    @open_writer.batch
    async def batch_open_writers(self, args_list, kwargs_list):
        extracted_args = None
        data_keys, sizes = [], []
        for args, kwargs in zip(args_list, kwargs_list):
            session_id, data_key, size, level, request_quota = \
                self.open_writer.bind(*args, **kwargs)
            if extracted_args:
                assert extracted_args == (session_id, level, request_quota)
            extracted_args = (session_id, level, request_quota)
            data_keys.append(data_key)
            sizes.append(size)
        session_id, level, request_quota = extracted_args
        if request_quota:  # pragma: no cover
            await self.request_quota_with_spill(level, sum(sizes))
        writers = await asyncio.gather(*[self._clients[level].open_writer(size)
                                         for size in sizes])
        wrapped_writers = []
        for writer, size, data_key in zip(writers, sizes, data_keys):
            wrapped_writers.append(
                WrappedStorageFileObject(writer, level, size, session_id, data_key,
                                         self._data_manager_ref, self._clients[level]))
        return wrapped_writers

    async def _get_meta_api(self, session_id: str):
        if self._supervisor_address is None:
            cluster_api = await ClusterAPI.create(self.address)
            [self._supervisor_address] = await cluster_api.get_supervisors_by_keys([session_id])

        return await MetaAPI.create(session_id=session_id,
                                    address=self._supervisor_address)

    async def _fetch_remote(self,
                            session_id: str,
                            data_keys: List[Union[str, tuple]],
                            remote_band: BandType,
                            error: str):
        remote_manager_ref: Union[mo.ActorRef, DataManagerActor] = await mo.actor_ref(
            uid=DataManagerActor.default_uid(), address=remote_band[0])
        get_data_infos = []
        for data_key in data_keys:
            get_data_infos.append(
                remote_manager_ref.get_data_info.delay(session_id, data_key, error))
        data_infos = await remote_manager_ref.get_data_info.batch(*get_data_infos)
        data_infos, data_keys = zip(*[(data_info, data_key) for data_info, data_key in
                                    zip(data_infos, data_keys) if data_info is not None])
        put_data_info_delays = []
        fetch_tasks = []
        for data_info, data_key in zip(data_infos, data_keys):
            put_data_info_delays.append(
                self._data_manager_ref.put_data_info.delay(
                    session_id, data_key, data_info, None))
            fetch_tasks.append(self._clients[StorageLevel.REMOTE].fetch(data_info.object_id))
        await self._data_manager_ref.put_data_info.batch(*put_data_info_delays)
        await asyncio.gather(*fetch_tasks)

    async def _fetch_via_transfer(self,
                                  session_id: str,
                                  data_keys: List[Union[str, tuple]],
                                  level: StorageLevel,
                                  remote_band: BandType,
                                  fetch_band_name: str,
                                  error: str):
        from .transfer import SenderManagerActor

        logger.debug('Begin to fetch %s from band %s', data_keys, remote_band)
        sender_ref: Union[mo.ActorRef, SenderManagerActor] = await mo.actor_ref(
            address=remote_band[0], uid=SenderManagerActor.gen_uid(remote_band[1]))
        await sender_ref.send_batch_data(
            session_id, data_keys, self._data_manager_ref.address,
            level, fetch_band_name, error=error)
        logger.debug('Finish fetching %s from band %s', data_keys, remote_band)

    async def fetch_batch(self,
                          session_id: str,
                          data_keys: List[str],
                          level: StorageLevel,
                          band_name: str,
                          address: str,
                          error: str):
        if error not in ('raise', 'ignore'):  # pragma: no cover
            raise ValueError('error must be raise or ignore')

        meta_api = await self._get_meta_api(session_id)
        remote_keys = defaultdict(set)
        missing_keys = []
        get_metas = []
        get_info_delays = []
        for data_key in data_keys:
            get_info_delays.append(
                self._data_manager_ref.get_data_info.delay(
                    session_id, data_key, band_name, error='ignore'))
        data_infos = await self._data_manager_ref.get_data_info.batch(*get_info_delays)
        pin_delays = []
        for data_key, info in zip(data_keys, data_infos):
            # for gpu bands, need transfer between gpu cards
            if info is not None:
                if band_name and band_name != info.band:
                    missing_keys.append(data_key)
                else:
                    pin_delays.append(self._data_manager_ref.pin.delay(
                        session_id, data_key, self._band_name))
            else:
                # Not exists in local, fetch from remote worker
                missing_keys.append(data_key)
        if address is None or band_name is None:
            # some mapper keys are absent, specify error='ignore'
            get_metas = [(meta_api.get_chunk_meta.delay(
                data_key, fields=['bands'], error='ignore')) for data_key in missing_keys]
        await self._data_manager_ref.pin.batch(*pin_delays)

        if get_metas:
            metas = await meta_api.get_chunk_meta.batch(*get_metas)
        else:  # pragma: no cover
            metas = [(address, band_name)] * len(missing_keys)
        for data_key, bands in zip(missing_keys, metas):
            if bands is not None:
                remote_keys[bands['bands'][0]].add(data_key)

        transfer_tasks = []
        fetch_keys = []
        for band, keys in remote_keys.items():
            if StorageLevel.REMOTE in self._quota_refs:
                # if storage support remote level, just fetch object id
                transfer_tasks.append(self._fetch_remote(
                    session_id, list(keys), band, error))
            else:
                # fetch via transfer
                transfer_tasks.append(self._fetch_via_transfer(
                    session_id, list(keys), level, band,
                    band_name or band[1], error))
            fetch_keys.extend(list(keys))

        await asyncio.gather(*transfer_tasks)

        append_bands_delays = []
        for data_key in fetch_keys:
            append_bands_delays.append(meta_api.add_chunk_bands.delay(
                    data_key, [(self.address, self._band_name)]))
        await meta_api.add_chunk_bands.batch(*append_bands_delays)

    async def request_quota_with_spill(self,
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
        if await self._spill_manager_refs[level].has_spill_task():
            total, used = await self._quota_refs[level].get_quota()
            tasks = []
            if total is not None:
                spillable_size = await self._data_manager_ref.get_spillable_size(
                    level, self._band_name)
                tasks.append(self._spill_manager_refs[level].notify_spillable_space(
                    spillable_size, total - used))
            await asyncio.gather(*tasks)

    async def spill(self,
                    level: StorageLevel,
                    request_size: int,
                    object_size: int):
        from .spill import spill

        try:
            await spill(request_size, level, self._band_name, self._data_manager_ref, self)
        except NoDataToSpill:
            logger.warning('No data to spill %s bytes, waiting more space', request_size)
            size = await self._spill_manager_refs[level].wait_for_space(object_size)
            await spill(size, level, self._band_name, self._data_manager_ref, self)

    async def list(self, level: StorageLevel) -> List:
        return await self._data_manager_ref.list(level, self._band_name)

    @mo.extensible
    async def unpin(self, session_id: str, data_key: str, error: str = 'raise'):
        levels = await self._data_manager_ref.unpin(
            session_id, [data_key], self._band_name, error)
        if levels:
            await self.notify_spillable_space(levels[0])

    @unpin.batch
    async def batch_unpin(self, args_list, kwargs_list):
        extracted_args = []
        data_keys = []
        for args, kw in zip(args_list, kwargs_list):
            session_id, data_key, error = self.unpin.bind(*args, **kw)
            if extracted_args:
                assert extracted_args == (session_id, error)
            extracted_args = session_id, error
            data_keys.append(data_key)
        if extracted_args:
            session_id, error = extracted_args
            levels = await self._data_manager_ref.unpin(
                    session_id, data_keys, self._band_name, error)
            for level in levels:
                await self.notify_spillable_space(level)

    async def get_storage_level_info(self, level: StorageLevel) -> StorageInfo:
        quota_ref = self._quota_refs[level]
        total_size, used_size = await quota_ref.get_quota()
        return StorageInfo(storage_level=level,
                           total_size=int(total_size) if total_size else total_size,
                           used_size=int(used_size))
