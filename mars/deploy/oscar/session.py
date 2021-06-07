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
import itertools
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from urllib.parse import urlparse
from weakref import WeakKeyDictionary
from typing import Any, Dict, List, Tuple, Union

from ...core import TileableType, ChunkType, enter_mode
from ...core.operand import Fetch
from ...core.session import AbstractAsyncSession, register_session_cls, \
    ExecutionInfo as AbstractExecutionInfo, gen_submit_tileable_graph
from ...lib.aio import alru_cache
from ...services.cluster import ClusterAPI
from ...services.lifecycle import AbstractLifecycleAPI, LifecycleAPI
from ...services.meta import MetaAPI, AbstractMetaAPI
from ...services.session import AbstractSessionAPI, SessionAPI
from ...services.storage import StorageAPI
from ...services.task import AbstractTaskAPI, TaskAPI, TaskResult
from ...tensor.utils import slice_split
from ...utils import implements, merge_chunks, sort_dataframe_result, \
    register_asyncio_task_timeout_detector
from .typing import ClientType


@dataclass
class Progress:
    value: float = 0.0


class ExecutionInfo(AbstractExecutionInfo):
    def __init__(self,
                 task_id: str,
                 task_api: AbstractTaskAPI,
                 aio_task: asyncio.Task,
                 progress: Progress):
        super().__init__(aio_task)
        self._task_api = task_api
        self._task_id = task_id
        self._progress = progress

    def progress(self) -> float:
        return self._progress.value


@dataclass
class ChunkFetchInfo:
    tileable: TileableType
    chunk: ChunkType
    indexes: List[Union[int, slice]]
    data: Any = None


@register_session_cls
class Session(AbstractAsyncSession):
    name = 'oscar'

    def __init__(self,
                 address: str,
                 session_id: str,
                 session_api: AbstractSessionAPI,
                 meta_api: AbstractMetaAPI,
                 lifecycle_api: AbstractLifecycleAPI,
                 task_api: AbstractTaskAPI,
                 cluster_api: ClusterAPI,
                 client: ClientType = None):
        super().__init__(address, session_id)
        self._session_api = session_api
        self._task_api = task_api
        self._meta_api = meta_api
        self._lifecycle_api = lifecycle_api
        self._cluster_api = cluster_api
        self.client = client

        self._tileable_to_fetch = WeakKeyDictionary()
        self._asyncio_task_timeout_detector_task = register_asyncio_task_timeout_detector()

    @classmethod
    async def _init(cls,
                    address: str,
                    session_id: str,
                    new: bool = True):
        session_api = await SessionAPI.create(address)
        if new:
            # create new session
            session_address = await session_api.create_session(session_id)
        else:
            session_address = await session_api.get_session_address(session_id)
        lifecycle_api = await LifecycleAPI.create(session_id, session_address)
        meta_api = await MetaAPI.create(session_id, session_address)
        task_api = await TaskAPI.create(session_id, session_address)
        cluster_api = await ClusterAPI.create(session_address)
        return cls(address, session_id,
                   session_api, meta_api,
                   lifecycle_api, task_api,
                   cluster_api)

    @classmethod
    @implements(AbstractAsyncSession.init)
    async def init(cls,
                   address: str,
                   session_id: str,
                   new: bool = True,
                   **kwargs) -> "Session":
        init_local = kwargs.pop('init_local', False)
        if init_local:
            from .local import new_cluster
            return (await new_cluster(address, **kwargs)).session

        if kwargs:  # pragma: no cover
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError(f'Oscar session got unexpected '
                            f'arguments: {unexpected_keys}')

        if urlparse(address).scheme == 'http':
            return await WebSession._init(address, session_id, new=new)
        else:
            return await cls._init(address, session_id, new=new)

    async def _run_in_background(self,
                                 tileables: list,
                                 task_id: str,
                                 progress: Progress):
        with enter_mode(build=True, kernel=True):
            # wait for task to finish
            while True:
                task_result: TaskResult = await self._task_api.wait_task(
                    task_id, timeout=0.5)
                if task_result is None:
                    # not finished, set progress
                    progress.value = await self._task_api.get_task_progress(task_id)
                else:
                    progress.value = 1.0
                    break
            if task_result.error:
                raise task_result.error.with_traceback(task_result.traceback)
            fetch_tileables = await self._task_api.get_fetch_tileables(task_id)
            assert len(tileables) == len(fetch_tileables)

            for tileable, fetch_tileable in zip(tileables, fetch_tileables):
                self._tileable_to_fetch[tileable] = fetch_tileable
                # update meta, e.g. unknown shape
                tileable.params = fetch_tileable.params

    async def execute(self,
                      *tileables,
                      **kwargs) -> ExecutionInfo:
        fuse_enabled: bool = kwargs.pop('fuse_enabled', True)
        task_name: str = kwargs.pop('task_name', None)
        extra_config: dict = kwargs.pop('extra_config', None)
        if kwargs:  # pragma: no cover
            raise TypeError(f'run got unexpected key arguments {list(kwargs)!r}')

        tileables = [tileable.data if hasattr(tileable, 'data') else tileable
                     for tileable in tileables]

        # build tileable graph
        tileable_graph = gen_submit_tileable_graph(self, tileables)

        # submit task
        task_id = await self._task_api.submit_tileable_graph(
            tileable_graph, task_name=task_name, fuse_enabled=fuse_enabled,
            extra_config=extra_config)

        progress = Progress()
        # create asyncio.Task
        future = asyncio.create_task(
            self._run_in_background(tileables, task_id, progress))
        return ExecutionInfo(task_id, self._task_api, future, progress)

    def _get_to_fetch_tileable(self, tileable: TileableType) -> \
            Tuple[TileableType, List[Union[slice, Integral]]]:
        from ...tensor.indexing import TensorIndex
        from ...dataframe.indexing.iloc import \
            DataFrameIlocGetItem, SeriesIlocGetItem

        slice_op_types = \
            TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem

        if hasattr(tileable, 'data'):
            tileable = tileable.data

        indexes = None
        while tileable not in self._tileable_to_fetch:
            # if tileable's op is slice, try to check input
            if isinstance(tileable.op, slice_op_types):
                indexes = tileable.op.indexes
                tileable = tileable.inputs[0]
                if not all(isinstance(index, (slice, Integral))
                           for index in indexes):
                    raise ValueError('Only support fetch data slices')
            elif isinstance(tileable.op, Fetch):
                break
            else:
                raise ValueError(f'Cannot fetch unexecuted '
                                 f'tileable: {tileable}')

        if isinstance(tileable.op, Fetch):
            return tileable, indexes
        else:
            return self._tileable_to_fetch[tileable], indexes

    @classmethod
    def _calc_chunk_indexes(cls,
                            fetch_tileable: TileableType,
                            indexes: List[Union[slice, Integral]]) -> \
            Dict[ChunkType, List[Union[slice, int]]]:
        axis_to_slices = {
            axis: slice_split(ind, fetch_tileable.nsplits[axis])
            for axis, ind in enumerate(indexes)}
        result = dict()
        for chunk_index in itertools.product(
                *[v.keys() for v in axis_to_slices.values()]):
            # slice_obj: use tuple, since numpy complains
            #
            # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use
            # `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array
            # index, `arr[np.array(seq)]`, which will result either in an error or a different result.
            slice_obj = [axis_to_slices[axis][chunk_idx]
                         for axis, chunk_idx in enumerate(chunk_index)]
            chunk = fetch_tileable.cix[chunk_index]
            result[chunk] = slice_obj
        return result

    def _process_result(self, tileable, result):  # pylint: disable=no-self-use
        return sort_dataframe_result(tileable, result)

    @alru_cache(cache_exceptions=False)
    async def _get_storage_api(self, address: str):
        if urlparse(self.address).scheme == 'http':
            from mars.services.storage.api import WebStorageAPI
            storage_api = WebStorageAPI(self._session_id, self.address)
        else:
            storage_api = await StorageAPI.create(self._session_id, address)
        return storage_api

    async def fetch(self, *tileables, **kwargs):
        from ...tensor.core import TensorOrder
        from ...tensor.array_utils import get_array_module

        if kwargs:  # pragma: no cover
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError(f'`fetch` got unexpected '
                            f'arguments: {unexpected_keys}')

        with enter_mode(build=True):
            chunks = []
            get_chunk_metas = []
            fetch_infos_list = []
            for tileable in tileables:
                fetch_tileable, indexes = self._get_to_fetch_tileable(tileable)
                chunk_to_slice = None
                if indexes is not None:
                    chunk_to_slice = self._calc_chunk_indexes(
                        fetch_tileable, indexes)
                fetch_infos = []
                for chunk in fetch_tileable.chunks:
                    if indexes and chunk not in chunk_to_slice:
                        continue
                    chunks.append(chunk)
                    get_chunk_metas.append(
                        self._meta_api.get_chunk_meta.delay(
                            chunk.key, fields=['bands']))
                    indexes = chunk_to_slice[chunk] \
                        if chunk_to_slice is not None else None
                    fetch_infos.append(ChunkFetchInfo(tileable=tileable,
                                                      chunk=chunk,
                                                      indexes=indexes))
                fetch_infos_list.append(fetch_infos)
            chunk_metas = \
                await self._meta_api.get_chunk_meta.batch(*get_chunk_metas)
            chunk_to_addr = {chunk: meta['bands'][0][0]
                             for chunk, meta in zip(chunks, chunk_metas)}

            storage_api_to_gets = defaultdict(list)
            storage_api_to_fetch_infos = defaultdict(list)
            for fetch_info in itertools.chain(*fetch_infos_list):
                conditions = fetch_info.indexes
                chunk = fetch_info.chunk
                addr = chunk_to_addr[chunk]
                storage_api = await self._get_storage_api(addr)
                storage_api_to_gets[storage_api].append(
                    storage_api.get.delay(chunk.key, conditions=conditions))
                storage_api_to_fetch_infos[storage_api].append(fetch_info)
            for storage_api in storage_api_to_gets:
                fetched_data = await storage_api.get.batch(
                    *storage_api_to_gets[storage_api])
                infos = storage_api_to_fetch_infos[storage_api]
                for info, data in zip(infos, fetched_data):
                    info.data = data

            result = []
            for tileable, fetch_infos in zip(tileables, fetch_infos_list):
                index_to_data = [(fetch_info.chunk.index, fetch_info.data)
                                 for fetch_info in fetch_infos]
                merged = merge_chunks(index_to_data)
                if hasattr(tileable, 'order') and tileable.ndim > 0:
                    module = get_array_module(merged)
                    if tileable.order == TensorOrder.F_ORDER and \
                            hasattr(module, 'asfortranarray'):
                        merged = module.asfortranarray(merged)
                    elif tileable.order == TensorOrder.C_ORDER and \
                            hasattr(module, 'ascontiguousarray'):
                        merged = module.ascontiguousarray(merged)
                if hasattr(tileable, 'isscalar') and tileable.isscalar() and \
                        getattr(merged, 'size', None) == 1:
                    merged = merged.item()
                result.append(self._process_result(tileable, merged))
            return result

    async def decref(self, *tileable_keys):
        return await self._lifecycle_api.decref_tileables(list(tileable_keys))

    async def _get_ref_counts(self) -> Dict[str, int]:
        return await self._lifecycle_api.get_all_chunk_ref_counts()

    async def fetch_tileable_op_logs(self,
                                     tileable_op_key: str,
                                     offsets: Union[Dict[str, List[int]], str, int],
                                     sizes: Union[Dict[str, List[int]], str, int]) -> Dict:
        return await self._session_api.fetch_tileable_op_logs(
            self.session_id, tileable_op_key, offsets, sizes)

    async def get_total_n_cpu(self):
        all_bands = await self._cluster_api.get_all_bands()
        n_cpu = 0
        for band, size in all_bands.items():
            _, band_name = band
            if band_name.startswith('numa-'):
                n_cpu += size
        return n_cpu

    async def destroy(self):
        await super().destroy()
        await self._session_api.delete_session(self._session_id)
        if self._asyncio_task_timeout_detector_task:  # pragma: no cover
            self._asyncio_task_timeout_detector_task.cancel()

    async def stop_server(self):
        if self.client:
            await self.client.stop()


class WebSession(Session):
    @classmethod
    async def _init(cls,
                    address: str,
                    session_id: str,
                    new=True):
        from ...services.session import WebSessionAPI
        from ...services.lifecycle import WebLifecycleAPI
        from ...services.meta import WebMetaAPI
        from ...services.task import WebTaskAPI
        from ...services.cluster import ClusterAPI

        session_api = WebSessionAPI(address)
        if new:
            # create new session
            await session_api.create_session(session_id)
        else:
            await session_api.get_session_address(session_id)
        lifecycle_api = WebLifecycleAPI(session_id, address)
        meta_api = WebMetaAPI(session_id, address)
        task_api = WebTaskAPI(session_id, address)
        cluster_api = ClusterAPI(address)

        return cls(address, session_id,
                   session_api, meta_api,
                   lifecycle_api, task_api,
                   cluster_api)
