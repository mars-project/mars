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
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Union

from .. import oscar as mo
from ..core import TileableType
from ..core.context import Context
from ..core.session import AbstractAsyncSession, AbstractSyncSession, \
    ExecutionInfo, _get_session, _execute, _fetch
from ..utils import implements
from .cluster import ClusterAPI, NodeRole
from .session import SessionAPI
from .storage import StorageAPI
from .meta import MetaAPI


class ThreadedServiceContext(Context):
    _cluster_api: ClusterAPI
    _session_api: SessionAPI
    _meta_api: MetaAPI

    def __init__(self,
                 session_id: str,
                 supervisor_address: str,
                 current_address: str,
                 loop: asyncio.AbstractEventLoop):
        super().__init__(session_id=session_id,
                         supervisor_address=supervisor_address,
                         current_address=current_address)
        self._loop = loop

        # APIs
        self._cluster_api = None
        self._session_api = None
        self._meta_api = None

    async def init(self):
        self._cluster_api = await ClusterAPI.create(
            self.supervisor_address)
        self._session_api = await SessionAPI.create(
            self.supervisor_address)
        self._meta_api = await MetaAPI.create(
            self.session_id, self.supervisor_address)

    def _call(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(Context.get_current_session)
    def get_current_session(self) -> AbstractSyncSession:
        sess = self._call(_get_session(self.supervisor_address,
                                       self.session_id))
        return ThreadedServiceSession(sess, self._loop)

    @implements(Context.get_supervisor_addresses)
    def get_supervisor_addresses(self) -> List[str]:
        return self._call(self._cluster_api.get_supervisors())

    @implements(Context.get_worker_addresses)
    def get_worker_addresses(self) -> List[str]:
        return list(self._call(
            self._cluster_api.get_nodes_info(role=NodeRole.WORKER)))

    @implements(Context.get_total_n_cpu)
    def get_total_n_cpu(self) -> int:
        all_bands = self._call(
            self._cluster_api.get_all_bands())
        n_cpu = 0
        for band, size in all_bands.items():
            _, band_name = band
            if band_name.startswith('numa-'):
                n_cpu += size
        return n_cpu

    async def _get_chunks_meta(self,
                               data_keys: List[str],
                               fields: List[str] = None,
                               error: str = 'raise') -> List[Dict]:
        # get chunks meta
        get_metas = []
        for data_key in data_keys:
            meta = self._meta_api.get_chunk_meta.delay(
                data_key, fields=fields, error=error)
            get_metas.append(meta)
        metas = await self._meta_api.get_chunk_meta.batch(*get_metas)
        return metas

    async def _get_chunks_result(self, data_keys: List[str]) -> List:
        metas = await self._get_chunks_meta(data_keys, fields=['bands'])
        addresses = [meta['bands'][0][0] for meta in metas]

        storage_api_to_gets = defaultdict(lambda: (list(), list()))
        for data_key, address in zip(data_keys, addresses):
            storage_api = await StorageAPI.create(self.session_id, address)
            storage_api_to_gets[storage_api][0].append(data_key)
            storage_api_to_gets[storage_api][1].append(
                storage_api.get.delay(data_key))
        results = dict()
        for storage_api, (keys, gets) in storage_api_to_gets.items():
            chunks_data = await storage_api.get.batch(*gets)
            for chunk_key, chunk_data in zip(keys, chunks_data):
                results[chunk_key] = chunk_data
        return [results[key] for key in data_keys]

    @implements(Context.get_chunks_result)
    def get_chunks_result(self,
                          data_keys: List[str]) -> List:
        return self._call(self._get_chunks_result(data_keys))

    @implements(Context.get_chunks_meta)
    def get_chunks_meta(self,
                        data_keys: List[str],
                        fields: List[str] = None,
                        error='raise') -> List[Dict]:
        return self._call(self._get_chunks_meta(data_keys, fields=fields, error=error))

    @implements(Context.create_remote_object)
    def create_remote_object(self,
                             name: str,
                             object_cls, *args, **kwargs):
        ref = self._call(self._session_api.create_remote_object(
            self.session_id, name, object_cls, *args, **kwargs))
        return _RemoteObjectWrapper(ref, self._loop)

    @implements(Context.get_remote_object)
    def get_remote_object(self, name: str):
        ref = self._call(self._session_api.get_remote_object(
            self.session_id, name))
        return _RemoteObjectWrapper(ref, self._loop)

    @implements(Context.destroy_remote_object)
    def destroy_remote_object(self,
                              name: str):
        return self._call(self._session_api.destroy_remote_object(
            self.session_id, name))

    @implements(Context.register_custom_log_path)
    def register_custom_log_path(self,
                                 session_id: str,
                                 tileable_op_key: str,
                                 chunk_op_key: str,
                                 worker_address: str,
                                 log_path: str):
        return self._call(self._session_api.register_custom_log_path(
            session_id, tileable_op_key, chunk_op_key, worker_address, log_path))

    @implements(Context.new_custom_log_dir)
    @lru_cache(50)
    def new_custom_log_dir(self) -> str:
        return self._call(self._session_api.new_custom_log_dir(
            self.current_address, self.session_id))


class _RemoteObjectWrapper:
    def __init__(self,
                 ref: mo.ActorRef,
                 loop: asyncio.AbstractEventLoop):
        self._ref = ref
        self._loop = loop

    def __getattr__(self, attr):
        func = getattr(self._ref, attr)

        def wrap(*args, **kwargs):
            coro = func(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            return fut.result()

        return wrap


class ThreadedServiceSession(AbstractSyncSession):
    def __init__(self,
                 session: AbstractAsyncSession,
                 loop: asyncio.AbstractEventLoop):
        super().__init__(session.address, session.session_id)
        self._session = session
        self._loop = loop

    @implements(AbstractSyncSession.execute)
    def execute(self,
                tileable,
                *tileables,
                **kwargs) -> Union[List[TileableType], TileableType, ExecutionInfo]:
        wait = kwargs.get('wait', True)
        if kwargs.get('show_progress', 'auto') == 'auto':
            # disable progress show in context session
            kwargs['show_progress'] = False
        coro = _execute(tileable, *tileables, session=self, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        execution_info = fut.result()
        if wait:
            return tileable if len(tileables) == 0 else \
                [tileable] + list(tileables)
        else:
            return execution_info

    @implements(AbstractSyncSession.fetch)
    def fetch(self, *tileables, **kwargs) -> list:
        coro = _fetch(*tileables, session=self._session, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.decref)
    def decref(self, *tileables_keys):
        coro = self._session.decref(*tileables_keys)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.fetch_tileable_op_logs)
    def fetch_tileable_op_logs(self,
                               tileable_op_key: str,
                               offsets: Dict[str, List[int]],
                               sizes: Dict[str, List[int]]) -> Dict:
        coro = self._session.fetch_tileable_op_logs(
            tileable_op_key, offsets, sizes)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.get_total_n_cpu)
    def get_total_n_cpu(self):
        coro = self._session.get_total_n_cpu()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.to_async)
    def to_async(self):
        return self._session
