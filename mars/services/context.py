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
from typing import List, Dict

from ..core.context import Context
from ..core.session import AbstractSession, AbstractSyncSession, \
    ExecutionInfo, _get_session
from ..utils import implements
from .cluster import ClusterAPI, NodeRole
from .session import SessionAPI
from .storage import StorageAPI
from .meta import MetaAPI


class ThreadedServiceContext(Context):
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
            addr, band_name = band
            if band_name.startswith('numa-'):
                n_cpu += size
        return n_cpu

    async def _get_chunks_meta(self,
                               data_keys: List[str],
                               fields: List[str] = None) -> List[Dict]:
        # get chunks meta
        get_metas = []
        for data_key in data_keys:
            meta = self._meta_api.get_chunk_meta.delay(
                data_key, fields=fields)
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
                        fields: List[str] = None) -> List[Dict]:
        return self._call(self._get_chunks_meta(data_keys, fields=fields))


class ThreadedServiceSession(AbstractSyncSession):
    def __init__(self,
                 session: AbstractSession,
                 loop: asyncio.AbstractEventLoop):
        self._session = session
        self._loop = loop

    @implements(AbstractSyncSession.execute)
    def execute(self,
                *tileables,
                **kwargs) -> ExecutionInfo:
        coro = self._session.execute(*tileables, **kwargs)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.fetch)
    def fetch(self, *tileables) -> list:
        coro = self._session.fetch(*tileables)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(AbstractSyncSession.decref)
    def decref(self, *tileables_keys):
        coro = self._session.decref(*tileables_keys)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()
