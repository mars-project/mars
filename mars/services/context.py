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
from functools import lru_cache
from typing import List, Dict

from .. import oscar as mo
from ..lib.aio import new_isolation
from ..core.context import Context
from ..storage.base import StorageLevel
from ..typing import BandType, SessionType
from ..utils import implements, is_ray_address
from .cluster import ClusterAPI, NodeRole
from .session import SessionAPI
from .storage import StorageAPI
from .subtask import SubtaskAPI
from .meta import MetaAPI, WorkerMetaAPI

logger = logging.getLogger(__name__)


class ThreadedServiceContext(Context):
    _cluster_api: ClusterAPI
    _session_api: SessionAPI
    _meta_api: MetaAPI
    _subtask_api: SubtaskAPI

    def __init__(
        self,
        session_id: str,
        supervisor_address: str,
        worker_address: str,
        local_address: str,
        loop: asyncio.AbstractEventLoop,
        band: BandType = None,
    ):
        super().__init__(
            session_id=session_id,
            supervisor_address=supervisor_address,
            worker_address=worker_address,
            local_address=local_address,
            band=band,
        )
        self._loop = loop
        # new isolation with current loop,
        # so that session created in tile and execute
        # can get the right isolation
        new_isolation(loop=self._loop, threaded=False)

        self._running_session_id = None
        self._running_op_key = None

        # APIs
        self._cluster_api = None
        self._session_api = None
        self._meta_api = None
        self._subtask_api = None

    async def init(self):
        self._cluster_api = await ClusterAPI.create(self.supervisor_address)
        self._session_api = await SessionAPI.create(self.supervisor_address)
        self._meta_api = await MetaAPI.create(self.session_id, self.supervisor_address)
        try:
            self._subtask_api = await SubtaskAPI.create(self.local_address)
        except mo.ActorNotExist:
            pass

    def _call(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @implements(Context.get_current_session)
    def get_current_session(self) -> SessionType:
        from ..deploy.oscar.session import new_session

        return new_session(
            self.supervisor_address, self.session_id, new=False, default=False
        )

    @implements(Context.get_local_host_ip)
    def get_local_host_ip(self) -> str:
        local_address = self.local_address
        if is_ray_address(local_address):
            import ray

            return ray.util.get_node_ip_address()
        else:
            return local_address.split(":", 1)[0]

    @implements(Context.get_supervisor_addresses)
    def get_supervisor_addresses(self) -> List[str]:
        return self._call(self._cluster_api.get_supervisors())

    @implements(Context.get_worker_addresses)
    def get_worker_addresses(self) -> List[str]:
        return list(self._call(self._cluster_api.get_nodes_info(role=NodeRole.WORKER)))

    @implements(Context.get_worker_bands)
    def get_worker_bands(self) -> List[BandType]:
        return list(self._call(self._cluster_api.get_all_bands(NodeRole.WORKER)))

    @implements(Context.get_total_n_cpu)
    def get_total_n_cpu(self) -> int:
        all_bands = self._call(self._cluster_api.get_all_bands())
        n_cpu = 0
        for band, resource in all_bands.items():
            _, band_name = band
            if band_name.startswith("numa-"):
                n_cpu += resource.num_cpus
        return n_cpu

    @implements(Context.get_slots)
    def get_slots(self) -> int:
        worker_bands = self._call(self._get_worker_bands())
        resource = worker_bands[self.band]
        return int(resource.num_cpus or resource.num_gpus)

    async def _get_worker_bands(self):
        worker_cluster_api = await ClusterAPI.create(self.worker_address)
        return await worker_cluster_api.get_bands()

    async def _get_chunks_meta(
        self, data_keys: List[str], fields: List[str] = None, error: str = "raise"
    ) -> List[Dict]:
        # get chunks meta
        get_metas = []
        for data_key in data_keys:
            meta = self._meta_api.get_chunk_meta.delay(
                data_key, fields=["bands"], error=error
            )
            get_metas.append(meta)
        supervisor_metas = await self._meta_api.get_chunk_meta.batch(*get_metas)
        key_to_supervisor_metas = dict(zip(data_keys, supervisor_metas))
        api_to_keys_calls = defaultdict(lambda: (list(), list()))
        for data_key, meta in zip(data_keys, supervisor_metas):
            addr = meta["bands"][0][0]
            worker_meta_api = await WorkerMetaAPI.create(self.session_id, addr)
            keys, calls = api_to_keys_calls[worker_meta_api]
            keys.append(data_key)
            calls.append(
                worker_meta_api.get_chunk_meta.delay(
                    data_key, fields=fields, error=error
                )
            )
        coros = []
        for api, (keys, calls) in api_to_keys_calls.items():
            coros.append(api.get_chunk_meta.batch(*calls))
        all_metas = await asyncio.gather(*coros)
        key_to_meta = dict()
        for (keys, _), metas in zip(api_to_keys_calls.values(), all_metas):
            for k, meta in zip(keys, metas):
                meta["bands"] = key_to_supervisor_metas[k]["bands"]
                key_to_meta[k] = meta
        return [key_to_meta[k] for k in data_keys]

    async def _get_chunks_result(self, data_keys: List[str]) -> List:
        metas = await self._get_chunks_meta(data_keys, fields=["bands"])
        addresses = [meta["bands"][0][0] for meta in metas]

        storage_api_to_gets = defaultdict(lambda: (list(), list()))
        for data_key, address in zip(data_keys, addresses):
            storage_api = await StorageAPI.create(self.session_id, address)
            storage_api_to_gets[storage_api][0].append(data_key)
            storage_api_to_gets[storage_api][1].append(storage_api.get.delay(data_key))
        results = dict()
        for storage_api, (keys, gets) in storage_api_to_gets.items():
            chunks_data = await storage_api.get.batch(*gets)
            for chunk_key, chunk_data in zip(keys, chunks_data):
                results[chunk_key] = chunk_data
        return [results[key] for key in data_keys]

    async def _fetch_chunks(self, data_keys: List[str]):
        metas = await self._get_chunks_meta(data_keys, fields=["bands"])
        bands = [meta["bands"][0] for meta in metas]

        storage_api = await StorageAPI.create(self.session_id, self.local_address)
        fetches = []
        for data_key, (address, band_name) in zip(data_keys, bands):
            fetches.append(
                storage_api.fetch.delay(
                    data_key, remote_address=address, band_name=band_name
                )
            )
        await storage_api.fetch.batch(*fetches)

    @implements(Context.get_chunks_result)
    def get_chunks_result(self, data_keys: List[str], fetch_only: bool = False) -> List:
        if not fetch_only:
            return self._call(self._get_chunks_result(data_keys))
        else:
            return self._call(self._fetch_chunks(data_keys))

    @implements(Context.get_chunks_meta)
    def get_chunks_meta(
        self, data_keys: List[str], fields: List[str] = None, error="raise"
    ) -> List[Dict]:
        return self._call(self._get_chunks_meta(data_keys, fields=fields, error=error))

    async def _get_backend_info(
        self, address: str = None, level: StorageLevel = StorageLevel.MEMORY
    ) -> dict:
        if address is None:
            address = self.worker_address
        storage_api = await StorageAPI.create(self.session_id, address)
        return await storage_api.get_storage_info(level)

    @implements(Context.get_storage_info)
    def get_storage_info(
        self, address: str = None, level: StorageLevel = StorageLevel.MEMORY
    ):
        return self._call(self._get_backend_info(address, level))

    @implements(Context.create_remote_object)
    def create_remote_object(self, name: str, object_cls, *args, **kwargs):
        ref = self._call(
            self._session_api.create_remote_object(
                self.session_id, name, object_cls, *args, **kwargs
            )
        )
        return _RemoteObjectWrapper(ref, self._loop)

    @implements(Context.get_remote_object)
    def get_remote_object(self, name: str):
        ref = self._call(self._session_api.get_remote_object(self.session_id, name))
        return _RemoteObjectWrapper(ref, self._loop)

    @implements(Context.destroy_remote_object)
    def destroy_remote_object(self, name: str):
        return self._call(
            self._session_api.destroy_remote_object(self.session_id, name)
        )

    @implements(Context.register_custom_log_path)
    def register_custom_log_path(
        self,
        session_id: str,
        tileable_op_key: str,
        chunk_op_key: str,
        worker_address: str,
        log_path: str,
    ):
        return self._call(
            self._session_api.register_custom_log_path(
                session_id, tileable_op_key, chunk_op_key, worker_address, log_path
            )
        )

    @implements(Context.new_custom_log_dir)
    @lru_cache(50)
    def new_custom_log_dir(self) -> str:
        return self._call(
            self._session_api.new_custom_log_dir(self.local_address, self.session_id)
        )

    def set_running_operand_key(self, session_id: str, op_key: str):
        self._running_session_id = session_id
        self._running_op_key = op_key

    def set_progress(self, progress: float):
        if (
            self._running_op_key is None or self._subtask_api is None
        ):  # pragma: no cover
            return
        return self._call(
            self._subtask_api.set_running_operand_progress(
                session_id=self._running_session_id,
                op_key=self._running_op_key,
                slot_address=self.local_address,
                progress=progress,
            )
        )


class _RemoteObjectWrapper:
    def __init__(self, ref: mo.ActorRef, loop: asyncio.AbstractEventLoop):
        self._ref = ref
        self._loop = loop

    def __getattr__(self, attr):
        func = getattr(self._ref, attr)

        def wrap(*args, **kwargs):
            coro = func(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            return fut.result()

        return wrap
