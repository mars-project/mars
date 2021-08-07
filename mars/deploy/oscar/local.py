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
import atexit
import logging
import sys
from concurrent.futures import Future as SyncFuture
from typing import Dict, List, Union

import numpy as np

from ... import oscar as mo
from ...lib.aio import get_isolation, stop_isolation
from ...resource import cpu_count, cuda_count
from ...services import NodeRole
from ...typing import ClusterType, ClientType
from ..utils import get_third_party_modules_from_config
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker, load_config
from .session import AbstractSession, _new_session, ensure_isolation_created

logger = logging.getLogger(__name__)

_is_exiting_future = SyncFuture()
atexit.register(lambda: _is_exiting_future.set_result(0)
                if not _is_exiting_future.done() else None)
atexit.register(stop_isolation)


async def new_cluster_in_isolation(
        address: str = '0.0.0.0',
        n_worker: int = 1,
        n_cpu: Union[int, str] = 'auto',
        cuda_devices: Union[List[int], str] = 'auto',
        subprocess_start_method: str = None,
        backend: str = None,
        config: Union[str, Dict] = None,
        web: bool = True,
        timeout: float = None) -> ClientType:
    if subprocess_start_method is None:
        subprocess_start_method = \
            'spawn' if sys.platform == 'win32' else 'forkserver'
    cluster = LocalCluster(address, n_worker, n_cpu, cuda_devices,
                           subprocess_start_method, config, web)
    await cluster.start()
    return await LocalClient.create(cluster, backend, timeout)


async def new_cluster(address: str = '0.0.0.0',
                      n_worker: int = 1,
                      n_cpu: Union[int, str] = 'auto',
                      cuda_devices: Union[List[int], str] = 'auto',
                      subprocess_start_method: str = None,
                      config: Union[str, Dict] = None,
                      web: bool = True,
                      loop: asyncio.AbstractEventLoop = None,
                      use_uvloop: Union[bool, str] = 'auto') -> ClientType:
    coro = new_cluster_in_isolation(
        address, n_worker=n_worker, n_cpu=n_cpu, cuda_devices=cuda_devices,
        subprocess_start_method=subprocess_start_method,
        config=config, web=web)
    isolation = ensure_isolation_created(
        dict(loop=loop, use_uvloop=use_uvloop))
    fut = asyncio.run_coroutine_threadsafe(coro, isolation.loop)
    client = await asyncio.wrap_future(fut)
    client.session.as_default()
    return client


async def stop_cluster(cluster: ClusterType):
    isolation = get_isolation()
    coro = cluster.stop()
    await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(coro, isolation.loop))


class LocalCluster:
    def __init__(self: ClusterType,
                 address: str = '0.0.0.0',
                 n_worker: int = 1,
                 n_cpu: Union[int, str] = 'auto',
                 cuda_devices: Union[List[int], List[List[int]], str] = 'auto',
                 subprocess_start_method: str = None,
                 config: Union[str, Dict] = None,
                 web: Union[bool, str] = 'auto',
                 timeout: float = None):
        # load config file to dict.
        if not config or isinstance(config, str):
            config = load_config(config)
        self._address = address
        self._subprocess_start_method = subprocess_start_method
        self._config = config
        self._n_cpu = cpu_count() if n_cpu == 'auto' else n_cpu
        if cuda_devices == 'auto':
            total = cuda_count()
            all_devices = np.arange(total)
            devices_list = [list(arr) for arr
                            in np.array_split(all_devices, n_worker)]

        else:  # pragma: no cover
            if isinstance(cuda_devices[0], int):
                assert n_worker == 1
                devices_list = [cuda_devices]
            else:
                assert len(cuda_devices) == n_worker
                devices_list = cuda_devices

        self._n_worker = n_worker
        self._web = web
        self._bands_to_slot = bands_to_slot = []
        worker_cpus = self._n_cpu // n_worker
        if sum(len(devices) for devices in devices_list) == 0:
            assert worker_cpus > 0, f"{self._n_cpu} cpus are not enough " \
                                    f"for {n_worker}, try to decrease workers."
        for _, devices in zip(range(n_worker), devices_list):
            worker_band_to_slot = dict()
            worker_band_to_slot['numa-0'] = worker_cpus
            for i in devices:  # pragma: no cover
                worker_band_to_slot[f'gpu-{i}'] = 1
            bands_to_slot.append(worker_band_to_slot)
        self._supervisor_pool = None
        self._worker_pools = []

        self.supervisor_address = None
        self.web_address = None

        self._exiting_check_task = None

    @property
    def external_address(self):
        return self._supervisor_pool.external_address

    async def start(self):
        await self._start_supervisor_pool()
        await self._start_worker_pools()
        # start service
        await self._start_service()

        if self._web:
            from ...services.web.supervisor import WebActor

            web_actor = await mo.actor_ref(WebActor.default_uid(),
                                           address=self.supervisor_address)
            self.web_address = await web_actor.get_web_address()
            logger.warning('Web service started at %s', self.web_address)

        self._exiting_check_task = asyncio.create_task(self._check_exiting())

    async def _check_exiting(self):
        await asyncio.wrap_future(_is_exiting_future)
        await self.stop()

    async def _start_supervisor_pool(self):
        supervisor_modules = get_third_party_modules_from_config(
            self._config, NodeRole.SUPERVISOR)
        self._supervisor_pool = await create_supervisor_actor_pool(
            self._address, n_process=0, modules=supervisor_modules,
            subprocess_start_method=self._subprocess_start_method)
        self.supervisor_address = self._supervisor_pool.external_address

    async def _start_worker_pools(self):
        worker_modules = get_third_party_modules_from_config(
                self._config, NodeRole.WORKER)
        for band_to_slot in self._bands_to_slot:
            worker_pool = await create_worker_actor_pool(
                self._address, band_to_slot, modules=worker_modules,
                subprocess_start_method=self._subprocess_start_method)
            self._worker_pools.append(worker_pool)

    async def _start_service(self):
        self._web = await start_supervisor(
            self.supervisor_address, config=self._config,
            web=self._web)
        for worker_pool, band_to_slot in zip(
                self._worker_pools, self._bands_to_slot):
            await start_worker(
                worker_pool.external_address,
                self.supervisor_address,
                band_to_slot,
                config=self._config)

    async def stop(self):
        for worker_pool in self._worker_pools:
            await stop_worker(worker_pool.external_address, self._config)
        await stop_supervisor(self._supervisor_pool.external_address, self._config)
        for worker_pool in self._worker_pools:
            await worker_pool.stop()
        await self._supervisor_pool.stop()
        AbstractSession.reset_default()
        self._exiting_check_task.cancel()


class LocalClient:
    def __init__(self: ClientType,
                 cluster: ClusterType,
                 session: AbstractSession):
        self._cluster = cluster
        self.session = session

    @classmethod
    async def create(cls,
                     cluster: LocalCluster,
                     backend: str = None,
                     timeout: float = None) -> ClientType:
        backend = backend or 'oscar'
        session = await _new_session(
            cluster.external_address, backend=backend,
            default=True, timeout=timeout)
        client = LocalClient(cluster, session)
        session.client = client
        return client

    @property
    def web_address(self):
        return self._cluster.web_address

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await stop_cluster(self._cluster)
