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

import sys
import weakref
from typing import Union, Dict

from ... import oscar as mo
from ...core.session import _new_session
from ...resource import cpu_count, cuda_count
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import Session
from .typing import ClusterType, ClientType


async def new_cluster(address: str = '0.0.0.0',
                      n_worker: int = 1,
                      n_cpu: Union[int, str] = 'auto',
                      n_gpu: Union[int, str] = 'auto',
                      subprocess_start_method: str = None,
                      backend: str = None,
                      config: Union[str, Dict] = None,
                      web: bool = True) -> ClientType:
    if subprocess_start_method is None:
        subprocess_start_method = \
            'spawn' if sys.platform == 'win32' else 'forkserver'
    cluster = LocalCluster(address, n_worker, n_cpu, n_gpu,
                           subprocess_start_method, config, web)
    await cluster.start()
    return await LocalClient.create(cluster, backend)


class LocalCluster:
    def __init__(self: ClusterType,
                 address: str = '0.0.0.0',
                 n_worker: int = 1,
                 n_cpu: Union[int, str] = 'auto',
                 n_gpu: Union[int, str] = 'auto',
                 subprocess_start_method: str = None,
                 config: Union[str, Dict] = None,
                 web: Union[bool, str] = 'auto'):
        self._address = address
        self._subprocess_start_method = subprocess_start_method
        self._config = config
        self._n_cpu = cpu_count() if n_cpu == 'auto' else n_cpu
        self._n_gpu = cuda_count() if n_gpu == 'auto' else n_gpu
        self._n_worker = n_worker
        self._web = web

        self._band_to_slot = band_to_slot = dict()
        worker_cpus = self._n_cpu // n_worker
        assert worker_cpus > 0, f"{self._n_cpu} cpus are not enough " \
                                f"for {n_worker}, try to decrease workers."
        band_to_slot['numa-0'] = worker_cpus
        for i in range(self._n_gpu):  # pragma: no cover
            band_to_slot[f'gpu-{i}'] = 1
        self._supervisor_pool = None
        self._worker_pools = []

        self.supervisor_address = None
        self.web_address = None

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

    async def _start_supervisor_pool(self):
        self._supervisor_pool = await create_supervisor_actor_pool(
            self._address, n_process=0,
            subprocess_start_method=self._subprocess_start_method)
        self.supervisor_address = self._supervisor_pool.external_address

    async def _start_worker_pools(self):
        for _ in range(self._n_worker):
            worker_pool = await create_worker_actor_pool(
                self._address, self._band_to_slot,
                subprocess_start_method=self._subprocess_start_method)
            self._worker_pools.append(worker_pool)

    async def _start_service(self):
        self._web = await start_supervisor(
            self.supervisor_address, config=self._config,
            web=self._web)
        for worker_pool in self._worker_pools:
            await start_worker(
                worker_pool.external_address,
                self.supervisor_address,
                self._band_to_slot,
                config=self._config)

    async def stop(self):
        for worker_pool in self._worker_pools:
            await stop_worker(worker_pool.external_address, self._config)
        await stop_supervisor(self._supervisor_pool.external_address, self._config)
        for worker_pool in self._worker_pools:
            await worker_pool.stop()
        await self._supervisor_pool.stop()


class LocalClient:
    def __init__(self: ClientType,
                 cluster: ClusterType,
                 session: Session):
        self._cluster = cluster
        self._session = weakref.ref(session)

    @property
    def session(self):
        return self._session()

    @classmethod
    async def create(cls,
                     cluster: LocalCluster,
                     backend: str = None) -> ClientType:
        backend = Session.name if backend is None else backend
        session = await _new_session(
            cluster.external_address,
            backend=backend, default=True)
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
        await self._cluster.stop()
