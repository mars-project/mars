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

from typing import Union, Dict

from ...core.session import new_session
from ...resource import cpu_count, cuda_count
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import Session


async def new_cluster(address: str = '0.0.0.0',
                      n_cpu: Union[int, str] = 'auto',
                      n_gpu: Union[int, str] = 'auto',
                      subprocess_start_method: str = 'spawn',
                      config: Union[str, Dict] = None):
    cluster = LocalCluster(address, n_cpu, n_gpu,
                           subprocess_start_method, config)
    await cluster.start()
    return await LocalClient.create(cluster)


class LocalCluster:
    def __init__(self,
                 address: str = '0.0.0.0',
                 n_cpu: Union[int, str] = 'auto',
                 n_gpu: Union[int, str] = 'auto',
                 subprocess_start_method: str = 'spawn',
                 config: Union[str, Dict] = None):
        self._address = address
        self._subprocess_start_method = subprocess_start_method
        self._config = config
        self._n_cpu = cpu_count() if n_cpu == 'auto' else n_cpu
        self._n_gpu = cuda_count() if n_gpu == 'auto' else n_gpu
        self._band_to_slot = band_to_slot = dict()
        band_to_slot['numa-0'] = n_cpu
        for i in range(self._n_gpu):  # pragma: no cover
            band_to_slot[f'gpu-{i}'] = 1
        self._supervisor_pool = None
        self._worker_pool = None

    async def start(self):
        self._supervisor_pool = await create_supervisor_actor_pool(
            self._address, n_process=0,
            subprocess_start_method=self._subprocess_start_method)
        self._worker_pool = await create_worker_actor_pool(
            self._address, self._band_to_slot,
            subprocess_start_method=self._subprocess_start_method)
        # start service
        await start_supervisor(
            self._supervisor_pool.external_address, config=self._config)
        await start_worker(
            self._worker_pool.external_address,
            self._supervisor_pool.external_address,
            self._band_to_slot,
            config=self._config)

    async def stop(self):
        await stop_worker(self._worker_pool, self._config)
        await stop_supervisor(self._supervisor_pool, self._config)
        await self._worker_pool.stop()
        await self._supervisor_pool.stop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()


class LocalClient:
    def __init__(self,
                 cluster: LocalCluster,
                 session: Session):
        self._cluster = cluster
        self._session = session

    @classmethod
    async def create(cls,
                     cluster: LocalCluster) -> "LocalClient":
        session = await new_session(
            cluster._supervisor_pool.external_address,
            backend=Session.name, default=True)
        return LocalClient(cluster, session)

    @property
    def session(self):
        return self._session

    @property
    def cluster(self):
        return self._cluster

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await self._session.destroy()
        await self._cluster.stop()
