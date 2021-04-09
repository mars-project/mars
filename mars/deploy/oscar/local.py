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

import asyncio

from ... import oscar as mo
from ...core.session import new_session
from ...resource import cpu_count, cuda_count
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker
from .session import Session


async def new_cluster(address: str = '0.0.0.0',
                      n_cpu: Union[int, str] = 'auto',
                      n_gpu: Union[int, str] = 'auto',
                      subprocess_start_method: str = 'spawn',
                      config: Union[str, Dict] = None):
    # create supervisor actor pool
    supervisor_pool = await create_supervisor_actor_pool(
        address, n_process=0,
        subprocess_start_method=subprocess_start_method)
    band_to_slot = dict()
    if n_cpu == 'auto':
        n_cpu = cpu_count()
    band_to_slot['numa-0'] = n_cpu
    if n_gpu == 'auto':
        n_gpu = cuda_count()
    for i in range(n_gpu):
        band_to_slot[f'gpu-{i}'] = 1
    worker_pool = await create_worker_actor_pool(
        address, band_to_slot,
        subprocess_start_method=subprocess_start_method)

    # start service
    await start_supervisor(
        supervisor_pool.external_address, config=config)
    await start_worker(
        worker_pool.external_address,
        supervisor_pool.external_address,
        band_to_slot,
        config=config)

    return await LocalClient.create(supervisor_pool, worker_pool)


class LocalClient:
    def __init__(self,
                 supervisor_pool: mo.MainActorPoolType,
                 worker_pool: mo.MainActorPoolType,
                 session: Session):
        self._supervisor_pool = supervisor_pool
        self._worker_pool = worker_pool
        self._session = session
        self._address = self._supervisor_pool.external_address

    @classmethod
    async def create(cls,
                     supervisor_pool: mo.MainActorPoolType,
                     worker_pool: mo.MainActorPoolType) -> "LocalClient":
        session = await new_session(
            supervisor_pool.external_address,
            backend=Session.name, default=True)
        return LocalClient(supervisor_pool, worker_pool, session)

    @property
    def address(self):
        return self._session

    @property
    def session(self):
        return self._session

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self._stop()

    async def _stop(self):
        await self._worker_pool.stop()
        await self._supervisor_pool.stop()

    def stop(self):
        asyncio.run(self._stop())
