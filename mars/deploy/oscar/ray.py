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

import logging
import os
import yaml
from typing import Union, Dict, List

from mars.oscar.backends.ray.driver import RayActorDriver
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import Session
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from ... import oscar as mo
from ...core.session import _new_session, AbstractSession
from ...utils import lazy_import

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


def _load_config(filename=None):
    # use default config
    if not filename:  # pragma: no cover
        d = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(d, 'rayconfig.yml')
    with open(filename) as f:
        return yaml.safe_load(f)


async def new_cluster(cluster_name: str,
                      supervisor_mem: int = 4 * 1024 ** 3,
                      worker_num: int = 1,
                      worker_cpu: int = 16,
                      worker_mem: int = 32 * 1024 ** 3,
                      config: Union[str, Dict] = None):
    config = config or _load_config()
    cluster = RayCluster(cluster_name, supervisor_mem, worker_num,
                         worker_cpu, worker_mem, config)
    await cluster.start()
    return await RayClient.create(cluster)


class RayCluster:
    _supervisor_pool: 'ray.actor.ActorHandle'
    _worker_pools: List['ray.actor.ActorHandle']

    def __init__(self,
                 cluster_name: str,
                 supervisor_mem: int = 4 * 1024 ** 3,
                 worker_num: int = 1,
                 worker_cpu: int = 16,
                 worker_mem: int = 32 * 1024 ** 3,
                 config: Union[str, Dict] = None):
        self._cluster_name = cluster_name
        self._supervisor_mem = supervisor_mem
        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = worker_mem
        self._config = config
        self._band_to_slot = band_to_slot = dict()
        # TODO(chaokunyang) Support gpu
        band_to_slot['numa-0'] = self._worker_cpu
        self.supervisor_address = None
        # Hold actor handles to avoid being freed
        self._supervisor_pool = None
        self._worker_addresses = []
        self._worker_pools = []

    async def start(self):
        address_to_resources = dict()
        supervisor_node_address = f'ray://{self._cluster_name}/0'
        address_to_resources[supervisor_node_address] = {
            'CPU': 1,
            # 'memory': self._supervisor_mem
        }
        worker_node_addresses = []
        for worker_index in range(1, self._worker_num + 1):
            worker_node_address = f'ray://{self._cluster_name}/{worker_index}'
            worker_node_addresses.append(worker_node_address)
            address_to_resources[worker_node_address] = {
                'CPU': self._worker_cpu,
                # 'memory': self._worker_mem
            }
        mo.setup_cluster(address_to_resources)

        # create supervisor actor pool
        self._supervisor_pool = await create_supervisor_actor_pool(
            supervisor_node_address, n_process=0)
        # start service
        self.supervisor_address = f'{supervisor_node_address}/0'
        await start_supervisor(self.supervisor_address, config=self._config)

        for worker_node_address in worker_node_addresses:
            worker_pool = await create_worker_actor_pool(worker_node_address, self._band_to_slot)
            self._worker_pools.append(worker_pool)
            worker_address = f'{worker_node_address}/0'
            self._worker_addresses.append(worker_address)
            await start_worker(worker_address,
                               self.supervisor_address,
                               self._band_to_slot,
                               config=self._config)

    async def stop(self):
        for worker_address in self._worker_addresses:
            await stop_worker(worker_address, self._config)
        await stop_supervisor(self.supervisor_address, self._config)
        for pool in self._worker_pools:
            await pool.actor_pool.remote('stop')
        await self._supervisor_pool.actor_pool.remote('stop')


class RayClient:
    def __init__(self,
                 cluster: RayCluster,
                 session: AbstractSession):
        self._cluster = cluster
        self._address = cluster.supervisor_address
        self._session = session

    @classmethod
    async def create(cls, cluster: RayCluster) -> "RayClient":
        session = await _new_session(cluster.supervisor_address, backend=Session.name, default=True)
        return RayClient(cluster, session)

    @property
    def address(self):
        return self._session.address

    @property
    def session(self):
        return self._session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await self._session.destroy()
        RayActorDriver.stop_cluster()
