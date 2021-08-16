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
import os
from typing import Union, Dict, List

from ... import oscar as mo
from ...oscar.backends.ray.driver import RayActorDriver
from ...oscar.backends.ray.utils import (
    process_placement_to_address,
    node_placement_to_address,
)
from ...services import NodeRole
from ...utils import lazy_import
from ..utils import load_service_config_file, get_third_party_modules_from_config
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from .session import _new_session, AbstractSession, ensure_isolation_created
from .pool import create_supervisor_actor_pool, create_worker_actor_pool

ray = lazy_import("ray")
logger = logging.getLogger(__name__)

# The default value for supervisor standalone (not share node with worker).
DEFAULT_SUPERVISOR_STANDALONE = False
# The default value for supervisor sub pool count.
DEFAULT_SUPERVISOR_SUB_POOL_NUM = 0


def _load_config(filename=None):
    # use default config
    if not filename:  # pragma: no cover
        d = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(d, 'rayconfig.yml')
    return load_service_config_file(filename)


async def new_cluster(cluster_name: str,
                      supervisor_mem: int = 4 * 1024 ** 3,
                      worker_num: int = 1,
                      worker_cpu: int = 16,
                      worker_mem: int = 32 * 1024 ** 3,
                      config: Union[str, Dict] = None,
                      **kwargs):
    ensure_isolation_created(kwargs)
    if kwargs:  # pragma: no cover
        raise TypeError(f'new_cluster got unexpected '
                        f'arguments: {list(kwargs)}')
    cluster = RayCluster(cluster_name, supervisor_mem, worker_num,
                         worker_cpu, worker_mem, config)
    try:
        await cluster.start()
        return await RayClient.create(cluster)
    except Exception as ex:
        # cleanup the cluster if failed.
        await cluster.stop()
        raise ex


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
        # load config file to dict.
        if not config or isinstance(config, str):
            config = _load_config(config)
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
        self._stopped = False
        self.web_address = None

    async def start(self):
        address_to_resources = dict()
        supervisor_standalone = self._config \
            .get('cluster', {}) \
            .get('ray', {}) \
            .get('supervisor', {}) \
            .get('standalone', DEFAULT_SUPERVISOR_STANDALONE)
        supervisor_sub_pool_num = self._config \
            .get('cluster', {}) \
            .get('ray', {}) \
            .get('supervisor', {}) \
            .get('sub_pool_num', DEFAULT_SUPERVISOR_SUB_POOL_NUM)
        self.supervisor_address = process_placement_to_address(self._cluster_name, 0, 0)
        address_to_resources[node_placement_to_address(self._cluster_name, 0)] = {
            'CPU': 1,
            # 'memory': self._supervisor_mem
        }
        worker_addresses = []
        if supervisor_standalone:
            for worker_index in range(1, self._worker_num + 1):
                worker_address = process_placement_to_address(self._cluster_name, worker_index, 0)
                worker_addresses.append(worker_address)
                worker_node_address = node_placement_to_address(self._cluster_name, worker_index)
                address_to_resources[worker_node_address] = {
                    'CPU': self._worker_cpu,
                    # 'memory': self._worker_mem
                }
        else:
            for worker_index in range(self._worker_num):
                worker_process_index = supervisor_sub_pool_num + 1 if worker_index == 0 else 0
                worker_address = process_placement_to_address(self._cluster_name, worker_index, worker_process_index)
                worker_addresses.append(worker_address)
                worker_node_address = node_placement_to_address(self._cluster_name, worker_index)
                address_to_resources[worker_node_address] = {
                    'CPU': self._worker_cpu,
                    # 'memory': self._worker_mem
                }
        mo.setup_cluster(address_to_resources)

        # third party modules from config
        supervisor_modules = get_third_party_modules_from_config(self._config, NodeRole.SUPERVISOR)
        worker_modules = get_third_party_modules_from_config(self._config, NodeRole.WORKER)

        # create supervisor actor pool
        self._supervisor_pool = await create_supervisor_actor_pool(
            self.supervisor_address, n_process=supervisor_sub_pool_num,
            main_pool_cpus=0, sub_pool_cpus=0, modules=supervisor_modules)
        logger.info('Create supervisor on node %s succeeds.', self.supervisor_address)
        # start service
        await start_supervisor(self.supervisor_address, config=self._config)
        logger.info('Start services on supervisor %s succeeds.', self.supervisor_address)
        worker_pools_and_addresses = await asyncio.gather(
            *[self._start_worker(addr, worker_modules) for addr in worker_addresses])
        logger.info('Create %s workers and start services on workers succeeds.', len(worker_addresses))
        for worker_address, worker_pool in worker_pools_and_addresses:
            self._worker_addresses.append(worker_address)
            self._worker_pools.append(worker_pool)

        from ...services.web.supervisor import WebActor
        web_actor = await mo.actor_ref(WebActor.default_uid(), address=self.supervisor_address)
        self.web_address = await web_actor.get_web_address()

    async def _start_worker(self, worker_address, worker_modules):
        logger.info('Create worker on node %s succeeds.', worker_address)
        worker_pool = await create_worker_actor_pool(
            worker_address, self._band_to_slot, modules=worker_modules)
        await start_worker(worker_address,
                           self.supervisor_address,
                           self._band_to_slot,
                           config=self._config)
        logger.info('Start services on worker %s succeeds.', worker_address)
        return worker_address, worker_pool

    async def stop(self):
        if not self._stopped:
            for worker_address in self._worker_addresses:
                await stop_worker(worker_address, self._config)
            for pool in self._worker_pools:
                await pool.actor_pool.remote('stop')
            if self._supervisor_pool is not None:
                await stop_supervisor(self.supervisor_address, self._config)
                await self._supervisor_pool.actor_pool.remote('stop')
            AbstractSession.reset_default()
            RayActorDriver.stop_cluster()
            self._stopped = True


class RayClient:
    def __init__(self,
                 cluster: RayCluster,
                 session: AbstractSession):
        self._cluster = cluster
        self._address = cluster.supervisor_address
        self._session = session

    @classmethod
    async def create(cls, cluster: RayCluster) -> "RayClient":
        session = await _new_session(cluster.supervisor_address, default=True)
        return RayClient(cluster, session)

    @property
    def address(self):
        return self._session.address

    @property
    def session(self):
        return self._session

    @property
    def web_address(self):
        return self._cluster.web_address

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await self._cluster.stop()
