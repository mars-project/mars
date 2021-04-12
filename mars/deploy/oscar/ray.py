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
from typing import Union, Dict

from mars.oscar.backends.ray.driver import RayActorDriver
from .service import start_supervisor, start_worker
from .session import Session
from ... import oscar as mo
from ...core.session import new_session
from ...utils import lazy_import

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


async def new_cluster(cluster_name: str,
                      supervisor_cpu: int = 2,
                      supervisor_mem: int = 4 * 1024 ** 3,
                      worker_num: int = 1,
                      worker_cpu: int = 16,
                      worker_mem: int = 32 * 1024 ** 3,
                      config: Union[str, Dict] = None):
    supervisor_node_address = f'ray://{cluster_name}/0'
    worker_node_addresses = []
    address_to_resources = {supervisor_node_address: {
        'CPU': supervisor_cpu,
        # 'memory': supervisor_mem
    }}
    for worker_index in range(1, worker_num + 1):
        worker_node_address = f'ray://{cluster_name}/{worker_index}'
        worker_node_addresses.append(worker_node_address)
        address_to_resources[worker_node_address] = {
            'CPU': worker_cpu,
            # 'memory': worker_mem
        }
    mo.setup_cluster(address_to_resources)
    # start service
    supervisor_address = f'{supervisor_node_address}/0'
    await start_supervisor(supervisor_address, config=config)
    band_to_slot = {'numa-0': worker_cpu}
    # TODO(chaokunyang) Support gpu
    for worker_node_address in worker_node_addresses:
        worker_main_actor_pool_address = f'{worker_node_address}/0'
        await start_worker(worker_main_actor_pool_address,
                           supervisor_address,
                           band_to_slot,
                           config=config)
    return await RayClient.create(supervisor_address)


class RayClient:
    def __init__(self,
                 supervisor_address: str,
                 session: Session):
        self._address = supervisor_address
        self._session = session

    @classmethod
    async def create(cls, supervisor_address: str) -> "RayClient":
        session = await new_session(supervisor_address, backend=Session.name, default=True)
        return RayClient(supervisor_address, session)

    @property
    def address(self):
        return self._session

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
