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
import logging
import os
import posixpath
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse, unquote

from .communication.ray import ChannelID, RayServer
from .config import ActorPoolConfig
from .pool import AbstractActorPool, MainActorPool, SubActorPool, create_actor_pool
from ....utils import lazy_import

ray = lazy_import('ray')
logger = logging.getLogger(__name__)


class RayActorPoolMixin(ABC, AbstractActorPool):

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        assert len(self._servers) == 1, "Ray only support single server."
        ray_server: RayServer = self._servers[0]
        await ray_server.__on_ray_recv__(channel_id, message)


class RayMainActorPool(RayActorPoolMixin, MainActorPool):

    def get_server(self):
        assert len(self._servers) == 1, "Ray only support single server."
        ray_server: RayServer = self._servers[0]
        return ray_server

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        assert not ports, f"ports should be none when actor pool running on ray, but got {ports}"
        return [f'{address}/{i}' for i in range(n_process + 1)]

    @classmethod
    def get_sub_pool_manager_cls(cls):
        return MainActorPool.ProcessSubActorPoolManager

    class RaySubActorPoolManager(MainActorPool.SubActorPoolManager):

        @classmethod
        def start_sub_pool(
                cls,
                actor_config: ActorPoolConfig,
                process_index: int,
                start_method: str = None):
            external_addresses = \
                actor_config.get_pool_config(process_index)['external_address']
            assert len(external_addresses) == 1,\
                f"Ray pool allows only one external address but got {external_addresses}"
            external_address = external_addresses[0]
            pg_name, bundle_index, _process_index = address_to_placement_info(external_address)
            assert process_index == _process_index
            pg = ray.util.get_placement_group(pg_name) if pg_name else None
            # Hold actor_handle to avoid actor being freed.
            actor_handle = ray.remote(RaySubPool).options(
                name=external_address, placement_group=pg,
                placement_group_bundle_index=bundle_index).remote()
            ray.get(actor_handle.start(actor_config, process_index).remote())
            return actor_handle

        def kill_sub_pool(self, process: 'ray.actor.ActorHandle'):
            ray.kill(process)

        async def is_sub_pool_alive(self, process: 'ray.actor.ActorHandle'):
            try:
                await process.health_check.remote()
                return True
            except Exception:
                logger.info("Detected RaySubPool actor {} died", process)
                return False


class PoolStatus(Enum):
    HEALTHY = 0
    UNHEALTHY = 1


class RayPoolBase(ABC):
    actor_pool: Optional['RayActorPoolMixin']

    def __init__(self):
        self.actor_pool = None

    @abstractmethod
    async def start(self, *args, **kwargs):
        raise NotImplementedError

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        await self.actor_pool.__on_ray_recv__(channel_id, message)

    def health_check(self):
        return PoolStatus.HEALTHY


class RayMainPool(RayPoolBase):
    actor_pool: RayMainActorPool

    async def start(self, *args, **kwargs):
        self.actor_pool = await create_actor_pool(*args, pool_cls=RayMainActorPool, **kwargs)


class RaySubPool(RayPoolBase):
    actor_pool: SubActorPool

    async def start(self, *args, **kwargs):
        actor_config, process_index = args
        env = actor_config.get_pool_config(process_index)['env']
        if env:
            os.environ.update(env)
        pool = asyncio.run(SubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        }))
        asyncio.run(pool.start())
        self.actor_pool = pool
        asyncio.create_task(pool.join())


def address_to_placement_info(address):
    """
    Args:
        address: The address of an actor pool which running in a ray actor. It's also
        the name of the ray actor.

    Returns:
        A tuple consisting of placement group name, bundle index, process index.

    """
    parsed_url = urlparse(unquote(address))
    if parsed_url.scheme != "ray":
        raise ValueError(f"The address scheme is not ray: {address}")
    # os.path.split will not handle backslashes (\) correctly,
    # so we use the posixpath.
    parts = []
    if parsed_url.netloc:
        tmp = parsed_url.path
        while tmp and tmp != "/":
            tmp, item = posixpath.split(tmp)
            parts.append(item)
    if parts and len(parts) != 2:
        raise ValueError(f"Only bundle index and process index path are allowed in ray "
                         f"address {address}.")
    name, bundle_index, process_index = [parsed_url.netloc] + parts if parts else ""
    if bool(name) != bool(bundle_index) or bool(bundle_index) != bool(process_index):
        raise ValueError(f"Missing placement group name or bundle index or process index "
                         f"from address {address}")
    if name and bundle_index:
        return name, int(bundle_index), int(process_index)
    else:
        return name, -1, -1
