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
import inspect
import logging
import os
import types
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from ....serialization.ray import register_ray_serializers
from ....utils import lazy_import
from ..config import ActorPoolConfig
from ..pool import AbstractActorPool, MainActorPoolBase, SubActorPoolBase, create_actor_pool, _register_message_handler
from .communication import ChannelID, RayServer
from .utils import process_address_to_placement, process_placement_to_address, get_placement_group

ray = lazy_import('ray')
logger = logging.getLogger(__name__)


@_register_message_handler
class RayMainActorPool(MainActorPoolBase):

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        assert not ports, f"ports should be none when actor pool running on ray, but got {ports}"
        pg_name, bundle_index, _process_index = process_address_to_placement(address)
        return [process_placement_to_address(pg_name, bundle_index, i) for i in range(n_process + 1)]

    @classmethod
    def gen_internal_address(cls, process_index: int, external_address: str = None) -> str:
        return external_address

    @classmethod
    async def start_sub_pool(
            cls,
            actor_pool_config: ActorPoolConfig,
            process_index: int,
            start_method: str = None):
        external_addresses = \
            actor_pool_config.get_pool_config(process_index)['external_address']
        assert len(external_addresses) == 1, \
            f"Ray pool allows only one external address but got {external_addresses}"
        external_address = external_addresses[0]
        pg_name, bundle_index, _process_index = process_address_to_placement(external_address)
        assert process_index == _process_index, \
            f"process_index {process_index} is not consistent with index {_process_index} " \
            f"in external_address {external_address}"
        pg = get_placement_group(pg_name) if pg_name else None
        if not pg:
            bundle_index = -1
        # Hold actor_handle to avoid actor being freed.
        actor_handle = ray.remote(RaySubPool).options(
            num_cpus=1, name=external_address, placement_group=pg,
            placement_group_bundle_index=bundle_index).remote()
        await actor_handle.start.remote(actor_pool_config, process_index)
        return actor_handle

    async def kill_sub_pool(self, process: 'ray.actor.ActorHandle', force: bool = False):
        ray.kill(process)

    async def is_sub_pool_alive(self, process: 'ray.actor.ActorHandle'):
        try:
            await process.health_check.remote()
            return True
        except Exception:
            logger.info("Detected RaySubPool %s died", process)
            return False


@_register_message_handler
class RaySubActorPool(SubActorPoolBase):
    pass


class PoolStatus(Enum):
    HEALTHY = 0
    UNHEALTHY = 1


class RayPoolBase(ABC):
    __slots__ = '_actor_pool', '_ray_server'

    _actor_pool: Optional['AbstractActorPool']

    def __init__(self):
        self._actor_pool = None
        self._ray_server = None
        register_ray_serializers()

    @abstractmethod
    async def start(self, *args, **kwargs):
        """Start actor pool in ray actor"""

    def _set_ray_server(self, actor_pool: AbstractActorPool):
        ray_servers = [server for server in actor_pool._servers if isinstance(server, RayServer)]
        assert len(ray_servers) == 1, f"Ray only support single server but got {ray_servers}."
        self._ray_server = ray_servers[0]

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        return await self._ray_server.__on_ray_recv__(channel_id, message)

    def health_check(self):  # noqa: R0201  # pylint: disable=no-self-use
        return PoolStatus.HEALTHY

    async def actor_pool(self, attribute, *args, **kwargs):
        attr = getattr(self._actor_pool, attribute)
        if isinstance(attr, types.MethodType):
            if inspect.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            return attr(*args, **kwargs)
        else:
            return attr


class RayMainPool(RayPoolBase):
    _actor_pool: RayMainActorPool

    async def start(self, *args, **kwargs):
        address, n_process = args
        self._actor_pool = await create_actor_pool(
            address, n_process=n_process, pool_cls=RayMainActorPool, **kwargs)
        self._set_ray_server(self._actor_pool)


class RaySubPool(RayPoolBase):
    _actor_pool: RaySubActorPool

    async def start(self, *args, **kwargs):
        actor_config, process_index = args
        env = actor_config.get_pool_config(process_index)['env']
        if env:
            os.environ.update(env)
        self._actor_pool = await RaySubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        })
        self._set_ray_server(self._actor_pool)
        await self._actor_pool.start()
        asyncio.create_task(self._actor_pool.join())
