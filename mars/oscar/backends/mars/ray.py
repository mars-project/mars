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
from numbers import Number
from typing import List, Optional, Dict
from urllib.parse import urlparse, unquote

from .communication.ray import ChannelID, RayServer
from .config import ActorPoolConfig
from .pool import AbstractActorPool, MainActorPool, SubActorPool, create_actor_pool
from ....utils import lazy_import
from ...backend import BaseActorBackend, register_backend
from ...context import BaseActorContext
from .driver import MarsActorDriver
from .context import MarsActorContext

ray = lazy_import('ray')
logger = logging.getLogger(__name__)


@register_backend
class RayActorBackend(BaseActorBackend):
    @staticmethod
    def name():
        # return None because Mars is default scheme
        return "ray"

    @staticmethod
    def get_context_cls():
        return MarsActorContext

    @staticmethod
    def get_driver_cls():
        return MarsActorDriver


class RayActorPoolMixin(AbstractActorPool, ABC):

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        print(f"__on_ray_recv__ start channel_id {channel_id} message {message}")
        if not hasattr(self, '_external_servers'):
            ray_servers = [server for server in self._servers if isinstance(server, RayServer)]
            assert len(ray_servers) == 1, f"Ray only support single server but got {ray_servers}."
            self._external_servers = ray_servers
        print(f"__on_ray_recv__ end channel_id {channel_id} message {message}")
        return await self._external_servers[0].__on_ray_recv__(channel_id, message)


class RayMainActorPool(RayActorPoolMixin, MainActorPool):

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        assert not ports, f"ports should be none when actor pool running on ray, but got {ports}"
        pg_name, bundle_index, _process_index = address_to_placement_info(address)
        return [pg_bundle_to_address(pg_name, bundle_index, i) for i in range(n_process + 1)]

    @classmethod
    def get_sub_pool_manager_cls(cls):
        return cls.RaySubActorPoolManager

    class RaySubActorPoolManager(MainActorPool.SubActorPoolManager):

        @classmethod
        async def start_sub_pool(
                cls,
                actor_pool_config: ActorPoolConfig,
                process_index: int,
                start_method: str = None):
            ray.init(ignore_reinit_error=True)
            external_addresses = \
                actor_pool_config.get_pool_config(process_index)['external_address']
            assert len(external_addresses) == 1,\
                f"Ray pool allows only one external address but got {external_addresses}"
            external_address = external_addresses[0]
            pg_name, bundle_index, _process_index = address_to_placement_info(external_address)
            assert process_index == _process_index,\
                f"process_index {process_index} is not consistent with index {_process_index} " \
                f"in external_address {external_address}"
            pg = ray.util.get_placement_group(pg_name) if pg_name else None
            # Hold actor_handle to avoid actor being freed.
            actor_handle = ray.remote(RaySubPool).options(
                name=external_address, placement_group=pg,
                placement_group_bundle_index=bundle_index).remote()
            await actor_handle.start.remote(actor_pool_config, process_index)
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


class RaySubActorPool(RayActorPoolMixin, SubActorPool):
    pass


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

    async def start(self, address, n_process, **kwargs):
        self.actor_pool = await create_actor_pool(
            address, n_process=n_process, pool_cls=RayMainActorPool,
            subprocess_start_method="ray", **kwargs)


class RaySubPool(RayPoolBase):
    actor_pool: RaySubActorPool

    async def start(self, *args, **kwargs):
        actor_config, process_index = args
        env = actor_config.get_pool_config(process_index)['env']
        if env:
            os.environ.update(env)
        pool = await RaySubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        })
        await pool.start()
        self.actor_pool = pool
        asyncio.create_task(pool.join())


def address_to_placement_info(address):
    """
    Args:
        address: The address of an actor pool which running in a ray actor. It's also
        the name of the ray actor. address ex: ray://${pg_name}/${bundle_index}/${process_index}

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
    parts = list(reversed(parts))
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


def pg_bundle_to_address(pg_name: str, bundle_index: int, process_index: int = 0):
    return f"ray://{pg_name}/{bundle_index}/{process_index}"


class NodeResourceSpec:
    __slots__ = 'n_process', 'num_cpus', 'memory', 'num_gpus'

    def __init__(self, n_process, num_cpus=None, num_gpus=None, memory=None):
        self.n_process = n_process
        self.num_cpus = num_cpus or n_process
        assert n_process <= self.num_cpus
        self.num_gpus = num_gpus
        self.memory = memory

    def to_bundle(self):
        bundle = {'CPU': self.num_cpus}
        if self.num_gpus:
            bundle['GPU'] = self.num_gpus
        if self.memory:
            bundle['memory'] = self.memory
        return bundle


class ClusterManager:

    def __init__(self, cluster_name, resource_specs: List[NodeResourceSpec]):
        self.cluster_name = cluster_name
        self.resource_specs = resource_specs
        self.nodes_info = []
        # Hold actor handle to avoid being freed.
        self.main_pool_handles = [None] * len(resource_specs)

    def add_node(self, index: int, main_handle, address):
        pg_name, bundle_index, _process_index = address_to_placement_info(address)
        assert len(self.nodes_info) == index
        node_info = [pg_bundle_to_address(pg_name, bundle_index, process_index=i)
                     for i in range(self.resource_specs[index].n_process + 1)]
        self.nodes_info.append(node_info)
        self.main_pool_handles[index] = main_handle

    def get_node_info(self, index):
        return self.nodes_info[index]

    def get_all_nodes_info(self):
        return list(self.nodes_info)

    def addresses(self):
        return [process for node in self.get_all_nodes_info() for process in node]

    def address_to_resources(self) -> Dict[str, Dict[str, Number]]:
        # TODO(chaokunyang) support address resources mapping
        return {process: {"CPU": 1} for node in self.get_all_nodes_info() for process in node}


def create_cluster(cluster_name, resource_specs: List[NodeResourceSpec]):
    pg_name = cluster_name
    bundles = [spec.to_bundle() for spec in resource_specs]
    logger.info("Creating placement group %s with bundles %s.", pg_name, bundles)
    pg = ray.util.placement_group(name=pg_name, bundles=bundles, strategy="SPREAD")
    ray.get(pg.ready())
    logger.info("Create placement group success.")
    cluster_manager = ClusterManager(cluster_name, resource_specs)
    for index, (spec, bundle) in enumerate(zip(resource_specs, bundles)):
        address = pg_bundle_to_address(pg_name, index, process_index=0)
        print(f"address {address}")
        # Hold actor_handle to avoid actor being freed.
        actor_handle = ray.remote(RayMainPool).options(
            name=address, placement_group=pg, placement_group_bundle_index=index).remote()
        ray.get(actor_handle.start.remote(address, spec.n_process))
        cluster_manager.add_node(index, actor_handle, address)
    return cluster_manager
