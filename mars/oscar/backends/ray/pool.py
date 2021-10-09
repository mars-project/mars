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
import inspect
import itertools
import logging
import os
import sys
import types
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from .communication import ChannelID, RayServer, RayChannelException
from .utils import process_address_to_placement, process_placement_to_address, get_placement_group, kill_and_wait
from ..config import ActorPoolConfig
from ..message import CreateActorMessage
from ..pool import AbstractActorPool, MainActorPoolBase, SubActorPoolBase, create_actor_pool, _register_message_handler
from ..router import Router
from ... import ServerClosed
from ....serialization.ray import register_ray_serializers
from ....utils import lazy_import

ray = lazy_import('ray')
logger = logging.getLogger(__name__)
_is_windows: bool = sys.platform.startswith('win')


class RayPoolState(Enum):
    INIT = 0
    POOL_READY = 1
    SERVICE_READY = 2


@_register_message_handler
class RayMainActorPool(MainActorPoolBase):
    @classmethod
    def process_index_gen(cls, address):
        _, __, process_index = process_address_to_placement(address)
        return itertools.count(process_index)

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        assert not ports, f"ports should be none when actor pool running on ray, but got {ports}"
        pg_name, bundle_index, process_index = process_address_to_placement(address)
        return [process_placement_to_address(pg_name, bundle_index, process_index + i) for i in range(n_process + 1)]

    @classmethod
    def gen_internal_address(cls, process_index: int, external_address: str = None) -> str:
        return external_address

    @classmethod
    async def start_sub_pool(
            cls,
            actor_pool_config: ActorPoolConfig,
            process_index: int,
            start_method: str = None):
        config = actor_pool_config.get_pool_config(process_index)
        external_addresses = config['external_address']
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
        num_cpus = config['kwargs'].get('sub_pool_cpus', 1)
        actor_handle = ray.remote(RaySubPool).options(
            num_cpus=num_cpus, name=external_address,
            max_concurrency=10000,  # By default, 1000 tasks can be running concurrently.
            max_restarts=-1,  # Auto restarts by ray
            placement_group=pg, placement_group_bundle_index=bundle_index).remote(
                actor_pool_config, process_index)
        await actor_handle.start.remote()
        return actor_handle

    @classmethod
    async def wait_sub_pools_ready(cls,
                                   create_pool_tasks: List[asyncio.Task]):
        return [await t for t in create_pool_tasks]

    async def recover_sub_pool(self, address: str):
        process = self.sub_processes[address]
        await process.start.remote()

        if self._auto_recover == 'actor':
            # need to recover all created actors
            for _, message in self._allocated_actors[address].values():
                create_actor_message: CreateActorMessage = message
                await self.call(address, create_actor_message)
            await process.mark_service_ready.remote()

    async def kill_sub_pool(self, process: 'ray.actor.ActorHandle', force: bool = False):
        await kill_and_wait(process)

    async def is_sub_pool_alive(self, process: 'ray.actor.ActorHandle'):
        try:
            if self._auto_recover == 'process':
                return await process.state.remote() in [RayPoolState.POOL_READY, RayPoolState.SERVICE_READY]
            else:
                return await process.state.remote() == RayPoolState.SERVICE_READY
        except Exception:
            logger.info("Detected RaySubPool %s died", process)
            return False


@_register_message_handler
class RaySubActorPool(SubActorPoolBase):

    async def stop(self):
        try:
            # clean global router
            Router.get_instance().remove_router(self._router)
            await self._caller.stop()
            self._servers = []
        finally:
            self._stopped.set()


class RayPoolBase(ABC):
    __slots__ = '_actor_pool', '_ray_server'

    _actor_pool: Optional['AbstractActorPool']
    _state: RayPoolState = RayPoolState.INIT

    def __new__(cls, *args, **kwargs):
        if not _is_windows:
            try:
                if 'COV_CORE_SOURCE' in os.environ:  # pragma: no branch
                    # register coverage hooks on SIGTERM
                    from pytest_cov.embed import cleanup_on_sigterm
                    cleanup_on_sigterm()
            except ImportError:  # pragma: no cover
                pass
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self._actor_pool = None
        self._ray_server = None
        register_ray_serializers()
        RayServer.set_ray_actor_started()

    @abstractmethod
    async def start(self):
        """Start actor pool in ray actor"""

    def _set_ray_server(self, actor_pool: AbstractActorPool):
        ray_servers = [server for server in actor_pool._servers if isinstance(server, RayServer)]
        assert len(ray_servers) == 1, f"Ray only support single server but got {ray_servers}."
        self._ray_server = ray_servers[0]

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        try:
            if self._ray_server is None:
                raise ServerClosed(f'Remote server {channel_id.dest_address} closed')
            return await self._ray_server.__on_ray_recv__(channel_id, message)
        except Exception:  # pragma: no cover
            return RayChannelException(*sys.exc_info())

    async def actor_pool(self, attribute, *args, **kwargs):
        attr = getattr(self._actor_pool, attribute)
        if isinstance(attr, types.MethodType):
            if inspect.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            return attr(*args, **kwargs)
        else:
            return attr

    def state(self):
        return self._state

    @staticmethod
    def getpid():
        return os.getpid()

    async def wait(self, seconds):
        await asyncio.sleep(seconds)

    def cleanup(self):
        logger.info('Cleaning up %s of process %s now', self, os.getpid())
        try:
            from pytest_cov.embed import cleanup
            cleanup()
        except ImportError:  # pragma: no cover
            pass


class RayMainPool(RayPoolBase):
    _actor_pool: RayMainActorPool

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    async def start(self):
        # create mars pool outside the constructor is to avoid ray actor creation failed.
        # ray can't get the creation exception.
        address, n_process = self._args
        assert self._state == RayPoolState.INIT, \
            f"The pool {address} is already started, current state is {self._state}"
        self._actor_pool = await create_actor_pool(
            address, n_process=n_process, pool_cls=RayMainActorPool, **self._kwargs)
        self._set_ray_server(self._actor_pool)
        self._state = RayPoolState.POOL_READY

    async def mark_service_ready(self):
        results = []
        for _, sub_pool in self._actor_pool.sub_processes.items():
            r = sub_pool.mark_service_ready.remote()
            results.append(r)
        await asyncio.gather(*results)
        self._state = RayPoolState.SERVICE_READY
        await self._actor_pool.start_monitor()


class RaySubPool(RayPoolBase):
    _actor_pool: RaySubActorPool

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    async def start(self):
        # create mars pool outside the constructor is to avoid ray actor creation failed.
        # ray can't get the creation exception.
        actor_config, process_index = self._args
        pool_config = actor_config.get_pool_config(process_index)
        assert self._state == RayPoolState.INIT, \
            f"The pool {pool_config['external_address']} is already started, current state is {self._state}"
        env = pool_config['env']
        if env:
            os.environ.update(env)
        self._actor_pool = await RaySubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        })
        self._set_ray_server(self._actor_pool)
        await self._actor_pool.start()
        asyncio.create_task(self._actor_pool.join())
        self._state = RayPoolState.POOL_READY

    def mark_service_ready(self):
        self._state = RayPoolState.SERVICE_READY
