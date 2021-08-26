# -*- coding: utf-8 -*-
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
import signal
import uuid
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional, TypeVar

from ... import oscar as mo
from ...services import NodeRole
from ...services.cluster.backends import register_cluster_backend, \
    AbstractClusterBackend
from ...utils import to_binary, to_str
from ..utils import wait_all_supervisors_ready
from .config import MarsSupervisorConfig, MarsWorkerConfig

try:
    from skein import ApplicationClient, Client as SkeinClient, \
        properties as skein_props, ConnectionError as SkeinConnectionError, \
        SkeinError
except ImportError:  # pragma: no cover
    ApplicationClient, SkeinClient, skein_props = None, None, None
    SkeinConnectionError, SkeinError = None, None

RetType = TypeVar('RetType')
logger = logging.getLogger(__name__)

_role_to_config = {
    NodeRole.SUPERVISOR: MarsSupervisorConfig,
    NodeRole.WORKER: MarsWorkerConfig,
}


class YarnNodeWatchActor(mo.Actor):
    def __init__(self):
        assert ApplicationClient is not None
        self._app_client = ApplicationClient.from_current()

        self._nodes = defaultdict(set)
        self._supervisor_watch_task = None
        self._role_to_events = defaultdict(list)

    async def __post_create__(self):
        self._supervisor_watch_task = asyncio.create_task(self._watch_nodes(NodeRole.SUPERVISOR))

    async def __pre_destroy__(self):
        if self._supervisor_watch_task is not None:  # pragma: no branch
            self._watch_task.cancel()

    async def get_container_mappings(self, role: NodeRole) -> Dict[str, str]:
        key_prefix = _role_to_config[role].service_name

        container_specs = await asyncio.to_thread(
            self._app_client.get_containers, [key_prefix])
        cid_to_endpoint = {c.yarn_container_id: None for c in container_specs}

        prefixes = await asyncio.to_thread(
            self._app_client.kv.get_prefix, key_prefix)
        for val in prefixes.values():
            ep, cid = to_str(val).split('@', 1)
            cid_to_endpoint[cid] = ep
        return cid_to_endpoint

    async def _watch_nodes(self, role: NodeRole):
        while True:
            try:
                mappings = await self.get_container_mappings(role)
                eps = set(v for v in mappings.values() if v is not None)

                if eps != self._nodes[role]:
                    logger.info('New endpoints retrieved: %r', eps)
                    events = self._role_to_events.pop(role, [])
                    for ev in events:
                        ev.set()
                    self._nodes[role] = eps
                await asyncio.sleep(1)
            except SkeinConnectionError:  # pragma: no cover
                logger.warning('Skein application down, process will terminate')
                os.kill(os.getpid(), signal.SIGTERM)
            except (SkeinError, asyncio.CancelledError):  # pragma: no cover
                logger.exception('Error when watching nodes')
                break

    async def get_nodes(self, role: NodeRole) -> List[str]:
        if not self._nodes[role]:
            mappings = await self.get_container_mappings(role)
            eps = set(v for v in mappings.values() if v is not None)
            self._nodes[role] = eps
        return list(self._nodes[role])

    async def wait_nodes(self, role: NodeRole):
        event = asyncio.Event()
        self._role_to_events[role].append(event)

        async def waiter():
            await event.wait()
            return list(self._supervisors)

        return waiter()


@register_cluster_backend
class YarnClusterBackend(AbstractClusterBackend):
    name = "yarn"

    def __init__(self, pool_address: str, watch_ref: mo.ActorRef = None):
        self._pool_address = pool_address
        self._watch_ref = watch_ref

    @classmethod
    async def create(cls, node_role: NodeRole, lookup_address: Optional[str],
                     pool_address: str) -> "AbstractClusterBackend":
        try:
            ref = await mo.create_actor(
                YarnNodeWatchActor, uid=YarnNodeWatchActor.default_uid(),
                address=pool_address)
        except mo.ActorAlreadyExist:  # pragma: no cover
            ref = await mo.actor_ref(YarnNodeWatchActor.default_uid(),
                                     address=pool_address)
        return YarnClusterBackend(pool_address, ref)

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        if filter_ready:
            return await self._watch_ref.get_nodes(NodeRole.SUPERVISOR)
        else:
            mapping = await self._watch_ref.get_container_mappings(NodeRole.SUPERVISOR)
            return [v if v is not None else k for k, v in mapping.items()]

    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        while True:
            yield await self._watch_ref.wait_nodes(NodeRole.SUPERVISOR)


class YarnServiceMixin(object):
    service_name = None

    @property
    def app_client(self):
        if not hasattr(self, '_app_client'):
            self._app_client = ApplicationClient.from_current()
        return self._app_client

    def get_container_ip(self):
        svc_containers = self.app_client.get_containers([self.service_name])
        container = next(c for c in svc_containers
                         if c.yarn_container_id == skein_props['yarn_container_id'])
        return container.yarn_node_http_address.split(':')[0]

    def register_endpoint(self, prefix: str = None, endpoint: str = None):
        prefix = prefix or self.service_name
        endpoint = endpoint or self.args.endpoint

        container_key = prefix + '-' + str(uuid.uuid1())
        self.app_client.kv[container_key] = to_binary(
            f'{endpoint}@{skein_props["yarn_container_id"]}')

    async def wait_all_supervisors_ready(self):
        """
        Wait till all containers are ready, both in yarn and in Cluster Service
        """
        await wait_all_supervisors_ready(self.args.endpoint)
