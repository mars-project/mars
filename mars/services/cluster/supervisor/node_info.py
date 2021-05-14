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
import time
from collections import defaultdict
from typing import Dict, List

from .... import oscar as mo
from ...core import NodeRole, BandType
from ..core import NodeInfo

DEFAULT_NODE_DEAD_TIMEOUT = 120
DEFAULT_NODE_CHECK_INTERVAL = 1


class NodeInfoCollectorActor(mo.Actor):
    _node_infos: Dict[str, NodeInfo]

    def __init__(self, timeout=None, check_interval=None):
        self._role_to_nodes = defaultdict(set)
        self._role_to_events = defaultdict(set)

        self._node_infos = dict()

        self._node_timeout = timeout or DEFAULT_NODE_DEAD_TIMEOUT
        self._check_interval = check_interval or DEFAULT_NODE_CHECK_INTERVAL
        self._check_task = None

    async def __post_create__(self):
        self._check_task = self.ref().check_dead_nodes.tell_delay(delay=self._check_interval)

    async def __pre_destroy__(self):
        self._check_task.cancel()

    def check_dead_nodes(self):
        dead_nodes = []
        affect_roles = set()
        for address, info in self._node_infos.items():
            if time.time() - info.update_time > self._node_timeout:
                node_role = info.role
                self._role_to_nodes[node_role].difference_update([address])
                dead_nodes.append(address)
                affect_roles.add(node_role)

        if dead_nodes:
            for address in dead_nodes:
                self._node_infos.pop(address, None)

            self._notify_roles(affect_roles)

        self._check_task = self.ref().check_dead_nodes.tell_delay(delay=self._check_interval)

    def _notify_roles(self, roles):
        for role in roles:
            for event in self._role_to_events[role]:
                event.set()

    def update_node_info(self, address: str, role: NodeRole, env: Dict = None,
                         resource: Dict = None, state: Dict = None):
        is_new = False
        if address not in self._node_infos:
            is_new = True
            info = self._node_infos[address] = NodeInfo(role=role)
        else:
            info = self._node_infos[address]

        info.update_time = time.time()
        if env is not None:
            info.env.update(env)
        if resource is not None:
            info.resource.update(resource)
        if state is not None:
            info.state.update(state)

        if is_new:
            self._role_to_nodes[role].add(address)
            self._notify_roles([role])

    def get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                       env: bool = False, resource: bool = False,
                       state: bool = False):
        if nodes is None:
            nodes = self._role_to_nodes.get(role) if role is not None \
                else self._node_infos.keys()
            nodes = nodes or []
        ret_infos = dict()
        for node in nodes:
            if node not in self._node_infos:
                continue
            info = self._node_infos[node]
            ret_infos[node] = dict(
                update_time=info.update_time,
                env=info.env if env else None,
                resource=info.resource if resource else None,
                state=info.state if state else None,
            )
        return ret_infos

    def get_all_bands(self, role: NodeRole = None) -> Dict[BandType, int]:
        role = role or NodeRole.WORKER
        nodes = self._role_to_nodes.get(role, [])
        band_slots = dict()
        for node in nodes:
            node_resource = self._node_infos[node].resource
            for resource_type, info in node_resource.items():
                if resource_type.startswith('numa'):
                    # cpu
                    band_slots[(node, resource_type)] = info['cpu_total']
                else:  # pragma: no cover
                    assert resource_type.startswith('gpu')
                    band_slots[(node, resource_type)] = info['gpu_total']
        return band_slots

    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, state: bool = False):
        event = asyncio.Event()
        self._role_to_events[role].add(event)

        async def waiter():
            try:
                await event.wait()
                return self.get_nodes_info(
                    role=role, env=env, resource=resource, state=state)
            finally:
                self._role_to_events[role].remove(event)

        return waiter()

    async def watch_all_bands(self, role: NodeRole = None):
        role = role or NodeRole.WORKER
        event = asyncio.Event()
        self._role_to_events[role].add(event)

        async def waiter():
            try:
                await event.wait()
                return self.get_all_bands(role=role)
            finally:
                self._role_to_events[role].remove(event)

        return waiter()
