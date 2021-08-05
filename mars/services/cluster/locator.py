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
from typing import List, Optional

from ... import oscar as mo
from ...lib.uhashring import HashRing
from .backends import AbstractClusterBackend, get_cluster_backend
from .core import NodeRole, WatchNotifier

logger = logging.getLogger(__name__)


class SupervisorLocatorActor(mo.Actor):
    _backend: Optional[AbstractClusterBackend]
    _node_role: NodeRole = None

    def __init__(self, backend_name: str, lookup_address: str):
        self._backend_name = backend_name
        self._lookup_address = lookup_address
        self._backend = None
        self._supervisors = None
        self._hash_ring = None

        self._watch_notifier = WatchNotifier()
        self._watch_task = None

    async def __post_create__(self):
        backend_cls = get_cluster_backend(self._backend_name)
        self._backend = await backend_cls.create(
            self._node_role, self._lookup_address, self.address)
        await self._set_supervisors(await self._get_supervisors_from_backend())

        self._watch_task = asyncio.create_task(self._watch_supervisor_changes())

    async def __pre_destroy__(self):
        self._watch_task.cancel()

    async def _set_supervisors(self, supervisors: List[str]):
        self._supervisors = supervisors
        self._hash_ring = HashRing(nodes=supervisors, hash_fn='ketama')
        await self._watch_notifier.notify()

    async def _get_supervisors_from_backend(self, filter_ready: bool = True):
        raise NotImplementedError

    def _watch_supervisors_from_backend(self):
        raise NotImplementedError

    async def _watch_supervisor_changes(self):
        last_supervisors = set()
        try:
            async for sv_list in self._watch_supervisors_from_backend():
                if set(sv_list) != last_supervisors:
                    await self._set_supervisors(sv_list)
                    last_supervisors = set(sv_list)
        except asyncio.CancelledError:
            return

    async def get_supervisors(self, filter_ready: bool = True):
        if filter_ready:
            return self._supervisors
        else:
            return await self._get_supervisors_from_backend(filter_ready=filter_ready)

    @mo.extensible
    def get_supervisor(self, key: str, size=1):
        if self._supervisors is None:  # pragma: no cover
            return None
        elif size == 1:
            return self._hash_ring.get_node(key)
        else:
            return tuple(it['nodename']
                         for it in self._hash_ring.range(key, size=size))

    async def watch_supervisors(self, version: Optional[int] = None):
        version = yield self._watch_notifier.watch(version)
        raise mo.Return((version, self._supervisors))

    async def watch_supervisors_by_keys(self, keys: List[str],
                                        version: Optional[int] = None):
        version = yield self._watch_notifier.watch(version)
        raise mo.Return((version, [self.get_supervisor(k) for k in keys]))

    async def wait_all_supervisors_ready(self):
        version = None
        while True:
            expected_supervisors = await self._get_supervisors_from_backend(filter_ready=False)
            if self._supervisors and set(self._supervisors) == set(expected_supervisors):
                break
            version = yield self._watch_notifier.watch(version)
