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
from typing import List, Optional

from mars import oscar as mo
from mars.lib.uhashring import HashRing
from mars.services.cluster.backends import AbstractClusterBackend, get_cluster_backend
from mars.utils import extensible

logger = logging.getLogger(__name__)


class SupervisorLocatorActor(mo.Actor):
    _backend: Optional[AbstractClusterBackend]

    def __init__(self, backend_name: str, lookup_address: str):
        self._backend_name = backend_name
        self._lookup_address = lookup_address
        self._backend = None
        self._supervisors = None
        self._hash_ring = None

        self._watch_events = set()
        self._watch_task = None

    async def __post_create__(self):
        backend_cls = get_cluster_backend(self._backend_name)
        self._backend = await backend_cls.create(self._lookup_address)
        self._set_supervisors(await self._backend.get_supervisors())

        self._watch_task = asyncio.create_task(self._watch_backend())

    async def __pre_destroy__(self):
        self._watch_task.cancel()

    def get_supervisors(self):
        return self._supervisors

    def _set_supervisors(self, supervisors: List[str]):
        self._supervisors = supervisors
        self._hash_ring = HashRing(nodes=supervisors, hash_fn='ketama')

        for ev in self._watch_events:
            ev.set()

    @extensible
    def get_supervisor(self, key: str, size=1):
        if self._supervisors is None:  # pragma: no cover
            return None
        elif size == 1:
            return self._hash_ring.get_node(key)
        else:
            return tuple(it['nodename']
                         for it in self._hash_ring.range(key, size=size))

    async def _watch_backend(self):
        last_supervisors = set()
        async for sv_list in self._backend.watch_supervisors():
            if set(sv_list) != last_supervisors:
                self._set_supervisors(sv_list)
                last_supervisors = set(sv_list)

    async def watch_supervisors(self):
        event = asyncio.Event()
        self._watch_events.add(event)

        async def waiter():
            try:
                await event.wait()
                return self._supervisors
            finally:
                self._watch_events.remove(event)

        return waiter()

    async def watch_supervisors_by_keys(self, keys):
        event = asyncio.Event()
        self._watch_events.add(event)

        async def waiter():
            try:
                await event.wait()
                return [self.get_supervisor(k) for k in keys]
            finally:
                self._watch_events.remove(event)

        return waiter()

    async def wait_all_supervisors_ready(self):
        expected_supervisors = await self._backend.get_expected_supervisors()
        if set(self._supervisors or []) == set(expected_supervisors):
            return

        event = asyncio.Event()
        self._watch_events.add(event)

        async def waiter():
            while True:
                await event.wait()

                expected_supervisors = await self._backend.get_expected_supervisors()
                if set(self._supervisors) == set(expected_supervisors):
                    self._watch_events.remove(event)
                    return
                else:
                    event.clear()

        return waiter()
