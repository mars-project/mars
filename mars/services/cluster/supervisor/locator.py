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

from .... import oscar as mo
from ....lib.aio import alru_cache
from ..core import NodeRole
from ..locator import SupervisorLocatorActor


class SupervisorPeerLocatorActor(SupervisorLocatorActor):
    _node_role = NodeRole.SUPERVISOR

    @classmethod
    def default_uid(cls):
        return SupervisorLocatorActor.__name__

    async def __post_create__(self):
        await super().__post_create__()

        supervisors = await self._backend.get_supervisors(filter_ready=False)
        try:
            node_info_ref = await self._get_node_info_ref()
            await node_info_ref.put_starting_nodes(supervisors, NodeRole.SUPERVISOR)
        except mo.ActorNotExist:  # pragma: no cover
            pass

    @alru_cache(cache_exceptions=False)
    async def _get_node_info_ref(self):
        from .node_info import NodeInfoCollectorActor
        return await mo.actor_ref(uid=NodeInfoCollectorActor.default_uid(),
                                  address=self.address)

    async def _get_supervisors_from_backend(self, filter_ready: bool = True):
        return await self._backend.get_supervisors(filter_ready=filter_ready)

    async def _watch_supervisors_from_backend(self):
        async for supervisors in self._backend.watch_supervisors():
            yield supervisors
