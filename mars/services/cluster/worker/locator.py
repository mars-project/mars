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

import logging
from typing import List

from .... import oscar as mo
from ..core import NodeRole, NodeStatus
from ..locator import SupervisorLocatorActor

logger = logging.getLogger(__name__)


class WorkerSupervisorLocatorActor(SupervisorLocatorActor):
    _node_role = NodeRole.WORKER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_info_ref = None

    @classmethod
    def default_uid(cls):
        return SupervisorLocatorActor.__name__

    async def _set_supervisors(self, supervisors: List[str]):
        await super()._set_supervisors(supervisors)
        if supervisors and self._node_info_ref is None:
            from ..supervisor.node_info import NodeInfoCollectorActor
            supervisor_addr = self.get_supervisor(
                NodeInfoCollectorActor.default_uid())
            try:
                self._node_info_ref = await mo.actor_ref(
                    uid=NodeInfoCollectorActor.default_uid(), address=supervisor_addr
                )
            except (OSError, mo.ServerClosed, mo.ActorNotExist):
                self._node_info_ref = None

    async def _get_supervisors_from_backend(self, filter_ready: bool = True):
        try:
            assert self._node_info_ref is not None
            statuses = {NodeStatus.READY} if filter_ready \
                else {NodeStatus.READY, NodeStatus.STARTING}
            infos = await self._node_info_ref.get_nodes_info(
                role=NodeRole.SUPERVISOR, statuses=statuses)
            return list(infos)
        except (AssertionError, OSError,
                mo.ServerClosed, mo.ActorNotExist):
            self._node_info_ref = None
            return await self._backend.get_supervisors(filter_ready=filter_ready)

    async def _watch_supervisor_from_node_info(self):
        assert self._node_info_ref is not None
        version = None
        while True:
            version, infos = await self._node_info_ref.watch_nodes(
                role=NodeRole.SUPERVISOR, version=version)
            yield list(infos)

    async def _watch_supervisors_from_backend(self):
        while True:
            try:
                async for supervisors in self._watch_supervisor_from_node_info():
                    yield supervisors
            except (AssertionError, OSError,
                    mo.ServerClosed, mo.ActorNotExist):
                self._node_info_ref = None

            async for supervisors in self._backend.watch_supervisors():
                yield supervisors
                if self._node_info_ref is not None:
                    break
