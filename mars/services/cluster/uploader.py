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

from ... import oscar as mo
from .core import NodeInfo
from .gather import gather_node_env, gather_node_resource, gather_node_states

DEFAULT_INFO_UPLOAD_INTERVAL = 1


class NodeInfoUploaderActor(mo.Actor):
    def __init__(self, role=None, dirs=None, interval=None):
        self._info = NodeInfo(role=role)

        self._env_uploaded = False
        self._dirs = dirs

        self._locator_ref = None
        self._node_info_ref = None

        self._interval = interval or DEFAULT_INFO_UPLOAD_INTERVAL
        self._upload_task = None

    async def __post_create__(self):
        from .locator import SupervisorLocatorActor
        from .supervisor.node_info import NodeInfoCollectorActor

        self._locator_ref = await mo.actor_ref(
            SupervisorLocatorActor.default_uid(), address=self.address)
        supervisor_addr = await self._locator_ref.get_supervisor(
            NodeInfoCollectorActor.default_uid())
        self._node_info_ref = await mo.actor_ref(
            NodeInfoCollectorActor.default_uid(), address=supervisor_addr)

        self._upload_task = self.ref().upload_node_info.tell_delay(delay=0)

    async def __pre_destroy__(self):
        self._upload_task.cancel()

    async def upload_node_info(self, call_next: bool = True):
        if not self._info.env:
            self._info.env = gather_node_env()
        self._info.resource = gather_node_resource()
        self._info.state.update(gather_node_states(dirs=self._dirs))

        await self._node_info_ref.update_node_info(
            address=self.address, role=self._info.role,
            env=self._info.env if not self._env_uploaded else None,
            resource=self._info.resource, state=self._info.state,
        )
        self._env_uploaded = True

        if call_next:
            self._upload_task = self.ref().upload_node_info.tell_delay(delay=self._interval)

    def set_state_value(self, key, value):
        self._info.state[key] = value
