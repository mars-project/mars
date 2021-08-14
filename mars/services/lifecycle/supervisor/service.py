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
from ...core import AbstractService
from .tracker import LifecycleTrackerActor


class LifecycleSupervisorService(AbstractService):
    async def start(self):
        pass

    async def stop(self):
        pass

    async def create_session(self, session_id: str):
        await mo.create_actor(
            LifecycleTrackerActor, session_id, address=self._address,
            uid=LifecycleTrackerActor.gen_uid(session_id))

    async def destroy_session(self, session_id: str):
        await mo.destroy_actor(mo.create_actor_ref(
            uid=LifecycleTrackerActor.gen_uid(session_id),
            address=self._address)
        )
