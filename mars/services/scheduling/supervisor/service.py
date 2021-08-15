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

from .... import oscar as mo
from ...core import AbstractService


class SchedulingSupervisorService(AbstractService):
    """
    Scheduling service on supervisor.

    Scheduling Configuration
    ------------------------
    {
        "scheduling" : {
            "submit_period": 1
        }
    }
    """
    async def start(self):
        from .globalslot import GlobalSlotManagerActor
        await mo.create_actor(
            GlobalSlotManagerActor, uid=GlobalSlotManagerActor.default_uid(),
            address=self._address)

    async def stop(self):
        from .globalslot import GlobalSlotManagerActor
        await mo.destroy_actor(mo.create_actor_ref(
            uid=GlobalSlotManagerActor.default_uid(), address=self._address))

    async def create_session(self, session_id: str):
        service_config = self._config or dict()
        scheduling_config = service_config.get('scheduling', {})

        from .assigner import AssignerActor
        assigner_coro = mo.create_actor(
            AssignerActor, session_id, address=self._address,
            uid=AssignerActor.gen_uid(session_id))

        from .queueing import SubtaskQueueingActor
        queueing_coro = mo.create_actor(
            SubtaskQueueingActor, session_id, scheduling_config.get('submit_period'),
            address=self._address, uid=SubtaskQueueingActor.gen_uid(session_id))

        await asyncio.gather(assigner_coro, queueing_coro)

        from .manager import SubtaskManagerActor
        await mo.create_actor(
            SubtaskManagerActor, session_id, address=self._address,
            uid=SubtaskManagerActor.gen_uid(session_id)
        )

    async def destroy_session(self, session_id: str):
        from .queueing import SubtaskQueueingActor
        from .manager import SubtaskManagerActor
        from .assigner import AssignerActor

        destroy_tasks = []
        for actor_cls in [SubtaskManagerActor, SubtaskQueueingActor, AssignerActor]:
            ref = await mo.actor_ref(
                actor_cls.gen_uid(session_id), address=self._address)
            destroy_tasks.append(asyncio.create_task(ref.destroy()))
        await asyncio.gather(*destroy_tasks)
