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
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...utils import extensible
from ..subtask import Subtask

APIType = TypeVar('APIType', bound='SchedulingAPI')


class SchedulingAPI(ABC):
    def __init__(self, session_id: str, address: str,
                 manager_ref=None, queueing_ref=None):
        self._session_id = session_id
        self._address = address

        self._manager_ref = manager_ref
        self._queueing_ref = queueing_ref

    @classmethod
    @alru_cache
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str) -> APIType:
        from .supervisor.manager import SubtaskManagerActor
        manager_ref = await mo.actor_ref(
            SubtaskManagerActor.gen_uid(session_id), address=address
        )
        from .supervisor.queueing import SubtaskQueueingActor
        queueing_ref = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(session_id), address=address
        )

        scheduling_api = SchedulingAPI(
            session_id, address, manager_ref, queueing_ref)
        return scheduling_api

    @classmethod
    async def create_session(cls: Type[APIType],
                             session_id: str,
                             address: str,
                             service_config: Optional[Dict] = None) -> APIType:
        service_config = service_config or dict()
        scheduling_config = service_config.get('scheduling', {})

        from .supervisor.assigner import AssignerActor
        assigner_coro = mo.create_actor(
            AssignerActor, session_id, address=address,
            uid=AssignerActor.gen_uid(session_id))

        from .supervisor.queueing import SubtaskQueueingActor
        queueing_coro = mo.create_actor(
            SubtaskQueueingActor, session_id, scheduling_config.get('submit_period'),
            address=address, uid=SubtaskQueueingActor.gen_uid(session_id))

        _assigner_ref, queueing_ref = await asyncio.gather(assigner_coro, queueing_coro)

        from .supervisor.manager import SubtaskManagerActor
        manager_ref = await mo.create_actor(
            SubtaskManagerActor, session_id, address=address,
            uid=SubtaskManagerActor.gen_uid(session_id)
        )

        scheduling_api = SchedulingAPI(
            session_id, address, manager_ref, queueing_ref)
        return scheduling_api

    @classmethod
    async def destroy_session(cls,
                              session_id: str,
                              address: str):
        from .supervisor.queueing import SubtaskQueueingActor
        from .supervisor.manager import SubtaskManagerActor
        from .supervisor.assigner import AssignerActor

        destroy_tasks = []
        for actor_cls in [SubtaskManagerActor, SubtaskQueueingActor, AssignerActor]:
            ref = await mo.actor_ref(actor_cls.gen_uid(session_id), address=address)
            destroy_tasks.append(asyncio.create_task(ref.destroy()))
        await asyncio.gather(*destroy_tasks)

    async def add_subtasks(self,
                           subtasks: List[Subtask],
                           priorities: Optional[List[Tuple]] = None):
        """
        Submit subtasks into scheduling service

        Parameters
        ----------
        subtasks
            list of subtasks to be submitted to service
        priorities
            list of priorities of subtasks
        """
        if priorities is None:
            priorities = [(subtask.priority,) for subtask in subtasks]
        await self._manager_ref.add_subtasks(subtasks, priorities)

    @extensible
    async def update_subtask_priority(self,
                                      subtask_id: str,
                                      priority: Tuple):
        """
        Update priorities of subtasks

        Parameters
        ----------
        subtask_id
            id of subtask to update priority
        priority
            list of priority of subtasks
        """
        raise NotImplementedError

    @update_subtask_priority.batch
    async def update_subtask_priority(self, args_list, kwargs_list):
        await self._queueing_ref.update_subtask_priority.batch(
            *(self._queueing_ref.update_subtask_priority.delay(*args, **kwargs)
              for args, kwargs in zip(args_list, kwargs_list)))

    async def cancel_subtasks(self,
                              subtask_ids: List[str],
                              kill_timeout: Union[float, int] = 5):
        """
        Cancel pending and running subtasks.

        Parameters
        ----------
        subtask_ids
            ids of subtasks to cancel
        kill_timeout
            timeout seconds to kill actor process forcibly
        """
        await self._manager_ref.cancel_subtasks(
            subtask_ids, kill_timeout=kill_timeout)

    async def finish_subtasks(self,
                              subtask_ids: List[str],
                              schedule_next: bool = True):
        """
        Mark subtasks as finished, letting scheduling service to schedule
        next tasks in the ready queue

        Parameters
        ----------
        subtask_ids
            ids of subtasks to mark as finished
        schedule_next
            whether to schedule succeeding subtasks
        """
        await self._manager_ref.finish_subtasks(subtask_ids, schedule_next)


class MockSchedulingAPI(SchedulingAPI):
    @classmethod
    async def create(cls: Type[APIType],
                     session_id: str,
                     address: str) -> APIType:
        from .supervisor import GlobalSlotManagerActor
        await mo.create_actor(GlobalSlotManagerActor,
                              uid=GlobalSlotManagerActor.default_uid(),
                              address=address)

        await super().create_session(session_id, address)

        from ... import resource as mars_resource
        from .worker import SubtaskExecutionActor, MemQuotaActor, \
            WorkerSlotManagerActor
        await mo.create_actor(SubtaskExecutionActor,
                              uid=SubtaskExecutionActor.default_uid(),
                              address=address)
        await mo.create_actor(WorkerSlotManagerActor,
                              uid=WorkerSlotManagerActor.default_uid(),
                              address=address)
        await mo.create_actor(MemQuotaActor, mars_resource.virtual_memory().total,
                              uid=MemQuotaActor.gen_uid('numa-0'),
                              address=address)
