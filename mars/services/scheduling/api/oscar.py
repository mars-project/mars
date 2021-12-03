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

from typing import List, Optional, Tuple, Type, TypeVar, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ...subtask import Subtask
from ..core import SubtaskScheduleSummary
from .core import AbstractSchedulingAPI

APIType = TypeVar("APIType", bound="SchedulingAPI")


class SchedulingAPI(AbstractSchedulingAPI):
    def __init__(self, session_id: str, address: str, manager_ref=None):
        self._session_id = session_id
        self._address = address

        self._manager_ref = manager_ref

    @classmethod
    @alru_cache
    async def create(cls: Type[APIType], session_id: str, address: str) -> APIType:
        from ..supervisor.manager import SubtaskManagerActor

        manager_ref = await mo.actor_ref(
            SubtaskManagerActor.gen_uid(session_id), address=address
        )

        scheduling_api = SchedulingAPI(session_id, address, manager_ref)
        return scheduling_api

    async def get_subtask_schedule_summaries(
        self, task_id: Optional[str] = None
    ) -> List[SubtaskScheduleSummary]:
        return await self._manager_ref.get_schedule_summaries(task_id)

    async def cache_subtasks(
        self, subtasks: List[Subtask], priorities: Optional[List[Tuple]] = None
    ):
        """
        Add subtask graph to cache for fast forwarding

        Parameters
        ----------
        subtasks
            list of subtasks to be submitted to service
        priorities
            list of priorities of subtasks
        """
        await self._manager_ref.cache_subtasks(subtasks, priorities)

    async def add_subtasks(
        self, subtasks: List[Subtask], priorities: Optional[List[Tuple]] = None
    ):
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
            priorities = [subtask.priority or tuple() for subtask in subtasks]
        await self._manager_ref.add_subtasks(subtasks, priorities)

    @mo.extensible
    async def update_subtask_priority(self, subtask_id: str, priority: Tuple):
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
        subtask_ids, priorities = [], []
        for args, kwargs in zip(args_list, kwargs_list):
            subtask_id, priority = self.update_subtask_priority.bind(*args, **kwargs)
            subtask_ids.append(subtask_id)
            priorities.append(priority)
        await self._manager_ref.update_subtask_priorities(subtask_ids, priorities)

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Union[float, int] = 5
    ):
        """
        Cancel pending and running subtasks.

        Parameters
        ----------
        subtask_ids
            ids of subtasks to cancel
        kill_timeout
            timeout seconds to kill actor process forcibly
        """
        await self._manager_ref.cancel_subtasks(subtask_ids, kill_timeout=kill_timeout)

    async def finish_subtasks(self, subtask_ids: List[str], schedule_next: bool = True):
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
    async def create(cls: Type[APIType], session_id: str, address: str) -> APIType:
        # from ..supervisor import AutoscalerActor
        # await mo.create_actor(
        #     AutoscalerActor, {}, uid=AutoscalerActor.default_uid(), address=address
        # )

        from .... import resource as mars_resource
        from ..worker import (
            SubtaskExecutionActor,
            SubtaskPrepareQueueActor,
            SubtaskExecutionQueueActor,
            WorkerQuotaManagerActor,
            SlotManagerActor,
        )

        await mo.create_actor(
            SlotManagerActor,
            uid=SlotManagerActor.default_uid(),
            address=address,
        )
        await mo.create_actor(
            SubtaskPrepareQueueActor,
            uid=SubtaskPrepareQueueActor.default_uid(),
            address=address,
        )
        await mo.create_actor(
            SubtaskExecutionQueueActor,
            uid=SubtaskExecutionQueueActor.default_uid(),
            address=address,
        )
        await mo.create_actor(
            SubtaskExecutionActor,
            subtask_max_retries=0,
            uid=SubtaskExecutionActor.default_uid(),
            address=address,
        )
        await mo.create_actor(
            WorkerQuotaManagerActor,
            {"quota_size": mars_resource.virtual_memory().total},
            uid=WorkerQuotaManagerActor.default_uid(),
            address=address,
        )

        from ..supervisor import SchedulingSupervisorService

        service = SchedulingSupervisorService({}, address)
        await service.create_session(session_id)
        return await super().create(session_id, address)
