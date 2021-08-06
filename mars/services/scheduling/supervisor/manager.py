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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....typing import BandType
from ....utils import dataslots
from ...subtask import Subtask, SubtaskResult, SubtaskStatus
from ...task import TaskAPI
from ..utils import redirect_subtask_errors

logger = logging.getLogger(__name__)


@dataslots
@dataclass
class SubtaskScheduleInfo:
    subtask: Subtask
    band_futures: Dict[BandType, asyncio.Future] = field(default_factory=dict)


class SubtaskManagerActor(mo.Actor):
    _subtask_infos: Dict[str, SubtaskScheduleInfo]

    @classmethod
    def gen_uid(cls, session_id: str):
        return f'{session_id}_subtask_manager'

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._subtask_infos = dict()

        self._queueing_ref = None
        self._global_slot_ref = None

    async def __post_create__(self):
        from .queueing import SubtaskQueueingActor
        self._queueing_ref = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(self._session_id), address=self.address)
        from ..supervisor import GlobalSlotManagerActor
        self._global_slot_ref = await mo.actor_ref(
            GlobalSlotManagerActor.default_uid(), address=self.address)

    @alru_cache
    async def _get_task_api(self):
        return await TaskAPI.create(self._session_id, self.address)

    async def add_subtasks(self, subtasks: List[Subtask], priorities: List[Tuple]):
        async with redirect_subtask_errors(self, subtasks):
            for subtask in subtasks:
                self._subtask_infos[subtask.subtask_id] = SubtaskScheduleInfo(subtask)

            virtual_subtasks = [subtask for subtask in subtasks if subtask.virtual]
            for subtask in virtual_subtasks:
                task_api = await self._get_task_api()
                await task_api.set_subtask_result(SubtaskResult(
                    subtask_id=subtask.subtask_id,
                    session_id=subtask.session_id,
                    task_id=subtask.task_id,
                    progress=1.0,
                    status=SubtaskStatus.succeeded))

            await self._queueing_ref.add_subtasks(
                [subtask for subtask in subtasks if not subtask.virtual], priorities)
            await self._queueing_ref.submit_subtasks.tell()

    @alru_cache(maxsize=10000)
    async def _get_execution_ref(self, band: BandType):
        from ..worker.execution import SubtaskExecutionActor
        return await mo.actor_ref(SubtaskExecutionActor.default_uid(), address=band[0])

    async def finish_subtasks(self, subtask_ids: List[str], schedule_next: bool = True):
        band_tasks = defaultdict(lambda: 0)
        for subtask_id in subtask_ids:
            subtask_info = self._subtask_infos.pop(subtask_id, None)
            if schedule_next and subtask_info is not None:
                for band in subtask_info.band_futures.keys():
                    band_tasks[band] += 1

        if band_tasks:
            coros = []
            for band, subtask_count in band_tasks.items():
                coros.append(self._queueing_ref.submit_subtasks.tell(band, subtask_count))
            await asyncio.wait(coros)

    def _get_subtasks_by_ids(self, subtask_ids: List[str]) -> List[Optional[Subtask]]:
        subtasks = []
        for stid in subtask_ids:
            try:
                subtasks.append(self._subtask_infos[stid].subtask)
            except KeyError:
                subtasks.append(None)
        return subtasks

    async def submit_subtask_to_band(self, subtask_id: str, band: BandType):
        async with redirect_subtask_errors(self, self._get_subtasks_by_ids([subtask_id])):
            subtask_info = self._subtask_infos[subtask_id]
            task = asyncio.create_task(self._await_subtask_result(subtask_info.subtask, band))
            subtask_info.band_futures[band] = task

    async def _await_subtask_result(self, subtask: Subtask, band: BandType):
        """To capture the run_subtask error, e.g. exception, process crash."""
        async with redirect_subtask_errors(self, [subtask]):
            execution_ref = await self._get_execution_ref(band)
            await execution_ref.run_subtask(
                subtask, band[1], self.address)

    async def cancel_subtasks(self, subtask_ids: List[str],
                              kill_timeout: Union[float, int] = 5):
        queued_subtask_ids = []
        single_cancel_tasks = []

        task_api = await self._get_task_api()

        async def cancel_single_task(subtask, raw_tasks, cancel_tasks):
            if cancel_tasks:
                await asyncio.wait(cancel_tasks)
            if raw_tasks:
                dones, _ = await asyncio.wait(raw_tasks)
            else:
                dones = []
            if not dones or all(fut.cancelled() for fut in dones):
                await task_api.set_subtask_result(SubtaskResult(
                    subtask_id=subtask.subtask_id,
                    session_id=subtask.session_id,
                    task_id=subtask.task_id,
                    status=SubtaskStatus.cancelled))

        for subtask_id in subtask_ids:
            if subtask_id not in self._subtask_infos:
                # subtask may already finished or not submitted at all
                continue

            subtask_info = self._subtask_infos[subtask_id]
            raw_tasks_to_cancel = list(subtask_info.band_futures.values())

            if not subtask_info.band_futures:
                queued_subtask_ids.append(subtask_id)
                single_cancel_tasks.append(asyncio.create_task(
                    cancel_single_task(subtask_info.subtask,
                                       raw_tasks_to_cancel, [])
                ))
            else:
                cancel_tasks = []
                for band in subtask_info.band_futures.keys():
                    execution_ref = await self._get_execution_ref(band)
                    cancel_tasks.append(asyncio.create_task(
                        execution_ref.cancel_subtask(subtask_id, kill_timeout=kill_timeout)
                    ))
                single_cancel_tasks.append(asyncio.create_task(
                    cancel_single_task(subtask_info.subtask,
                                       raw_tasks_to_cancel,
                                       cancel_tasks)
                ))
        if queued_subtask_ids:
            await self._queueing_ref.remove_queued_subtasks(queued_subtask_ids)
        if single_cancel_tasks:
            yield asyncio.wait(single_cancel_tasks)

        for subtask_id in subtask_ids:
            self._subtask_infos.pop(subtask_id, None)
