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
from collections import defaultdict
from typing import List, Tuple

import pytest

import mars.oscar as mo
from mars.services.core import BandType
from mars.services.cluster import MockClusterAPI
from mars.services.scheduling.supervisor import SubtaskQueueingActor, \
    SubtaskManagerActor, GlobalSlotManagerActor
from mars.services.scheduling.worker import SubtaskExecutionActor
from mars.services.subtask import Subtask, SubtaskResult, SubtaskStatus
from mars.services.task.supervisor.manager import TaskManagerActor


class MockTaskManagerActor(mo.Actor):
    def __init__(self):
        self._results = dict()

    def set_subtask_result(self, result: SubtaskResult):
        self._results[result.subtask_id] = result

    def get_result(self, subtask_id: str) -> SubtaskResult:
        return self._results[subtask_id]


class MockSubtaskQueueingActor(mo.Actor):
    def __init__(self):
        self._subtasks = dict()
        self._error = None

    def add_subtasks(self, subtasks: List[Subtask], priorities: List[Tuple]):
        if self._error is not None:
            raise self._error
        for subtask, priority in zip(subtasks, priorities):
            self._subtasks[subtask.subtask_id] = (subtask, priority)

    def submit_subtasks(self, band: BandType, limit: int):
        pass

    def remove_queued_subtasks(self, subtask_ids: List[str]):
        for stid in subtask_ids:
            self._subtasks.pop(stid)

    def set_error(self, error):
        self._error = error


class MockSubtaskExecutionActor(mo.Actor):
    def __init__(self):
        self._subtask_aiotasks = defaultdict(dict)

    async def run_subtask(self, subtask: Subtask, band_name: str, supervisor_address: str):
        task = self._subtask_aiotasks[subtask.subtask_id][band_name] = \
            asyncio.create_task(asyncio.sleep(20))
        return task

    def cancel_subtask(self, subtask_id: str, kill_timeout: int = 5):
        for task in self._subtask_aiotasks[subtask_id].values():
            task.cancel()

    async def wait_subtask(self, subtask_id: str, band_name: str):
        try:
            yield self._subtask_aiotasks[subtask_id][band_name]
        except asyncio.CancelledError:
            pass


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        session_id = 'test_session'
        await MockClusterAPI.create(pool.external_address)
        queue_ref = await mo.create_actor(
            MockSubtaskQueueingActor, uid=SubtaskQueueingActor.gen_uid(session_id),
            address=pool.external_address)
        slots_ref = await mo.create_actor(
            GlobalSlotManagerActor, uid=GlobalSlotManagerActor.default_uid(),
            address=pool.external_address)
        task_manager_ref = await mo.create_actor(
            MockTaskManagerActor, uid=TaskManagerActor.gen_uid(session_id),
            address=pool.external_address)
        execution_ref = await mo.create_actor(
            MockSubtaskExecutionActor,
            uid=SubtaskExecutionActor.default_uid(),
            address=pool.external_address)
        submitter_ref = await mo.create_actor(
            SubtaskManagerActor, session_id, uid=SubtaskManagerActor.gen_uid(session_id),
            address=pool.external_address)

        yield pool, session_id, execution_ref, submitter_ref, queue_ref, task_manager_ref

        await mo.destroy_actor(slots_ref)


@pytest.mark.asyncio
async def test_subtask_manager(actor_pool):
    pool, session_id, execution_ref, manager_ref, queue_ref, task_manager_ref = actor_pool

    subtask1 = Subtask('subtask1', session_id)
    subtask2 = Subtask('subtask2', session_id)

    await manager_ref.add_subtasks([subtask1, subtask2], [(1,), (2,)])
    await manager_ref.submit_subtask_to_band(
        subtask1.subtask_id, (pool.external_address, 'gpu-0'))
    await manager_ref.submit_subtask_to_band(
        subtask1.subtask_id, (pool.external_address, 'gpu-1'))

    await manager_ref.cancel_subtasks([subtask1.subtask_id, subtask2.subtask_id])
    await asyncio.wait_for(
        asyncio.gather(
            execution_ref.wait_subtask(subtask1.subtask_id, 'gpu-0'),
            execution_ref.wait_subtask(subtask1.subtask_id, 'gpu-1'),
        ), timeout=10)
    assert (await task_manager_ref.get_result(subtask1.subtask_id)).status \
           == SubtaskStatus.cancelled
    assert (await task_manager_ref.get_result(subtask2.subtask_id)).status \
           == SubtaskStatus.cancelled

    subtask3 = Subtask('subtask3', session_id)

    await queue_ref.set_error(ValueError())
    await manager_ref.add_subtasks.tell([subtask3], [(3,)])
    await asyncio.sleep(0.1)
    subtask3_result = await task_manager_ref.get_result(subtask3.subtask_id)
    assert subtask3_result.status == SubtaskStatus.errored
    assert isinstance(subtask3_result.error, ValueError)
