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
from collections import defaultdict
from typing import List, Tuple, Set

import pytest

from ..... import oscar as mo
from .....typing import BandType
from ....cluster import MockClusterAPI
from ....subtask import Subtask, SubtaskResult, SubtaskStatus
from ....task.supervisor.manager import TaskManagerActor
from ...supervisor import (
    SubtaskQueueingActor,
    SubtaskManagerActor,
    GlobalResourceManagerActor,
)
from ...worker import SubtaskExecutionActor


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

    def add_subtasks(
        self,
        subtasks: List[Subtask],
        priorities: List[Tuple],
        exclude_bands: Set[Tuple] = None,
        random_when_unavailable: bool = True,
    ):
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


class MockSubtaskExecutionActor(mo.StatelessActor):
    def __init__(self):
        self._subtask_aiotasks = defaultdict(dict)
        self._run_subtask_events = {}

    async def set_run_subtask_event(self, subtask_id, event):
        self._run_subtask_events[subtask_id] = event

    async def run_subtask(
        self, subtask: Subtask, band_name: str, supervisor_address: str
    ):
        self._run_subtask_events[subtask.subtask_id].set()
        task = self._subtask_aiotasks[subtask.subtask_id][
            band_name
        ] = asyncio.create_task(asyncio.sleep(20))
        return await task

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
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        await MockClusterAPI.create(pool.external_address)
        queue_ref = await mo.create_actor(
            MockSubtaskQueueingActor,
            uid=SubtaskQueueingActor.gen_uid(session_id),
            address=pool.external_address,
        )
        slots_ref = await mo.create_actor(
            GlobalResourceManagerActor,
            uid=GlobalResourceManagerActor.default_uid(),
            address=pool.external_address,
        )
        task_manager_ref = await mo.create_actor(
            MockTaskManagerActor,
            uid=TaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        execution_ref = await mo.create_actor(
            MockSubtaskExecutionActor,
            uid=SubtaskExecutionActor.default_uid(),
            address=pool.external_address,
        )
        submitter_ref = await mo.create_actor(
            SubtaskManagerActor,
            session_id,
            uid=SubtaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )

        try:
            yield pool, session_id, execution_ref, submitter_ref, queue_ref, task_manager_ref
        finally:
            await mo.destroy_actor(slots_ref)
            await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_subtask_manager(actor_pool):
    (
        pool,
        session_id,
        execution_ref,
        manager_ref,
        queue_ref,
        task_manager_ref,
    ) = actor_pool

    subtask1 = Subtask("subtask1", session_id)
    subtask2 = Subtask("subtask2", session_id)

    await manager_ref.add_subtasks([subtask1, subtask2], [(1,), (2,)])
    run_subtask1_event, run_subtask2_event = asyncio.Event(), asyncio.Event()
    await execution_ref.set_run_subtask_event(subtask1.subtask_id, run_subtask1_event)
    await execution_ref.set_run_subtask_event(subtask2.subtask_id, run_subtask2_event)

    submit1 = asyncio.create_task(
        manager_ref.submit_subtask_to_band(
            subtask1.subtask_id, (pool.external_address, "gpu-0")
        )
    )
    submit2 = asyncio.create_task(
        manager_ref.submit_subtask_to_band(
            subtask2.subtask_id, (pool.external_address, "gpu-1")
        )
    )

    await asyncio.gather(run_subtask1_event.wait(), run_subtask2_event.wait())

    await manager_ref.cancel_subtasks([subtask1.subtask_id, subtask2.subtask_id])
    await asyncio.wait_for(
        asyncio.gather(
            execution_ref.wait_subtask(subtask1.subtask_id, "gpu-0"),
            execution_ref.wait_subtask(subtask2.subtask_id, "gpu-1"),
        ),
        timeout=10,
    )
    with pytest.raises(asyncio.CancelledError):
        await submit1
    with pytest.raises(asyncio.CancelledError):
        await submit2
    assert (
        await task_manager_ref.get_result(subtask1.subtask_id)
    ).status == SubtaskStatus.cancelled
    assert (
        await task_manager_ref.get_result(subtask2.subtask_id)
    ).status == SubtaskStatus.cancelled

    subtask3 = Subtask("subtask3", session_id)

    await queue_ref.set_error(ValueError())
    await manager_ref.add_subtasks.tell([subtask3], [(3,)])
    await asyncio.sleep(0.1)
    subtask3_result = await task_manager_ref.get_result(subtask3.subtask_id)
    assert subtask3_result.status == SubtaskStatus.errored
    assert isinstance(subtask3_result.error, ValueError)
