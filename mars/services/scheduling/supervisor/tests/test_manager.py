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
from typing import Dict, List, Optional, Tuple, Union

import pytest

from ..... import oscar as mo
from ....cluster import MockClusterAPI
from ....subtask import Subtask, SubtaskResult, SubtaskStatus
from ....task.supervisor.manager import TaskManagerActor
from ...supervisor import SubtaskManagerActor, AssignerActor
from ...worker import SubtaskExecutionActor


class MockAssignerActor(mo.Actor):
    async def assign_subtasks(self, subtasks: List[Subtask]):
        return [(self.address, "numa-0")] * len(subtasks)


class MockTaskManagerActor(mo.Actor):
    def __init__(self):
        self._results = dict()

    def set_subtask_result(self, result: SubtaskResult):
        self._results[result.subtask_id] = result

    def get_result(self, subtask_id: str) -> SubtaskResult:
        return self._results[subtask_id]


class MockSubtaskExecutionActor(mo.StatelessActor):
    _subtask_aiotasks: Dict[str, asyncio.Task]

    def __init__(self):
        self._subtask_caches = dict()
        self._subtasks = dict()
        self._subtask_aiotasks = dict()
        self._subtask_submit_events = dict()

    async def set_submit_subtask_event(self, subtask_id: str, event: asyncio.Event):
        self._subtask_submit_events[subtask_id] = event

    async def cache_subtasks(
        self,
        subtasks: List[Subtask],
        priorities: List[Tuple],
        supervisor_address: str,
        band_name: str,
    ):
        for subtask in subtasks:
            self._subtask_caches[subtask.subtask_id] = subtask

    async def submit_subtasks(
        self,
        subtasks: List[Union[str, Subtask]],
        priorities: List[Tuple],
        supervisor_address: str,
        band_name: str,
    ):
        for subtask in subtasks:
            if isinstance(subtask, str):
                subtask = self._subtask_caches[subtask]
            self._subtasks[subtask.subtask_id] = subtask
            self._subtask_aiotasks[subtask.subtask_id] = asyncio.create_task(
                asyncio.sleep(20)
            )
            self._subtask_submit_events[subtask.subtask_id].set()

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Optional[int] = 5
    ):
        for subtask_id in subtask_ids:
            self._subtask_aiotasks[subtask_id].cancel()

    async def wait_subtasks(self, subtask_ids: List[str]):
        yield asyncio.wait(
            [self._subtask_aiotasks[subtask_id] for subtask_id in subtask_ids]
        )
        results = []
        for subtask_id in subtask_ids:
            subtask = self._subtasks[subtask_id]
            aiotask = self._subtask_aiotasks[subtask_id]
            if not aiotask.done():
                subtask_kw = dict(status=SubtaskStatus.running)
            elif aiotask.cancelled():
                subtask_kw = dict(status=SubtaskStatus.cancelled)
            elif aiotask.exception() is not None:
                exc = aiotask.exception()
                tb = exc.__traceback__
                subtask_kw = dict(
                    status=SubtaskStatus.errored,
                    error=exc,
                    traceback=tb,
                )
            else:
                subtask_kw = dict(status=SubtaskStatus.succeeded)
            results.append(
                SubtaskResult(
                    subtask_id=subtask_id,
                    session_id=subtask.session_id,
                    task_id=subtask.task_id,
                    progress=1.0,
                    **subtask_kw
                )
            )
        raise mo.Return(results)


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        await MockClusterAPI.create(pool.external_address)
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
        assigner_ref = await mo.create_actor(
            MockAssignerActor,
            uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        manager_ref = await mo.create_actor(
            SubtaskManagerActor,
            session_id,
            uid=SubtaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )

        yield pool, session_id, execution_ref, manager_ref, task_manager_ref

        await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_subtask_manager(actor_pool):
    (
        pool,
        session_id,
        execution_ref,
        manager_ref,
        task_manager_ref,
    ) = actor_pool

    subtask1 = Subtask("subtask1", session_id)
    subtask2 = Subtask("subtask2", session_id)

    submit_subtask1_event = asyncio.Event()
    submit_subtask2_event = asyncio.Event()
    await execution_ref.set_submit_subtask_event(
        subtask1.subtask_id, submit_subtask1_event
    )
    await execution_ref.set_submit_subtask_event(
        subtask2.subtask_id, submit_subtask2_event
    )

    await manager_ref.cache_subtasks([subtask2], [(2,)])

    await manager_ref.add_subtasks([subtask1, subtask2], [(1,), (2,)])
    await asyncio.wait_for(
        asyncio.gather(submit_subtask1_event.wait(), submit_subtask2_event.wait()),
        timeout=10,
    )

    await manager_ref.cancel_subtasks([subtask1.subtask_id, subtask2.subtask_id])
    results = await asyncio.wait_for(
        execution_ref.wait_subtasks([subtask1.subtask_id, subtask2.subtask_id]),
        timeout=10,
    )
    assert results[0].status == SubtaskStatus.cancelled
    assert results[1].status == SubtaskStatus.cancelled
