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

import pytest
from typing import Tuple, List

from ..... import oscar as mo
from .....resource import Resource
from ....cluster import MockClusterAPI
from ....subtask import Subtask
from ...supervisor import (
    AssignerActor,
    SubtaskManagerActor,
    SubtaskQueueingActor,
    GlobalResourceManagerActor,
)


class MockSlotsActor(mo.Actor):
    def __init__(self):
        self._capacity = -1

    def set_capacity(self, capacity: int):
        self._capacity = capacity

    @mo.extensible
    def apply_subtask_resources(
        self,
        band: Tuple,
        session_id: str,
        subtask_ids: List[str],
        subtask_resources: List[Resource],
    ):
        idx = (
            min(self._capacity, len(subtask_ids))
            if self._capacity >= 0
            else len(subtask_ids)
        )
        return subtask_ids[:idx]


class MockAssignerActor(mo.Actor):
    def assign_subtasks(
        self, subtasks: List[Subtask], exclude_bands=None, random_when_unavailable=True
    ):
        return [(self.address, "numa-0")] * len(subtasks)


class MockSubtaskManagerActor(mo.Actor):
    def __init__(self):
        self._subtask_ids, self._bands = [], []

    @mo.extensible
    def submit_subtask_to_band(self, subtask_id: str, band: Tuple):
        self._subtask_ids.append(subtask_id)
        self._bands.append(band)

    def dump_data(self):
        return self._subtask_ids, self._bands


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        await MockClusterAPI.create(pool.external_address)

        # create assigner actor
        await mo.create_actor(
            MockAssignerActor,
            uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        # create queueing actor
        manager_ref = await mo.create_actor(
            MockSubtaskManagerActor,
            uid=SubtaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        # create slots actor
        slots_ref = await mo.create_actor(
            MockSlotsActor,
            uid=GlobalResourceManagerActor.default_uid(),
            address=pool.external_address,
        )
        # create queueing actor
        queueing_ref = await mo.create_actor(
            SubtaskQueueingActor,
            session_id,
            uid=SubtaskQueueingActor.gen_uid(session_id),
            address=pool.external_address,
        )
        try:
            yield pool, session_id, queueing_ref, slots_ref, manager_ref
        finally:
            await mo.destroy_actor(queueing_ref)
            await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_subtask_queueing(actor_pool):
    _pool, session_id, queueing_ref, slots_ref, manager_ref = actor_pool
    await slots_ref.set_capacity(2)

    subtasks = [Subtask(str(i)) for i in range(5)]
    priorities = [(i,) for i in range(5)]

    await queueing_ref.add_subtasks(subtasks, priorities)
    # queue: [4 3 2 1 0]
    assert await queueing_ref.all_bands_busy()
    await queueing_ref.submit_subtasks()
    # queue: [2 1 0]
    commited_subtask_ids, _commited_bands = await manager_ref.dump_data()
    assert commited_subtask_ids == ["4", "3"]

    await queueing_ref.remove_queued_subtasks(["1"])
    # queue: [2 0]
    await queueing_ref.update_subtask_priority.batch(
        queueing_ref.update_subtask_priority.delay("0", (3,)),
        queueing_ref.update_subtask_priority.delay("4", (5,)),
    )
    # queue: [0(3) 2]
    await queueing_ref.submit_subtasks()
    # queue: []
    commited_subtasks, _commited_bands = await manager_ref.dump_data()
    assert commited_subtasks == ["4", "3", "0", "2"]
    assert not await queueing_ref.all_bands_busy()
