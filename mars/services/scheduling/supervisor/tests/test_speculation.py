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
from typing import List, Tuple, Set

import pytest

from ..... import oscar as mo
from ....cluster import MockClusterAPI
from ....subtask import Subtask
from ...errors import NoAvailableBand
from ...supervisor import GlobalResourceManagerActor
from ..manager import SubtaskScheduleInfo
from ..speculation import SpeculativeScheduler


class MockSubtaskQueueingActor(mo.Actor):
    def __init__(self):
        self._subtasks = []
        self._exceptions = []

    async def add_subtasks(
        self,
        subtasks: List[Subtask],
        priorities: List[Tuple],
        exclude_bands: Set[Tuple] = None,
        random_when_unavailable: bool = True,
    ):
        if {
            ("addr0", "numa-0"),
            ("addr1", "numa-0"),
            ("addr2", "numa-0"),
        } - exclude_bands == set():
            self._exceptions.append(NoAvailableBand())
            raise self._exceptions[-1]
        self._subtasks.extend(subtasks)

    async def get_subtasks(self):
        return self._subtasks

    async def get_exceptions(self):
        return self._exceptions


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        cluster_api = await MockClusterAPI.create(pool.external_address)
        slots_ref = await mo.create_actor(
            GlobalResourceManagerActor,
            uid=GlobalResourceManagerActor.default_uid(),
            address=pool.external_address,
        )
        queue_ref = await mo.create_actor(
            MockSubtaskQueueingActor,
            address=pool.external_address,
        )
        try:
            yield pool, cluster_api, session_id, slots_ref, queue_ref
        finally:
            await mo.destroy_actor(queue_ref)
            await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_speculation(actor_pool):
    pool, cluster_api, session_id, slots_ref, queue_ref = actor_pool
    speculation_conf = {
        "enabled": True,
        "interval": 1000,
        "threshold": 0.2,
        "min_task_runtime": 0.01,
        "multiplier": 1.5,
        "max_concurrent_run": 2,
    }
    speculative_scheduler = SpeculativeScheduler(queue_ref, slots_ref, speculation_conf)
    await speculative_scheduler.start()
    await speculative_scheduler._speculative_execution()
    total_subtasks = 5
    subtasks = [
        Subtask(str(i), retryable=False, logic_key=f"logic_key1", logic_parallelism=5)
        for i in range(total_subtasks)
    ]
    subtask_infos = [
        SubtaskScheduleInfo(subtask, max_reschedules=3) for subtask in subtasks
    ]
    # add unfinished subtasks
    for subtask_info in subtask_infos:
        speculative_scheduler.add_subtask(subtask_info)
    await speculative_scheduler._speculative_execution()
    assert len(speculative_scheduler._grouped_finished_subtasks.values()) == 0
    # finished some subtasks
    for subtask_info in subtask_infos[:-1]:
        speculative_scheduler.finish_subtask(subtask_info)
    assert (
        len(next(iter(speculative_scheduler._grouped_finished_subtasks.values())))
        == total_subtasks - 1
    )
    assert (
        len(next(iter(speculative_scheduler._grouped_unfinished_subtasks.values())))
        == 1
    )
    await speculative_scheduler._speculative_execution()
    subtask_infos[-1].subtask.retryable = True
    # pretend subtask has been running on a band.
    subtask_infos[-1].band_futures[("addr0", "numa-0")] = asyncio.ensure_future(
        asyncio.sleep(1)
    )
    await speculative_scheduler._speculative_execution()
    submitted = await queue_ref.get_subtasks()
    # assert stale subtasks resubmitted
    assert subtask_infos[-1].subtask in submitted
    await speculative_scheduler._speculative_execution()
    # if resubmitted subtasks not running, don't resubmitted again.
    assert 1 == len(await queue_ref.get_subtasks())
    # pretend subtask has been running on a band.
    subtask_infos[-1].band_futures[("addr1", "numa-0")] = asyncio.ensure_future(
        asyncio.sleep(1)
    )
    await speculative_scheduler._speculative_execution()
    # stale subtasks resubmitted again
    assert 2 == len(await queue_ref.get_subtasks())
    # pretend subtask has been running on another band.
    subtask_infos[-1].band_futures[("addr2", "numa-0")] = asyncio.ensure_future(
        asyncio.sleep(1)
    )
    # speculative run reached max limit `max_concurrent_run`, i.e. 2
    await speculative_scheduler._speculative_execution()
    # assert raise queue_ref raise NoAvailableBand
    speculative_scheduler._subtask_speculation_max_concurrent_run += 1
    await speculative_scheduler._speculative_execution()
    assert isinstance((await queue_ref.get_exceptions())[0], NoAvailableBand)
    # finish subtasks
    speculative_scheduler.finish_subtask(subtask_infos[-1])
    assert len(speculative_scheduler._grouped_unfinished_subtasks) == 0
    await speculative_scheduler.stop()
