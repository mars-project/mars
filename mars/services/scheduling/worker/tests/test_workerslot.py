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
import os
import sys
import time
from typing import Tuple

import pytest
import pandas as pd

import mars.oscar as mo
from mars.oscar import ServerClosed
from mars.oscar.backends.allocate_strategy import IdleLabel
from mars.services.scheduling.supervisor import GlobalSlotManagerActor
from mars.services.scheduling.worker import BandSlotManagerActor, \
    BandSlotControlActor
from mars.utils import get_next_port, extensible


class MockGlobalSlotManagerActor(mo.Actor):
    def __init__(self):
        self._result = None

    @extensible
    def update_subtask_slots(self, band: Tuple, session_id: str, subtask_id: str, slots: int):
        self._result = (band, session_id, subtask_id, slots)

    def get_result(self):
        return self._result


@pytest.fixture
async def actor_pool(request):
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    n_slots = request.param
    pool = await mo.create_actor_pool(
        f'127.0.0.1:{get_next_port()}', n_process=n_slots,
        labels=[None] + ['numa-0'] * n_slots,
        subprocess_start_method=start_method)

    async with pool:
        global_slots_ref = await mo.create_actor(
            MockGlobalSlotManagerActor, uid=GlobalSlotManagerActor.default_uid(),
            address=pool.external_address)
        slot_manager_ref = await mo.create_actor(
            BandSlotManagerActor,
            (pool.external_address, 'numa-0'), n_slots, global_slots_ref,
            uid=BandSlotManagerActor.gen_uid('numa-0'),
            address=pool.external_address)
        yield pool, slot_manager_ref


class TaskActor(mo.Actor):
    def __init__(self, call_logs):
        self._call_logs = call_logs
        self._dispatch_ref = None

    async def __post_create__(self):
        self._dispatch_ref = await mo.actor_ref(
            BandSlotManagerActor.gen_uid('numa-0'), address=self.address)
        await self._dispatch_ref.release_free_slot.tell(self.ref())

    async def queued_call(self, key, delay):
        try:
            self._call_logs[key] = time.time()
            await asyncio.sleep(delay)
        finally:
            await self._dispatch_ref.release_free_slot(self.ref())


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [0], indirect=True)
async def test_slot_assign(actor_pool):
    pool, slot_manager_ref = actor_pool

    call_logs = dict()
    group_size = 4
    delay = 0.5
    await asyncio.gather(*(
        mo.create_actor(TaskActor, call_logs, address=pool.external_address)
        for _ in range(group_size)
    ))
    assert len((await slot_manager_ref.dump_data()).free_slots) == group_size

    async def task_fun(idx):
        ref = await slot_manager_ref.acquire_free_slot('subtask_id')
        await ref.queued_call(idx, delay)

    tasks = []
    start_time = time.time()
    for idx in range(group_size + 1):
        tasks.append(asyncio.create_task(task_fun(idx)))
    await asyncio.gather(*tasks)

    log_series = pd.Series(call_logs).sort_index() - start_time
    assert len(log_series) == group_size + 1
    assert log_series.iloc[:group_size].max() < delay / 4
    assert log_series.iloc[group_size:].min() > delay / 4

    call_logs.clear()
    tasks = []
    start_time = time.time()
    for idx in range(group_size * 2 + 1):
        tasks.append(asyncio.create_task(task_fun(idx)))
    await asyncio.sleep(delay / 10)
    tasks[group_size].cancel()
    await asyncio.wait(tasks)

    with pytest.raises(asyncio.CancelledError):
        tasks[group_size].result()

    log_series = pd.Series(call_logs).sort_index() - start_time

    assert len(log_series) == group_size * 2
    assert log_series.iloc[:group_size].max() < delay / 4
    assert log_series.iloc[group_size:].min() > delay / 4


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [1], indirect=True)
async def test_slot_kill(actor_pool):
    pool, slot_manager_ref = actor_pool

    strategy = IdleLabel('numa-0', 'task_actor')
    task_ref = await mo.create_actor(TaskActor, {},
                                     allocate_strategy=strategy,
                                     address=pool.external_address)

    assert await mo.actor_ref(BandSlotControlActor.gen_uid('numa-0', 0),
                              address=pool.external_address)
    delayed_task = asyncio.create_task(task_ref.queued_call('key', 10))
    await asyncio.sleep(0.1)

    # check if process hosting the actor is closed
    kill_task = asyncio.create_task(slot_manager_ref.kill_slot(0))
    with pytest.raises(ServerClosed):
        await delayed_task

    # check if slot actor is restored
    await kill_task
    assert await mo.actor_ref(BandSlotControlActor.gen_uid('numa-0', 0),
                              address=pool.external_address)
    assert await mo.actor_ref(task_ref)


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [1], indirect=True)
async def test_report_usage(actor_pool):
    pool, slot_manager_ref = actor_pool

    await slot_manager_ref.acquire_free_slot(('session_id', 'subtask_id'))
    await asyncio.sleep(1.3)

    global_slot_ref = await mo.actor_ref(uid=GlobalSlotManagerActor.default_uid(),
                                         address=pool.external_address)
    _band, session_id, subtask_id, slots = await global_slot_ref.get_result()
    assert slots == pytest.approx(1.0)
    assert session_id == 'session_id'
    assert subtask_id == 'subtask_id'
