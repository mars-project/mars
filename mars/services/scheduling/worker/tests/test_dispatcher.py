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
import time

import pytest
import pandas as pd

import mars.oscar as mo
from mars.services.scheduling.worker import DispatchActor
from mars.utils import get_next_port


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool(
        f'127.0.0.1:{get_next_port()}', n_process=0)
    await pool.start()
    yield pool
    await pool.stop()


class TaskActor(mo.Actor):
    def __init__(self, call_logs):
        self._call_logs = call_logs
        self._dispatch_ref = None

    async def __post_create__(self):
        self._dispatch_ref = await mo.actor_ref(
            'DispatchActor', address=self.address)
        await self._dispatch_ref.release_free_slot(self.ref())

    async def queued_call(self, key, delay):
        try:
            self._call_logs[key] = time.time()
            await asyncio.sleep(delay)
        finally:
            await self._dispatch_ref.release_free_slot(self.ref())


@pytest.mark.asyncio
async def test_dispatch(actor_pool):
    call_logs = dict()
    group_size = 4
    delay = 0.5

    dispatch_ref = await mo.create_actor(
        DispatchActor, uid='DispatchActor', address=actor_pool.external_address)
    await asyncio.gather(*(
        mo.create_actor(TaskActor, call_logs, address=actor_pool.external_address)
        for _ in range(group_size)
    ))
    assert len((await dispatch_ref.dump_data()).free_slots) == group_size

    async def task_fun(idx):
        ref = await dispatch_ref.acquire_free_slot()
        await ref.queued_call(idx, delay)

    tasks = []
    start_time = time.time()
    for idx in range(group_size + 1):
        tasks.append(asyncio.create_task(task_fun(idx)))
    await asyncio.wait(tasks)

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

    log_series = pd.Series(call_logs).sort_index() - start_time

    assert len(log_series) == group_size * 2
    assert log_series.iloc[:group_size].max() < delay / 4
    assert log_series.iloc[group_size:].min() > delay / 4
