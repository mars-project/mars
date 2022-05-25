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

import psutil
import pytest
import pandas as pd

from ..... import oscar as mo
from .....oscar import ServerClosed
from .....oscar.errors import NoFreeSlot, SlotStateError
from .....oscar.backends.allocate_strategy import IdleLabel
from .....resource import Resource
from .....tests.core import wait_for_condition
from .....utils import get_next_port
from ...supervisor import GlobalResourceManagerActor
from ...worker import BandSlotManagerActor, BandSlotControlActor


class MockGlobalResourceManagerActor(mo.Actor):
    def __init__(self):
        self._result = None

    @mo.extensible
    def update_subtask_resources(
        self, band: Tuple, session_id: str, subtask_id: str, resources: Resource
    ):
        self._result = (band, session_id, subtask_id, resources)

    def get_result(self):
        return self._result


@pytest.fixture
async def actor_pool(request):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    n_slots = request.param
    pool = await mo.create_actor_pool(
        f"127.0.0.1:{get_next_port()}",
        n_process=n_slots,
        labels=[None] + ["numa-0"] * n_slots,
        subprocess_start_method=start_method,
    )

    async with pool:
        global_resource_ref = await mo.create_actor(
            MockGlobalResourceManagerActor,
            uid=GlobalResourceManagerActor.default_uid(),
            address=pool.external_address,
        )
        slot_manager_ref = await mo.create_actor(
            BandSlotManagerActor,
            (pool.external_address, "numa-0"),
            n_slots,
            global_resource_ref,
            uid=BandSlotManagerActor.gen_uid("numa-0"),
            address=pool.external_address,
        )
        try:
            yield pool, slot_manager_ref
        finally:
            await slot_manager_ref.destroy()


ActorPoolType = Tuple[mo.MainActorPoolType, mo.ActorRefType[BandSlotManagerActor]]


class TaskActor(mo.Actor):
    def __init__(self, call_logs, slot_id=0):
        self._call_logs = call_logs
        self._dispatch_ref = None
        self._slot_id = slot_id

    @classmethod
    def gen_uid(cls, slot_id):
        return f"{slot_id}_task_actor"

    async def __post_create__(self):
        self._dispatch_ref = await mo.actor_ref(
            BandSlotManagerActor.gen_uid("numa-0"), address=self.address
        )
        await self._dispatch_ref.register_slot.tell(self._slot_id, os.getpid())

    async def queued_call(self, key, session_stid, delay):
        try:
            self._call_logs[key] = time.time()
            await asyncio.sleep(delay)
        finally:
            if session_stid is not None:
                await self._dispatch_ref.release_free_slot(self._slot_id, session_stid)

    def get_call_logs(self):
        return self._call_logs


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [0], indirect=True)
async def test_slot_assign(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool

    call_logs = dict()
    group_size = 4
    delay = 1
    await asyncio.gather(
        *(
            mo.create_actor(
                TaskActor,
                call_logs,
                slot_id=slot_id,
                uid=TaskActor.gen_uid(slot_id),
                address=pool.external_address,
            )
            for slot_id in range(group_size)
        )
    )
    assert len((await slot_manager_ref.dump_data()).free_slots) == group_size

    async def task_fun(idx):
        session_stid = ("session_id", f"subtask_id{idx}")
        slot_id = await slot_manager_ref.acquire_free_slot(session_stid)
        assert slot_id == await slot_manager_ref.get_subtask_slot(session_stid)
        ref = await mo.actor_ref(
            uid=TaskActor.gen_uid(slot_id), address=pool.external_address
        )
        await ref.queued_call(idx, session_stid, delay)

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
@pytest.mark.parametrize("actor_pool", [1], indirect=True)
async def test_slot_kill(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool

    strategy = IdleLabel("numa-0", "task_actor")
    task_ref = await mo.create_actor(
        TaskActor, {}, allocate_strategy=strategy, address=pool.external_address
    )

    assert await mo.actor_ref(
        BandSlotControlActor.gen_uid("numa-0", 0), address=pool.external_address
    )
    delayed_task = asyncio.create_task(task_ref.queued_call("key", None, 10))
    await asyncio.sleep(0.1)

    # check if process hosting the actor is closed
    kill_task = asyncio.create_task(slot_manager_ref.kill_slot(0))
    await asyncio.sleep(0)
    kill_task2 = asyncio.create_task(slot_manager_ref.kill_slot(0))

    with pytest.raises(ServerClosed):
        await delayed_task

    # check if slot actor is restored
    await kill_task
    # check if secondary task makes no change
    await kill_task2

    assert await mo.actor_ref(
        BandSlotControlActor.gen_uid("numa-0", 0), address=pool.external_address
    )

    async def check_alive():
        assert await mo.actor_ref(task_ref)
        return True

    await wait_for_condition(check_alive)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [3], indirect=True)
async def test_slot_restart(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool

    strategy = IdleLabel("numa-0", "task_actor")
    task_refs = []
    for idx in range(3):
        ref = await mo.create_actor(
            TaskActor,
            {},
            slot_id=idx,
            allocate_strategy=strategy,
            address=pool.external_address,
        )
        await ref.queued_call("idx", None, idx)
        task_refs.append(ref)

    await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id1"))
    slot_id2 = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id2"))
    await slot_manager_ref.release_free_slot(slot_id2, ("session_id", "subtask_id2"))

    async def record_finish_time(coro):
        await coro
        return time.time()

    restart_task1 = asyncio.create_task(
        record_finish_time(slot_manager_ref.restart_free_slots())
    )
    await asyncio.sleep(0)
    restart_task2 = asyncio.create_task(
        record_finish_time(slot_manager_ref.restart_free_slots())
    )
    acquire_task = asyncio.create_task(
        record_finish_time(
            slot_manager_ref.acquire_free_slot(("session_id", "subtask_id3"))
        )
    )

    await asyncio.gather(restart_task1, restart_task2, acquire_task)

    # check only slots with running records are restarted
    assert len(await task_refs[0].get_call_logs()) > 0
    assert len(await task_refs[1].get_call_logs()) == 0
    assert len(await task_refs[2].get_call_logs()) > 0

    assert abs(restart_task1.result() - acquire_task.result()) < 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [1], indirect=True)
async def test_report_usage(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool

    await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id"))
    await asyncio.sleep(1.3)

    global_resource_ref = await mo.actor_ref(
        uid=GlobalResourceManagerActor.default_uid(), address=pool.external_address
    )
    _band, session_id, subtask_id, resources = await global_resource_ref.get_result()
    assert resources.num_cpus == pytest.approx(1.0)
    assert session_id == "session_id"
    assert subtask_id == "subtask_id"


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [1], indirect=True)
async def test_slot_fault_tolerance(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool
    # acquire -> slot restarted = can't acquire more.
    slot_id = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id"))
    await slot_manager_ref.register_slot(slot_id, os.getpid())
    with pytest.raises(NoFreeSlot):
        await slot_manager_ref.acquire_free_slot(
            ("session_id", "subtask_id"), block=False
        )
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id"))

    # acquire -> release -> slot restarted = can only acquire once.
    slot_id = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id2"))
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id2"))
    await slot_manager_ref.register_slot(slot_id, os.getpid())
    await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id2"))
    with pytest.raises(NoFreeSlot):
        await slot_manager_ref.acquire_free_slot(
            ("session_id", "subtask_id2"), block=False
        )
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id2"))

    # acquire -> release -> acquire -> slot restarted = can't acquire more.
    slot_id = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id3"))
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id3"))
    await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id3"))
    await slot_manager_ref.register_slot(slot_id, os.getpid())
    with pytest.raises(NoFreeSlot):
        await slot_manager_ref.acquire_free_slot(
            ("session_id", "subtask_id3"), block=False
        )
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id3"))


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [1], indirect=True)
async def test_slot_exception(actor_pool: ActorPoolType):
    pool, slot_manager_ref = actor_pool

    # make sure the BandSlotControlActor has registered.
    slot_id = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id"))
    await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id"))

    if sys.platform == "win32":
        with pytest.raises(ValueError):
            await slot_manager_ref.register_slot(1, -1)
    else:
        with pytest.raises((psutil.AccessDenied, psutil.NoSuchProcess)):
            await slot_manager_ref.register_slot(1, 0)

    dump_data = await slot_manager_ref.dump_data()
    # after the register_slot is correctly handled,
    # we can assert 1 not in free slots.
    assert 1 in dump_data.free_slots

    slot_id = await slot_manager_ref.acquire_free_slot(("session_id", "subtask_id"))
    with pytest.raises(SlotStateError):
        # release session_stid not matched the acquired value.
        await slot_manager_ref.release_free_slot(slot_id, ("session_id", "subtask_id1"))

    dump_data = await slot_manager_ref.dump_data()
    # the slot is not released.
    assert slot_id not in dump_data.free_slots

    not_acquired_slot = next(iter(dump_data.free_slots))
    with pytest.raises(SlotStateError):
        await slot_manager_ref.release_free_slot(
            not_acquired_slot, ("session_id", "subtask_id1")
        )
