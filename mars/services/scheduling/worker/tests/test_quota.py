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

import pytest

from ..... import oscar as mo
from .....tests.core import mock
from .....utils import get_next_port
from ...worker import QuotaActor, MemQuotaActor, BandSlotManagerActor


class MockBandSlotManagerActor(mo.Actor):
    def get_restart_record(self):
        return getattr(self, "_restart_record", False)

    def restart_free_slots(self):
        self._restart_record = True


@pytest.fixture
async def actor_pool():
    start_method = (
        os.environ.get("POOL_START_METHOD", "fork") if sys.platform != "win32" else None
    )
    pool = await mo.create_actor_pool(
        f"127.0.0.1:{get_next_port()}",
        n_process=0,
        subprocess_start_method=start_method,
    )
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_quota(actor_pool):
    quota_ref = await mo.create_actor(
        QuotaActor,
        (actor_pool.external_address, "numa-0"),
        300,
        uid=QuotaActor.gen_uid("cpu-0"),
        address=actor_pool.external_address,
    )  # type: mo.ActorRefType[QuotaActor]

    # test quota options with non-existing keys
    await quota_ref.hold_quotas(["non_exist"])
    await quota_ref.release_quotas(["non_exist"])

    with pytest.raises(ValueError):
        await quota_ref.request_batch_quota({"ERROR": 1000})

    # test quota request with immediate return
    await quota_ref.request_batch_quota({"0": 100})
    await quota_ref.request_batch_quota({"0": 50})
    await quota_ref.request_batch_quota({"0": 200})

    # test request with process_quota=True
    await quota_ref.request_batch_quota({"0": 200})
    await quota_ref.alter_allocations(["0"], [190])
    assert (await quota_ref.dump_data()).allocations["0"] == 190

    await quota_ref.hold_quotas(["0"])
    assert "0" in (await quota_ref.dump_data()).hold_sizes

    req_task1 = asyncio.create_task(quota_ref.request_batch_quota({"1": 150}))
    req_task2 = asyncio.create_task(quota_ref.request_batch_quota({"2": 50}))
    asyncio.create_task(quota_ref.request_batch_quota({"3": 200}))
    asyncio.create_task(quota_ref.request_batch_quota({"3": 180}))

    await asyncio.sleep(0.1)
    assert "2" not in (await quota_ref.dump_data()).allocations

    req_task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await req_task1

    await asyncio.wait_for(req_task2, timeout=1)
    assert "1" not in (await quota_ref.dump_data()).allocations
    assert "2" in (await quota_ref.dump_data()).allocations
    assert "3" not in (await quota_ref.dump_data()).allocations

    await quota_ref.release_quotas(["0"])
    assert "3" in (await quota_ref.dump_data()).allocations

    req_task4 = asyncio.create_task(quota_ref.request_batch_quota({"4": 180}))
    await asyncio.sleep(0)
    assert "4" not in (await quota_ref.dump_data()).allocations

    await quota_ref.alter_allocations(["3"], [50])
    await req_task4
    assert "4" in (await quota_ref.dump_data()).allocations


@pytest.mark.asyncio
async def test_batch_quota_allocation(actor_pool):
    quota_ref = await mo.create_actor(
        QuotaActor,
        (actor_pool.external_address, "numa-0"),
        300,
        uid=QuotaActor.gen_uid("cpu-0"),
        address=actor_pool.external_address,
    )  # type: mo.ActorRefType[QuotaActor]

    end_time = []

    async def task_fun(b):
        await quota_ref.request_batch_quota(b)
        await asyncio.sleep(0.5)
        assert set(b.keys()) == set((await quota_ref.dump_data()).allocations.keys())
        await quota_ref.release_quotas(list(b.keys()))
        end_time.append(time.time())

    tasks = []
    for idx in (0, 1):
        keys = [f"{idx}_0", f"{idx}_1"]
        batch = dict((k, 100) for k in keys)
        tasks.append(asyncio.create_task(task_fun(batch)))
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

    assert abs(end_time[0] - end_time[1]) > 0.4
    assert await quota_ref.get_allocated_size() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_kill_slot", [False, True])
async def test_mem_quota_allocation(actor_pool, enable_kill_slot):
    from .....utils import AttributeDict

    mock_mem_stat = AttributeDict(dict(total=300, available=50, used=0, free=50))
    mock_band_slot_manager_ref = await mo.create_actor(
        MockBandSlotManagerActor,
        uid=BandSlotManagerActor.gen_uid("numa-0"),
        address=actor_pool.external_address,
    )
    quota_ref = await mo.create_actor(
        MemQuotaActor,
        (actor_pool.external_address, "numa-0"),
        300,
        hard_limit=300,
        refresh_time=0.1,
        enable_kill_slot=enable_kill_slot,
        uid=MemQuotaActor.gen_uid("cpu-0"),
        address=actor_pool.external_address,
    )  # type: mo.ActorRefType[QuotaActor]

    with mock.patch("mars.resource.virtual_memory", new=lambda: mock_mem_stat):
        time_recs = [time.time()]

        async def task_fun():
            await quota_ref.request_batch_quota({"req": 100})
            await quota_ref.release_quotas(["req"])
            time_recs.append(time.time())

        task = asyncio.create_task(task_fun())
        await asyncio.sleep(0.2)
        assert "req" not in (await quota_ref.dump_data()).allocations

        mock_mem_stat["available"] = 150
        mock_mem_stat["free"] = 150
        await asyncio.wait_for(task, timeout=1)
        assert 0.15 < abs(time_recs[0] - time_recs[1]) < 1
        assert (
            bool(await mock_band_slot_manager_ref.get_restart_record())
            == enable_kill_slot
        )
