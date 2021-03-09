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

import mars.oscar as mo
from mars.services.scheduling.worker import QuotaActor, MemQuotaActor
from mars.tests.core import mock
from mars.utils import get_next_port


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool(
        f'127.0.0.1:{get_next_port()}', n_process=0)
    await pool.start()
    yield pool
    await pool.stop()


@pytest.mark.asyncio
async def test_quota(actor_pool):
    quota_ref = await mo.create_actor(
        QuotaActor, 300, uid='QuotaActor', address=actor_pool.external_address)

    # test quota options with non-existing keys
    await quota_ref.process_quotas(['non_exist'])
    await quota_ref.hold_quotas(['non_exist'])
    await quota_ref.release_quotas(['non_exist'])

    with pytest.raises(ValueError):
        await quota_ref.request_quota('ERROR', 1000)

    # test quota request with immediate return
    await quota_ref.request_quota('0', 100)
    await quota_ref.request_quota('0', 50)
    await quota_ref.request_quota('0', 200)

    # test request with process_quota=True
    await quota_ref.request_quota('0', 200, process_quota=True)
    assert '0' in (await quota_ref.dump_data()).proc_sizes
    await quota_ref.alter_allocation('0', 190, new_key=('0', 0), process_quota=True)
    assert (await quota_ref.dump_data()).allocations[('0', 0)] == 190

    await quota_ref.hold_quotas([('0', 0)])
    assert ('0', 0) in (await quota_ref.dump_data()).hold_sizes
    await quota_ref.alter_allocation(('0', 0), new_key=('0', 1))
    assert (await quota_ref.dump_data()).allocations[('0', 1)] == 190

    req_task1 = asyncio.create_task(quota_ref.request_quota('1', 150))
    req_task2 = asyncio.create_task(quota_ref.request_quota('2', 50))
    asyncio.create_task(quota_ref.request_quota('3', 200))
    asyncio.create_task(quota_ref.request_quota('3', 180))

    await asyncio.sleep(0.1)
    assert '2' not in (await quota_ref.dump_data()).allocations

    req_task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await req_task1

    await asyncio.wait_for(req_task2, timeout=1)
    assert '1' not in (await quota_ref.dump_data()).allocations
    assert '2' in (await quota_ref.dump_data()).allocations
    assert '3' not in (await quota_ref.dump_data()).allocations

    await quota_ref.release_quotas([('0', 1)])
    assert '3' in (await quota_ref.dump_data()).allocations

    req_task4 = asyncio.create_task(quota_ref.request_quota('4', 180))
    await asyncio.sleep(0)
    assert '4' not in (await quota_ref.dump_data()).allocations

    await quota_ref.alter_allocations(['3'], [50])
    await req_task4
    assert '4' in (await quota_ref.dump_data()).allocations


@pytest.mark.asyncio
async def test_quota_allocation(actor_pool):
    quota_ref = await mo.create_actor(
        QuotaActor, 300, uid='QuotaActor', address=actor_pool.external_address)

    end_time = []
    finished = set()

    async def task_fun(x):
        await quota_ref.request_quota(x, 100)
        await asyncio.sleep(0.2)
        await quota_ref.release_quotas([x])
        end_time.append(time.time())
        finished.add(x)

    tasks = []
    for idx in range(5):
        tasks.append(asyncio.create_task(task_fun(idx)))
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

    assert abs(end_time[0] - end_time[1]) < 0.05
    assert abs(end_time[0] - end_time[2]) < 0.05
    assert abs(end_time[0] - end_time[3]) > 0.15
    assert abs(end_time[3] - end_time[4]) < 0.05
    assert await quota_ref.get_allocated_size() == 0


@pytest.mark.asyncio
async def test_batch_quota_allocation(actor_pool):
    quota_ref = await mo.create_actor(
        QuotaActor, 300, uid='QuotaActor', address=actor_pool.external_address)

    end_time = []

    async def task_fun(b):
        await quota_ref.request_batch_quota(b, process_quota=True)
        await asyncio.sleep(0.5)
        assert set(b.keys()) == set((await quota_ref.dump_data()).proc_sizes.keys())
        await quota_ref.release_quotas(list(b.keys()))
        end_time.append(time.time())

    tasks = []
    for idx in (0, 1):
        keys = [f'{idx}_0', f'{idx}_1']
        batch = dict((k, 100) for k in keys)
        tasks.append(asyncio.create_task(task_fun(batch)))
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

    assert abs(end_time[0] - end_time[1]) > 0.4
    assert await quota_ref.get_allocated_size() == 0


@pytest.mark.asyncio
async def test_mem_quota_allocation(actor_pool):
    from mars.utils import AttributeDict

    mock_mem_stat = AttributeDict(dict(total=300, available=50, used=0, free=50))
    quota_ref = await mo.create_actor(
        MemQuotaActor, 300, refresh_time=0.1, uid='MemQuotaActor',
        address=actor_pool.external_address)
    with mock.patch('mars.resource.virtual_memory', new=lambda: mock_mem_stat):
        time_recs = [time.time()]

        async def task_fun():
            await quota_ref.request_quota('req', 100)
            await quota_ref.release_quotas(['req'])
            time_recs.append(time.time())

        task = asyncio.create_task(task_fun())
        await asyncio.sleep(0.2)
        assert 'req' not in (await quota_ref.dump_data()).allocations

        mock_mem_stat['available'] = 150
        mock_mem_stat['free'] = 150
        await asyncio.wait_for(task, timeout=1)
        assert 0.15 < abs(time_recs[0] - time_recs[1]) < 1
