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
import functools
import time

from mars.tests.core import aio_case, patch_method, create_actor_pool
from mars.utils import get_next_port, build_exc_info
from mars.worker import QuotaActor, MemQuotaActor, DispatchActor, \
    ProcessHelperActor, StatusActor
from mars.worker.utils import WorkerClusterInfoActor
from mars.worker.tests.base import WorkerCase


@aio_case
class Test(WorkerCase):
    async def testQuota(self):
        async def _raiser(*_, **__):
            raise ValueError

        local_pool_addr = 'localhost:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=local_pool_addr) as pool:
            await pool.create_actor(WorkerClusterInfoActor, [local_pool_addr],
                                    uid=WorkerClusterInfoActor.default_uid())
            await pool.create_actor(StatusActor, local_pool_addr, uid=StatusActor.default_uid())

            quota_ref = await pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            # test quota options with non-existing keys
            await quota_ref.process_quotas(['non_exist'])
            await quota_ref.hold_quotas(['non_exist'])
            await quota_ref.release_quotas(['non_exist'])

            with self.assertRaises(ValueError):
                await quota_ref.request_quota('ERROR', 1000)

            # test quota request with immediate return
            self.assertTrue(await quota_ref.request_quota('0', 100))
            self.assertTrue(await quota_ref.request_quota('0', 50))
            self.assertTrue(await quota_ref.request_quota('0', 200))

            # test request with process_quota=True
            await quota_ref.request_quota('0', 200, process_quota=True)
            self.assertIn('0', (await quota_ref.dump_data()).proc_sizes)
            await quota_ref.alter_allocation('0', 190, new_key=('0', 0), process_quota=True)
            self.assertEqual((await quota_ref.dump_data()).allocations[('0', 0)], 190)

            await quota_ref.hold_quotas([('0', 0)])
            self.assertIn(('0', 0), (await quota_ref.dump_data()).hold_sizes)
            await quota_ref.alter_allocation(('0', 0), new_key=('0', 1))
            self.assertEqual((await quota_ref.dump_data()).allocations[('0', 1)], 190)

            async with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())

                future = asyncio.ensure_future(self.waitp(ref.request_quota('1', 150, _promise=True)))
                await asyncio.sleep(0.5)

                self.assertFalse(await quota_ref.request_quota('2', 50))
                self.assertFalse(await quota_ref.request_quota('3', 200))

                self.assertFalse(await quota_ref.request_quota('3', 180))

                self.assertNotIn('2', (await quota_ref.dump_data()).allocations)

                await ref.cancel_requests(('1',), reject_exc=build_exc_info(OSError))
                with self.assertRaises(OSError):
                    await future

                with patch_method(QuotaActor._request_quota, new=_raiser):
                    with self.assertRaises(ValueError):
                        await self.waitp(ref.request_quota('err_raise', 1, _promise=True))

                    ref.request_batch_quota({'err_raise': 1}, _promise=True) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                    with self.assertRaises(ValueError):
                        await self.waitp(ref.request_batch_quota({'err_raise': 1}, _promise=True))

                self.assertNotIn('1', (await quota_ref.dump_data()).requests)
                self.assertIn('2', (await quota_ref.dump_data()).allocations)
                self.assertNotIn('3', (await quota_ref.dump_data()).allocations)

            await quota_ref.release_quotas([('0', 1)])
            self.assertIn('3', (await quota_ref.dump_data()).allocations)

            self.assertFalse(await quota_ref.request_quota('4', 180))
            await quota_ref.alter_allocations(['3'], [50])
            self.assertIn('4', (await quota_ref.dump_data()).allocations)

            async with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())
                future = asyncio.ensure_future(self.waitp(ref.request_quota('5', 50, _promise=True)))

                with patch_method(QuotaActor.alter_allocation, new=_raiser):
                    await quota_ref.release_quotas(['2'])

                    with self.assertRaises(ValueError):
                        await future

    async def testQuotaAllocation(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=local_pool_addr) as pool:
            quota_ref = await pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            end_time = []
            finished = set()
            async with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())

                future = asyncio.Future()

                async def actual_exec(x):
                    await ref.release_quotas([x])
                    end_time.append(time.time())
                    finished.add(x)
                    if len(finished) == 5:
                        future.set_result(None)

                for idx in range(5):
                    x = str(idx)

                    ref.request_quota(x, 100, _promise=True) \
                        .then(functools.partial(test_actor.run_later, actual_exec, x, _delay=0.5))
                await asyncio.wait_for(future, timeout=10)

            self.assertLess(abs(end_time[0] - end_time[1]), 0.1)
            self.assertLess(abs(end_time[0] - end_time[2]), 0.1)
            self.assertGreater(abs(end_time[0] - end_time[3]), 0.4)
            self.assertLess(abs(end_time[3] - end_time[4]), 0.1)
            self.assertEqual(await quota_ref.get_allocated_size(), 0)

    async def testBatchQuotaAllocation(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=local_pool_addr) as pool:
            quota_ref = await pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            end_time = []

            async with self.run_actor_test(pool) as test_actor:
                future = asyncio.Future()

                for idx in (0, 1):
                    x = str(idx)
                    ref = test_actor.promise_ref(QuotaActor.default_uid())

                    async def actual_exec(b, set_result):
                        self.assertTrue(await ref.request_batch_quota(b, process_quota=True))
                        self.assertEqual(set(b.keys()), set((await quota_ref.dump_data()).proc_sizes.keys()))
                        await ref.release_quotas(list(b.keys()))
                        end_time.append(time.time())
                        if set_result:
                            future.set_result(None)

                    keys = [x + '_0', x + '_1']
                    batch = dict((k, 100) for k in keys)
                    ref.request_batch_quota(batch, _promise=True) \
                        .then(functools.partial(test_actor.run_later, actual_exec, batch,
                                                set_result=(idx == 1), _delay=0.5),
                              lambda *exc: future.set_exception(exc[1]))

                await future

            self.assertGreater(abs(end_time[0] - end_time[1]), 0.4)
            self.assertEqual(await quota_ref.get_allocated_size(), 0)

    async def testMemQuotaAllocation(self):
        from mars import resource
        from mars.utils import AttributeDict

        mock_mem_stat = AttributeDict(dict(total=300, available=50, used=0, free=50))
        local_pool_addr = 'localhost:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=local_pool_addr) as pool:
            with patch_method(resource.virtual_memory, new=lambda: mock_mem_stat):
                await pool.create_actor(WorkerClusterInfoActor, [local_pool_addr],
                                        uid=WorkerClusterInfoActor.default_uid())
                await pool.create_actor(StatusActor, local_pool_addr, uid=StatusActor.default_uid())

                await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
                await pool.create_actor(ProcessHelperActor, uid=ProcessHelperActor.default_uid())
                quota_ref = await pool.create_actor(MemQuotaActor, 300, refresh_time=0.1,
                                                    uid=MemQuotaActor.default_uid())

                time_recs = []
                async with self.run_actor_test(pool) as test_actor:
                    ref = test_actor.promise_ref(quota_ref)
                    time_recs.append(time.time())
                    future = asyncio.Future()

                    async def actual_exec(x):
                        await ref.release_quotas([x])
                        time_recs.append(time.time())
                        future.set_result(None)

                    ref.request_quota('req', 100, _promise=True) \
                        .then(functools.partial(actual_exec, 'req'))

                    await asyncio.sleep(0.5)
                    mock_mem_stat['available'] = 150
                    mock_mem_stat['free'] = 150

                    await asyncio.wait_for(future, timeout=10)

                self.assertGreater(abs(time_recs[0] - time_recs[1]), 0.4)
