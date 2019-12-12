# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import functools
import time

from mars.tests.core import patch_method, create_actor_pool
from mars.utils import get_next_port, build_exc_info
from mars.worker import QuotaActor, MemQuotaActor, DispatchActor, \
    ProcessHelperActor, StatusActor
from mars.worker.utils import WorkerClusterInfoActor
from mars.worker.tests.base import WorkerCase


class Test(WorkerCase):
    def testQuota(self):
        def _raiser(*_, **__):
            raise ValueError

        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=local_pool_addr) as pool:
            pool.create_actor(WorkerClusterInfoActor, schedulers=[local_pool_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, local_pool_addr, uid=StatusActor.default_uid())

            quota_ref = pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            # test quota options with non-existing keys
            quota_ref.process_quota('non_exist')
            quota_ref.hold_quota('non_exist')
            quota_ref.release_quota('non_exist')

            with self.assertRaises(ValueError):
                quota_ref.request_quota('ERROR', 1000)

            # test quota request with immediate return
            self.assertTrue(quota_ref.request_quota('0', 100))
            self.assertTrue(quota_ref.request_quota('0', 50))
            self.assertTrue(quota_ref.request_quota('0', 200))

            # test request with process_quota=True
            quota_ref.request_quota('0', 200, process_quota=True)
            self.assertIn('0', quota_ref.dump_data().proc_sizes)
            quota_ref.alter_allocation('0', 190, new_key=('0', 0), process_quota=True)
            self.assertEqual(quota_ref.dump_data().allocations[('0', 0)], 190)

            quota_ref.hold_quota(('0', 0))
            self.assertIn(('0', 0), quota_ref.dump_data().hold_sizes)
            quota_ref.alter_allocation(('0', 0), new_key=('0', 1))
            self.assertEqual(quota_ref.dump_data().allocations[('0', 1)], 190)

            with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())

                ref.request_quota('1', 150, _promise=True) \
                    .then(lambda *_: test_actor.set_result(True)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                pool.sleep(0.5)

                self.assertFalse(quota_ref.request_quota('2', 50))
                self.assertFalse(quota_ref.request_quota('3', 200))

                self.assertFalse(quota_ref.request_quota('3', 180))

                self.assertNotIn('2', quota_ref.dump_data().allocations)

                ref.cancel_requests(('1',), reject_exc=build_exc_info(OSError))
                with self.assertRaises(OSError):
                    self.get_result(5)

                with patch_method(QuotaActor._request_quota, new=_raiser):
                    ref.request_quota('err_raise', 1, _promise=True) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                    with self.assertRaises(ValueError):
                        self.get_result(5)

                    ref.request_batch_quota({'err_raise': 1}, _promise=True) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                    with self.assertRaises(ValueError):
                        self.get_result(5)

                self.assertNotIn('1', quota_ref.dump_data().requests)
                self.assertIn('2', quota_ref.dump_data().allocations)
                self.assertNotIn('3', quota_ref.dump_data().allocations)

            quota_ref.release_quotas([('0', 1)])
            self.assertIn('3', quota_ref.dump_data().allocations)

            self.assertFalse(quota_ref.request_quota('4', 180))
            quota_ref.alter_allocations(['3'], [50])
            self.assertIn('4', quota_ref.dump_data().allocations)

            with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())
                ref.request_quota('5', 50, _promise=True) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                with patch_method(QuotaActor.alter_allocation, new=_raiser):
                    quota_ref.release_quota('2')

                    with self.assertRaises(ValueError):
                        self.get_result(5)

    def testQuotaAllocation(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=local_pool_addr) as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            end_time = []
            finished = set()
            with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(QuotaActor.default_uid())

                def actual_exec(x):
                    ref.release_quota(x)
                    end_time.append(time.time())
                    finished.add(x)
                    if len(finished) == 5:
                        test_actor.set_result(None)

                for idx in range(5):
                    x = str(idx)

                    ref.request_quota(x, 100, _promise=True) \
                        .then(functools.partial(test_actor.run_later, actual_exec, x, _delay=0.5))
                self.get_result(10)

            self.assertLess(abs(end_time[0] - end_time[1]), 0.1)
            self.assertLess(abs(end_time[0] - end_time[2]), 0.1)
            self.assertGreater(abs(end_time[0] - end_time[3]), 0.4)
            self.assertLess(abs(end_time[3] - end_time[4]), 0.1)
            self.assertEqual(quota_ref.get_allocated_size(), 0)

    def testBatchQuotaAllocation(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=local_pool_addr) as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid=QuotaActor.default_uid())

            end_time = []

            with self.run_actor_test(pool) as test_actor:
                for idx in (0, 1):
                    x = str(idx)
                    ref = test_actor.promise_ref(QuotaActor.default_uid())

                    def actual_exec(b, set_result):
                        self.assertTrue(ref.request_batch_quota(b, process_quota=True))
                        self.assertEqual(set(b.keys()), set(quota_ref.dump_data().proc_sizes.keys()))
                        for k in b.keys():
                            ref.release_quota(k)
                        end_time.append(time.time())
                        if set_result:
                            test_actor.set_result(None)

                    keys = [x + '_0', x + '_1']
                    batch = dict((k, 100) for k in keys)
                    ref.request_batch_quota(batch, _promise=True) \
                        .then(functools.partial(test_actor.run_later, actual_exec, batch,
                                                set_result=(idx == 1), _delay=0.5),
                              lambda *exc: test_actor.set_result(exc, accept=False))

                self.get_result(10)

            self.assertGreater(abs(end_time[0] - end_time[1]), 0.4)
            self.assertEqual(quota_ref.get_allocated_size(), 0)

    def testMemQuotaAllocation(self):
        from mars import resource
        from mars.utils import AttributeDict

        mock_mem_stat = AttributeDict(dict(total=300, available=50, used=0, free=50))
        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=local_pool_addr) as pool, \
                patch_method(resource.virtual_memory, new=lambda: mock_mem_stat):
            pool.create_actor(WorkerClusterInfoActor, schedulers=[local_pool_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, local_pool_addr, uid=StatusActor.default_uid())

            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            pool.create_actor(ProcessHelperActor, uid=ProcessHelperActor.default_uid())
            quota_ref = pool.create_actor(MemQuotaActor, 300, refresh_time=0.1,
                                          uid=MemQuotaActor.default_uid())

            time_recs = []
            with self.run_actor_test(pool) as test_actor:
                ref = test_actor.promise_ref(quota_ref)
                time_recs.append(time.time())

                def actual_exec(x):
                    ref.release_quota(x)
                    time_recs.append(time.time())
                    test_actor.set_result(None)

                ref.request_quota('req', 100, _promise=True) \
                    .then(functools.partial(actual_exec, 'req'))

                pool.sleep(0.5)
                mock_mem_stat['available'] = 150
                mock_mem_stat['free'] = 150

                self.get_result(2)

            self.assertGreater(abs(time_recs[0] - time_recs[1]), 0.4)
