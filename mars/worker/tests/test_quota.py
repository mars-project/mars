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

from mars.actors import create_actor_pool
from mars.distributor import BaseDistributor
from mars.utils import get_next_port
from mars.worker import QuotaActor
from mars.worker.tests.base import WorkerCase


class Test(WorkerCase):
    def testQuota(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, distributor=BaseDistributor(0),
                               backend='gevent', address=local_pool_addr) as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid='QuotaActor')

            end_time = []
            for idx in range(4):
                x = str(idx)
                with self.run_actor_test(pool) as test_actor:
                    ref = test_actor.promise_ref('QuotaActor')

                    def actual_exec(x):
                        test_actor.ctx.sleep(1)
                        ref.release_quota(x)
                        end_time.append(time.time())
                        test_actor.set_result(None)

                    ref.request_quota(x, 100, _promise=True) \
                        .then(functools.partial(actual_exec, x))

            pool.sleep(2.5)
            self.assertLess(abs(end_time[0] - end_time[1]), 0.1)
            self.assertLess(abs(end_time[0] - end_time[2]), 0.1)
            self.assertGreater(abs(end_time[0] - end_time[3]), 0.9)
            self.assertEqual(quota_ref.get_allocated_size(), 0)

    def testBatchQuota(self):
        local_pool_addr = 'localhost:%d' % get_next_port()
        with create_actor_pool(n_process=1, distributor=BaseDistributor(0),
                               backend='gevent', address=local_pool_addr) as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid='QuotaActor')

            end_time = []
            for idx in range(2):
                x = str(idx)
                with self.run_actor_test(pool) as test_actor:
                    ref = test_actor.promise_ref('QuotaActor')

                    def actual_exec(keys):
                        test_actor.ctx.sleep(1)
                        for k in keys:
                            ref.release_quota(k)
                        end_time.append(time.time())
                        test_actor.set_result(None)

                    keys = [x + '_0', x + '_1']
                    batch = dict((k, 100) for k in keys)
                    ref.request_batch_quota(batch, _promise=True) \
                        .then(functools.partial(actual_exec, keys))

            pool.sleep(2.5)
            self.assertGreater(abs(end_time[0] - end_time[1]), 0.9)
            self.assertEqual(quota_ref.get_allocated_size(), 0)
