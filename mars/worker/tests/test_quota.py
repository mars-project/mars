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

import time
import gevent

from mars.actors import create_actor_pool
from mars.promise import PromiseActor
from mars.worker import QuotaActor
from mars.worker.tests.base import WorkerCase


class QuotaTestActor(PromiseActor):
    def __init__(self, alloc_size):
        super(QuotaTestActor, self).__init__()
        self._alloc_size = alloc_size
        self._end_time = None

    def mock_step(self, key):
        ref = self.promise_ref('QuotaActor')

        def actual_exec():
            gevent.sleep(1)
            ref.release_quota(key)
            self._end_time = time.time()

        ref.request_quota(key, self._alloc_size, _promise=True) \
            .then(actual_exec)

    def get_end_time(self):
        return self._end_time


class BatchQuotaTestActor(PromiseActor):
    def __init__(self, alloc_size):
        super(BatchQuotaTestActor, self).__init__()
        self._alloc_size = alloc_size
        self._end_time = None

    def mock_step(self, keys):
        ref = self.promise_ref('QuotaActor')

        def actual_exec():
            gevent.sleep(1)
            for k in keys:
                ref.release_quota(k)
            self._end_time = time.time()

        batch = dict((k, self._alloc_size) for k in keys)
        ref.request_batch_quota(batch, _promise=True) \
            .then(actual_exec)

    def get_end_time(self):
        return self._end_time


class Test(WorkerCase):
    def testQuota(self):
        with create_actor_pool() as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid='QuotaActor')
            test_refs = [pool.create_actor(QuotaTestActor, 100) for _ in range(4)]

            def test_method():
                for ref in test_refs:
                    ref.mock_step(str(id(ref)))
                gevent.sleep(3)
                return [ref.get_end_time() for ref in test_refs]

            gl = gevent.spawn(test_method)
            gl.join()
            end_time = gl.value
            self.assertLess(abs(end_time[0] - end_time[1]), 0.1)
            self.assertLess(abs(end_time[0] - end_time[2]), 0.1)
            self.assertGreater(abs(end_time[0] - end_time[3]), 0.9)

            self.assertEqual(quota_ref.get_allocated_size(), 0)

    def testBatchQuota(self):
        with create_actor_pool() as pool:
            quota_ref = pool.create_actor(QuotaActor, 300, uid='QuotaActor')
            test_refs = [pool.create_actor(BatchQuotaTestActor, 100) for _ in range(2)]

            def test_method():
                for ref in test_refs:
                    ref_str = str(id(ref))
                    ref.mock_step([ref_str + '_0', ref_str + '_1'])
                gevent.sleep(3)
                return [ref.get_end_time() for ref in test_refs]

            gl = gevent.spawn(test_method)
            gl.join()
            end_time = gl.value
            self.assertGreater(abs(end_time[0] - end_time[1]), 0.9)

            self.assertEqual(quota_ref.get_allocated_size(), 0)
