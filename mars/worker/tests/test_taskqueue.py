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
import uuid

import gevent

from mars.actors import FunctionActor, create_actor_pool
from mars.worker import QuotaActor, TaskQueueActor
from mars.worker.tests.base import WorkerCase


class MockExecutionActor(FunctionActor):
    def __init__(self, quota):
        self._quota = quota

    def prepare_quota_request(self, _, graph_key):
        return {graph_key: self._quota}


class Test(WorkerCase):
    def testTaskQueueActor(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            pool.create_actor(MockExecutionActor, 10, uid='ExecutionActor')
            quota_ref = pool.create_actor(QuotaActor, 30, uid='MemQuotaActor')
            pool.create_actor(TaskQueueActor, 4, uid=TaskQueueActor.__name__)

            session_id = str(uuid.uuid4())
            chunk_keys = [str(uuid.uuid4()).replace('-', '') for _ in range(5)]

            with self.run_actor_test(pool) as test_actor:
                queue_ref = test_actor.promise_ref(TaskQueueActor.__name__)
                res_times = dict()

                def callback_fun(key):
                    res_times[key] = time.time()

                for idx, k in enumerate(chunk_keys):
                    depth = len(chunk_keys) - idx
                    queue_ref.enqueue_task(session_id, k, dict(depth=depth), _promise=True) \
                        .then(functools.partial(callback_fun, k))

                gevent.sleep(1)
                self.assertEqual(queue_ref.get_allocated_count(), 3)

                queue_ref.update_priority(
                    session_id, chunk_keys[-1], dict(depth=len(chunk_keys)))
                quota_ref.release_quota(chunk_keys[0])
                queue_ref.release_task(session_id, chunk_keys[0])
                gevent.sleep(0.5)

                self.assertIn(chunk_keys[-1], res_times)
                for k in chunk_keys[:3]:
                    self.assertLessEqual(res_times[k], res_times[-1] - 1)
                    self.assertIn(k, res_times)
