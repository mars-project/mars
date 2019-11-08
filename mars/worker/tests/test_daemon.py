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

import os

from mars.actors import create_actor_pool
from mars.compat import six
from mars.errors import WorkerProcessStopped
from mars.utils import get_next_port, build_exc_info
from mars.worker import WorkerDaemonActor, DispatchActor, ProcessHelperActor
from mars.distributor import MarsDistributor
from mars.worker.utils import WorkerActor
from mars.worker.tests.base import WorkerCase


class DaemonSleeperActor(WorkerActor):
    def post_create(self):
        super(DaemonSleeperActor, self).__init__()
        self._daemon_ref = self.promise_ref(WorkerDaemonActor.default_uid())
        self._daemon_ref.register_process(self.ref(), os.getpid(), _tell=True)

    def test_sleep(self, t, callback):
        self.ctx.sleep(t)
        self.tell_promise(callback)


class DaemonTestActor(WorkerActor):
    def __init__(self):
        super(DaemonTestActor, self).__init__()
        self._result = None

    def run_test_sleep(self, sleeper_ref, t):
        ref = self.promise_ref(sleeper_ref)
        ref.test_sleep(t, _promise=True) \
            .then(lambda *_: self.set_result(None)) \
            .catch(lambda *exc: self.set_result(exc, False))

    def set_result(self, val, accept=True):
        self._result = (val, accept)

    def get_result(self):
        if not self._result:
            raise ValueError
        val, accept = self._result
        if not accept:
            six.reraise(*val)
        else:
            return val

    def handle_process_down_for_actors(self, halt_refs):
        self.reject_promise_refs(halt_refs, *build_exc_info(WorkerProcessStopped))


class Test(WorkerCase):
    def testDaemon(self):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=2, backend='gevent', distributor=MarsDistributor(2, 'w:0:'),
                               address=mock_scheduler_addr) as pool:
            daemon_ref = pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            sleeper_ref = daemon_ref.create_actor(DaemonSleeperActor,
                                                  uid='w:1:DaemonSleeperActor')
            daemon_ref.create_actor(ProcessHelperActor, uid='w:1:ProcHelper')
            test_actor = pool.create_actor(DaemonTestActor)
            daemon_ref.register_actor_callback(
                test_actor, DaemonTestActor.handle_process_down_for_actors.__name__)

            test_actor.run_test_sleep(sleeper_ref, 10, _tell=True)
            self.assertTrue(daemon_ref.is_actor_process_alive(sleeper_ref))

            pool.sleep(0.5)

            daemon_ref.kill_actor_process(sleeper_ref)
            # repeated kill shall not produce errors
            daemon_ref.kill_actor_process(sleeper_ref)
            self.assertFalse(daemon_ref.is_actor_process_alive(sleeper_ref))

            pool.restart_process(1)
            daemon_ref.handle_process_down([1])
            pool.sleep(1)
            self.assertTrue(pool.has_actor(sleeper_ref))
            with self.assertRaises(WorkerProcessStopped):
                test_actor.get_result()

            test_actor.run_test_sleep(sleeper_ref, 1)
            pool.sleep(1.5)
            test_actor.get_result()
