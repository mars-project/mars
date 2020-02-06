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
import os

from mars.tests.core import aio_case, create_actor_pool
from mars.errors import WorkerProcessStopped
from mars.utils import get_next_port, build_exc_info
from mars.worker import WorkerDaemonActor, DispatchActor, ProcessHelperActor
from mars.distributor import MarsDistributor
from mars.worker.utils import WorkerActor
from mars.worker.tests.base import WorkerCase


class DaemonSleeperActor(WorkerActor):
    async def post_create(self):
        await super().post_create()
        self._daemon_ref = self.promise_ref(WorkerDaemonActor.default_uid())
        await self._daemon_ref.register_process(self.ref(), os.getpid(), _tell=True)

    async def test_sleep(self, t, callback):
        await asyncio.sleep(t)
        await self.tell_promise(callback)


class DaemonTestActor(WorkerActor):
    def __init__(self):
        super().__init__()
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
            raise val[1].with_traceback(val[2])
        else:
            return val

    def handle_process_down_for_actors(self, halt_refs):
        self.reject_promise_refs(halt_refs, *build_exc_info(WorkerProcessStopped))


@aio_case
class Test(WorkerCase):
    async def testDaemon(self):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        async with create_actor_pool(n_process=2, distributor=MarsDistributor(2, 'w:0:'),
                                     address=mock_scheduler_addr) as pool:
            daemon_ref = await pool.create_actor(
                WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            sleeper_ref = await daemon_ref.create_actor(
                DaemonSleeperActor, uid='w:1:DaemonSleeperActor')
            await daemon_ref.create_actor(ProcessHelperActor, uid='w:1:ProcHelper')
            test_actor = await pool.create_actor(DaemonTestActor)
            await daemon_ref.register_actor_callback(
                test_actor, DaemonTestActor.handle_process_down_for_actors.__name__)

            await test_actor.run_test_sleep(sleeper_ref, 10, _tell=True)
            self.assertTrue(await daemon_ref.is_actor_process_alive(sleeper_ref))

            await asyncio.sleep(0.5)

            await daemon_ref.kill_actor_process(sleeper_ref)
            # repeated kill shall not produce errors
            await daemon_ref.kill_actor_process(sleeper_ref)
            self.assertFalse(await daemon_ref.is_actor_process_alive(sleeper_ref))

            await pool.restart_process(1)
            await daemon_ref.handle_process_down([1])
            await asyncio.sleep(1)
            self.assertTrue(await pool.has_actor(sleeper_ref))
            with self.assertRaises(WorkerProcessStopped):
                await test_actor.get_result()

            await test_actor.run_test_sleep(sleeper_ref, 1)
            await asyncio.sleep(1.1)
            await test_actor.get_result()
