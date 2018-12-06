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

import os
import sys
import signal
import subprocess
import time
import uuid

import gevent

from mars.actors import FunctionActor, create_actor_pool
from mars.compat import unittest
from mars.config import options
from mars.utils import get_next_port, serialize_graph
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import ResourceActor
from mars.scheduler.kvstore import KVStoreActor


class PromiseReplyTestActor(FunctionActor):
    def __init__(self):
        super(PromiseReplyTestActor, self).__init__()
        self._replied = False

    def reply(self, _):
        self._replied = True

    def get_reply(self):
        return self._replied


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from mars import kvstore
        options.worker.spill_directory = os.path.join(tempfile.gettempdir(), 'mars_test_spill')

        cls._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(options.worker.spill_directory):
            shutil.rmtree(options.worker.spill_directory)

    def testExecuteWorker(self):
        import mars.tensor as mt
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        worker_plasma_sock = '/tmp/plasma_%d_%d.sock' % (os.getpid(), id(PromiseReplyTestActor))
        try:

            session_id = str(uuid.uuid4())
            with create_actor_pool(n_process=1, backend='gevent',
                                   address=mock_scheduler_addr) as pool:
                pool.create_actor(ClusterInfoActor, schedulers=[mock_scheduler_addr],
                                  uid=ClusterInfoActor.default_name())
                kv_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
                pool.create_actor(ResourceActor, uid=ResourceActor.default_name())

                proc = subprocess.Popen([sys.executable, '-m', 'mars.worker',
                                         '-a', '127.0.0.1',
                                         '--schedulers', mock_scheduler_addr,
                                         '--cpu-procs', '1',
                                         '--cache-mem', '10m',
                                         '--plasma-socket', worker_plasma_sock,
                                         '--ignore-avail-mem'])
                worker_ips = []

                def waiter():
                    check_time = time.time()
                    while True:
                        if kv_ref.read('/workers/meta_timestamp', silent=True) is None:
                            gevent.sleep(0.5)
                            if proc.poll() is not None:
                                raise SystemError('Worker dead. exit code %s' % proc.poll())
                            if time.time() - check_time > 20:
                                raise SystemError('Check meta_timestamp timeout')
                            continue
                        else:
                            break
                    val = kv_ref.read('/workers/meta')
                    worker_ips.extend([c.key.rsplit('/', 1)[-1] for c in val.children])

                gl = gevent.spawn(waiter)
                gl.join()

                a = mt.ones((100, 50), chunks=30)
                b = mt.ones((50, 200), chunks=30)
                result = a.dot(b)

                graph = result.build_graph(tiled=True)

                reply_ref = pool.create_actor(PromiseReplyTestActor)
                reply_callback = ((reply_ref.uid, reply_ref.address), 'reply')

                executor_ref = pool.actor_ref('ExecutionActor', address=worker_ips[0])
                io_meta = dict(chunks=[c.key for c in result.chunks])
                executor_ref.execute_graph(session_id, str(id(graph)), serialize_graph(graph),
                                           io_meta, None, callback=reply_callback)

                check_time = time.time()
                while not reply_ref.get_reply():
                    gevent.sleep(0.1)
                    if time.time() - check_time > 20:
                        raise SystemError('Check reply timeout')
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                check_time = time.time()
                while True:
                    time.sleep(1)
                    if proc.poll() is not None or time.time() - check_time >= 5:
                        break
                if proc.poll() is None:
                    proc.kill()
            if os.path.exists(worker_plasma_sock):
                os.unlink(worker_plasma_sock)
