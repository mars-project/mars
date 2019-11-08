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

import contextlib
import os
import signal
import subprocess
import sys
import time
import unittest
import uuid

import gevent

from mars.actors import create_actor_pool
from mars.compat import TimeoutError  # pylint: disable=W0622
from mars.config import options
from mars.promise import PromiseActor
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ResourceActor, ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import require_cupy
from mars.worker import DispatchActor, WorkerDaemonActor


class WorkerProcessTestActor(PromiseActor):
    def __init__(self):
        super(WorkerProcessTestActor, self).__init__()
        self._replied = False

    def run_test(self, worker, calc_device=None):
        import mars.tensor as mt
        from mars.worker import ExecutionActor

        session_id = str(uuid.uuid4())

        gpu = calc_device in ('cuda',)
        a = mt.random.rand(100, 50, chunk_size=30, gpu=gpu)
        b = mt.random.rand(50, 200, chunk_size=30, gpu=gpu)
        result = a.dot(b)

        graph = result.build_graph(tiled=True)

        executor_ref = self.promise_ref(ExecutionActor.default_uid(), address=worker)
        io_meta = dict(chunks=[c.key for c in result.chunks])

        graph_key = str(id(graph))
        executor_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                   io_meta, None, calc_device=calc_device, _promise=True) \
            .then(lambda *_: setattr(self, '_replied', True))

    def get_reply(self):
        return self._replied


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from mars import kvstore
        cls._spill_dir = options.worker.spill_directory = os.path.join(tempfile.gettempdir(), 'mars_test_spill')
        cls._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(options.worker.spill_directory):
            shutil.rmtree(options.worker.spill_directory)

    @staticmethod
    def _wait_worker_ready(proc, resource_ref):
        worker_ips = []

        def waiter():
            check_time = time.time()
            while True:
                if not resource_ref.get_workers_meta():
                    gevent.sleep(0.1)
                    if proc.poll() is not None:
                        raise SystemError('Worker dead. exit code %s' % proc.poll())
                    if time.time() - check_time > 20:
                        raise TimeoutError('Check meta_timestamp timeout')
                    continue
                else:
                    break
            val = resource_ref.get_workers_meta()
            worker_ips.extend(val.keys())

        gl = gevent.spawn(waiter)
        gl.join()
        return worker_ips[0]

    @contextlib.contextmanager
    def _start_worker_process(self, no_cuda=True, cuda_device=None):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        try:
            with create_actor_pool(n_process=1, backend='gevent',
                                   address=mock_scheduler_addr) as pool:
                pool.create_actor(SchedulerClusterInfoActor, [mock_scheduler_addr],
                                  uid=SchedulerClusterInfoActor.default_uid())

                pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
                resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())

                args = [sys.executable, '-m', 'mars.worker',
                        '-a', '127.0.0.1',
                        '--schedulers', mock_scheduler_addr,
                        '--cpu-procs', '1',
                        '--cache-mem', '10m',
                        '--spill-dir', self._spill_dir,
                        '--ignore-avail-mem']
                env = os.environ.copy()
                if no_cuda:
                    args.append('--no-cuda')
                else:
                    env['CUDA_VISIBLE_DEVICES'] = cuda_device
                proc = subprocess.Popen(args, env=env)
                worker_endpoint = self._wait_worker_ready(proc, resource_ref)

                yield pool, worker_endpoint
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                check_time = time.time()
                while True:
                    time.sleep(0.1)
                    if proc.poll() is not None or time.time() - check_time >= 5:
                        break
                if proc.poll() is None:
                    proc.kill()
            if os.path.exists(options.worker.plasma_socket):
                os.unlink(options.worker.plasma_socket)

    def testExecuteWorker(self):
        with self._start_worker_process() as (pool, worker_endpoint):
            test_ref = pool.create_actor(WorkerProcessTestActor)
            test_ref.run_test(worker_endpoint, _tell=True)

            check_time = time.time()
            while not test_ref.get_reply():
                gevent.sleep(0.1)
                if time.time() - check_time > 20:
                    raise TimeoutError('Check reply timeout')

    @require_cupy
    def testExecuteCudaWorker(self):
        with self._start_worker_process(no_cuda=False, cuda_device='0') as (pool, worker_endpoint):
            test_ref = pool.create_actor(WorkerProcessTestActor)
            test_ref.run_test(worker_endpoint, calc_device='cuda', _tell=True)

            check_time = time.time()
            while not test_ref.get_reply():
                gevent.sleep(0.1)
                if time.time() - check_time > 2000:
                    raise TimeoutError('Check reply timeout')

    def testWorkerProcessRestart(self):
        with self._start_worker_process() as (pool, worker_endpoint):
            daemon_ref = pool.actor_ref(WorkerDaemonActor.default_uid(), address=worker_endpoint)
            dispatch_ref = pool.actor_ref(DispatchActor.default_uid(), address=worker_endpoint)
            cpu_slots = dispatch_ref.get_slots('cpu')
            calc_ref = pool.actor_ref(cpu_slots[0], address=worker_endpoint)
            daemon_ref.kill_actor_process(calc_ref)

            check_start = time.time()
            while not daemon_ref.is_actor_process_alive(calc_ref):
                gevent.sleep(0.1)
                if time.time() - check_start > 10:
                    raise TimeoutError('Check process restart timeout')
