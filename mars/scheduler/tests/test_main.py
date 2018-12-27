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
import os
import sys
import subprocess
import uuid
import json
import signal
import unittest

import numpy as np
from numpy.testing import assert_array_equal
import gevent

from mars.serialize.dataserializer import loads
from mars.config import options
from mars.tensor.expressions.datasource import ones
from mars.utils import get_next_port
from mars.actors.core import new_client
from mars.scheduler import SessionActor, KVStoreActor
from mars.scheduler.graph import GraphState


@unittest.skipIf(sys.platform == 'win32', "plasma don't support windows")
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

    def setUp(self):
        scheduler_port = str(get_next_port())
        proc_worker = subprocess.Popen([sys.executable, '-m', 'mars.worker',
                                        '-a', '127.0.0.1',
                                        '--cpu-procs', '2',
                                        '--level', 'debug',
                                        '--cache-mem', '16m',
                                        '--schedulers', '127.0.0.1:' + scheduler_port,
                                        '--ignore-avail-mem'])
        proc_scheduler = subprocess.Popen([sys.executable, '-m', 'mars.scheduler',
                                           '-H', '127.0.0.1',
                                           '--level', 'debug',
                                           '-p', scheduler_port,
                                           '--format', '%(asctime)-15s %(message)s'])

        self.scheduler_port = scheduler_port
        self.proc_worker = proc_worker
        self.proc_scheduler = proc_scheduler

        time.sleep(2)
        actor_client = new_client()
        check_time = time.time()
        while True:
            try:
                kv_ref = actor_client.actor_ref(KVStoreActor.default_name(), address='127.0.0.1:' + scheduler_port)
                if actor_client.has_actor(kv_ref):
                    break
                else:
                    raise SystemError('Check meta_timestamp timeout')
            except:
                if time.time() - check_time > 10:
                    raise
                time.sleep(1)

        check_time = time.time()
        while True:
            content = kv_ref.read('/workers/meta_timestamp', silent=True)
            if not content:
                time.sleep(0.5)
                self.check_process_statuses()
                if time.time() - check_time > 20:
                    raise SystemError('Check meta_timestamp timeout')
            else:
                break

        self.exceptions = gevent.hub.Hub.NOT_ERROR
        gevent.hub.Hub.NOT_ERROR = (Exception,)

    def tearDown(self):
        procs = (self.proc_worker, self.proc_scheduler)
        for p in procs:
            p.send_signal(signal.SIGINT)

        check_time = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(1)
            if time.time() - check_time > 5:
                break

        for p in procs:
            if p.poll() is None:
                p.kill()

        gevent.hub.Hub.NOT_ERROR = self.exceptions

    def check_process_statuses(self):
        if self.proc_scheduler.poll() is not None:
            raise SystemError('Scheduler not started. exit code %s' % self.proc_scheduler.poll())
        if self.proc_worker.poll() is not None:
            raise SystemError('Worker not started. exit code %s' % self.proc_worker.poll())

    def wait_for_termination(self, session_ref, graph_key):
        check_time = time.time()
        while True:
            time.sleep(1)
            self.check_process_statuses()
            if time.time() - check_time > 60:
                raise SystemError('Check graph status timeout')
            if session_ref.graph_state(graph_key) in GraphState.TERMINATED_STATES:
                return session_ref.graph_state(graph_key)

    def testMain(self):
        session_id = uuid.uuid1()
        scheduler_address = '127.0.0.1:' + self.scheduler_port
        actor_client = new_client()
        session_ref = actor_client.create_actor(SessionActor,
                                                uid=SessionActor.gen_name(session_id),
                                                address=scheduler_address,
                                                session_id=session_id)
        a = ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = ones((100, 100), chunk_size=30) * 2 * 1 + 1
        c = (a * b * 2 + 1).sum()
        graph = c.build_graph()
        targets = [c.key]
        graph_key = uuid.uuid1()
        session_ref.submit_tensor_graph(json.dumps(graph.to_json()),
                                        graph_key, target_tensors=targets)

        state = self.wait_for_termination(session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        result = session_ref.fetch_result(graph_key, c.key)
        expected = (np.ones(a.shape) * 2 * 1 + 1) ** 2 * 2 + 1
        assert_array_equal(loads(result), expected.sum())

        graph_key = uuid.uuid1()
        session_ref.submit_tensor_graph(json.dumps(graph.to_json()),
                                        graph_key, target_tensors=targets)

        # todo this behavior may change when eager mode is introduced
        state = self.wait_for_termination(session_ref, graph_key)
        self.assertEqual(state, GraphState.FAILED)

        a = ones((100, 50), chunk_size=30) * 2 + 1
        b = ones((50, 200), chunk_size=30) * 2 + 1
        c = a.dot(b)
        graph = c.build_graph()
        targets = [c.key]
        graph_key = uuid.uuid1()
        session_ref.submit_tensor_graph(json.dumps(graph.to_json()),
                                        graph_key, target_tensors=targets)

        state = self.wait_for_termination(session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)
        result = session_ref.fetch_result(graph_key, c.key)
        assert_array_equal(loads(result), np.ones((100, 200)) * 450)
