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

import json
import operator
import os
import signal
import subprocess
import sys
import time
import unittest
import uuid

import numpy as np
from numpy.testing import assert_array_equal
import gevent

from mars import tensor as mt
from mars.cluster_info import ClusterInfoActor
from mars.compat import reduce
from mars.serialize.dataserializer import loads
from mars.config import options
from mars.tests.core import EtcdProcessHelper
from mars.utils import get_next_port
from mars.actors.core import new_client
from mars.scheduler import SessionManagerActor, ResourceActor
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
        self.scheduler_endpoints = []
        self.proc_schedulers = []
        self.proc_workers = []
        self.etcd_helper = None

    def tearDown(self):
        procs = tuple(self.proc_workers) + tuple(self.proc_schedulers)
        for p in procs:
            p.send_signal(signal.SIGINT)

        check_time = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(0.1)
            if time.time() - check_time > 5:
                break

        for p in procs:
            if p.poll() is None:
                p.kill()

        if self.etcd_helper:
            self.etcd_helper.stop()

    def start_processes(self, n_schedulers=1, n_workers=2, etcd=False, modules=None):
        old_not_errors = gevent.hub.Hub.NOT_ERROR
        gevent.hub.Hub.NOT_ERROR = (Exception,)

        scheduler_ports = [str(get_next_port()) for _ in range(n_schedulers)]
        self.scheduler_endpoints = ['127.0.0.1:' + p for p in scheduler_ports]

        append_args = []
        if modules:
            append_args.extend(['--load-modules', ','.join(modules)])

        if etcd:
            etcd_port = get_next_port()
            self.etcd_helper = EtcdProcessHelper(port_range_start=etcd_port)
            self.etcd_helper.run()
            options.kv_store = 'etcd://127.0.0.1:%s' % etcd_port
            append_args.extend(['--kv-store', options.kv_store])
        else:
            append_args.extend(['--schedulers', ','.join(self.scheduler_endpoints)])

        self.proc_schedulers = [
            subprocess.Popen([sys.executable, '-m', 'mars.scheduler',
                              '-H', '127.0.0.1',
                              '--level', 'debug',
                              '-p', p,
                              '--format', '%(asctime)-15s %(message)s']
                             + append_args)
            for p in scheduler_ports]
        self.proc_workers = [
            subprocess.Popen([sys.executable, '-m', 'mars.worker',
                              '-a', '127.0.0.1',
                              '--cpu-procs', '1',
                              '--level', 'debug',
                              '--cache-mem', '16m',
                              '--ignore-avail-mem']
                             + append_args)
            for _ in range(n_workers)
        ]

        actor_client = new_client()
        self.cluster_info = actor_client.actor_ref(
            ClusterInfoActor.default_name(), address=self.scheduler_endpoints[0])

        check_time = time.time()
        while True:
            try:
                started_schedulers = self.cluster_info.get_schedulers()
                if len(started_schedulers) < n_schedulers:
                    raise RuntimeError('Schedulers does not met requirement: %d < %d.' % (
                        len(started_schedulers), n_schedulers
                    ))
                actor_address = self.cluster_info.get_scheduler(SessionManagerActor.default_name())
                self.session_manager_ref = actor_client.actor_ref(
                    SessionManagerActor.default_name(), address=actor_address)

                actor_address = self.cluster_info.get_scheduler(ResourceActor.default_name())
                resource_ref = actor_client.actor_ref(ResourceActor.default_name(), address=actor_address)

                if resource_ref.get_worker_count() < n_workers:
                    raise RuntimeError('Workers does not met requirement: %d < %d.' % (
                        resource_ref.get_worker_count(), n_workers
                    ))
                break
            except:
                if time.time() - check_time > 20:
                    raise
                time.sleep(0.1)

        gevent.hub.Hub.NOT_ERROR = old_not_errors

    def check_process_statuses(self):
        for scheduler_proc in self.proc_schedulers:
            if scheduler_proc.poll() is not None:
                raise SystemError('Scheduler not started. exit code %s' % self.proc_scheduler.poll())
        for worker_proc in self.proc_workers:
            if worker_proc.poll() is not None:
                raise SystemError('Worker not started. exit code %s' % worker_proc.poll())

    def wait_for_termination(self, session_ref, graph_key):
        check_time = time.time()
        while True:
            time.sleep(0.1)
            self.check_process_statuses()
            if time.time() - check_time > 60:
                raise SystemError('Check graph status timeout')
            if session_ref.graph_state(graph_key) in GraphState.TERMINATED_STATES:
                return session_ref.graph_state(graph_key)

    def testMainWithoutEtcd(self):
        self.start_processes(n_schedulers=2)

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(self.session_manager_ref.create_session(session_id))

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
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

        a = mt.ones((100, 50), chunk_size=35) * 2 + 1
        b = mt.ones((50, 200), chunk_size=35) * 2 + 1
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

        base_arr = np.random.random((100, 100))
        a = mt.array(base_arr)
        sumv = reduce(operator.add, [a[:10, :10] for _ in range(10)])
        graph = sumv.build_graph()
        targets = [sumv.key]
        graph_key = uuid.uuid1()
        session_ref.submit_tensor_graph(json.dumps(graph.to_json()),
                                        graph_key, target_tensors=targets)

        state = self.wait_for_termination(session_ref, graph_key)
        self.assertEqual(state, GraphState.SUCCEEDED)

        expected = reduce(operator.add, [base_arr[:10, :10] for _ in range(10)])
        result = session_ref.fetch_result(graph_key, sumv.key)
        assert_array_equal(loads(result), expected)

    def testMainWithEtcd(self):
        self.start_processes(n_schedulers=2, etcd=True)

        session_id = uuid.uuid1()
        actor_client = new_client()

        session_ref = actor_client.actor_ref(self.session_manager_ref.create_session(session_id))

        a = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
        b = mt.ones((100, 100), chunk_size=30) * 2 * 1 + 1
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
