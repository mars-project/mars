# -*- coding: utf-8 -*-
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
import requests
import json
import unittest
import mock
import os
import sys
import signal
import subprocess

import gevent
import numpy as np
from numpy.testing import assert_array_equal


from mars import tensor as mt
from mars.tensor.execution.core import Executor
from mars.actors import create_actor_pool, new_client
from mars.utils import get_next_port
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import SessionManagerActor, KVStoreActor, ResourceActor
from mars.scheduler.graph import GraphActor
from mars.web import MarsWeb
from mars.session import new_session
from mars.serialize.dataserializer import dumps, loads
from mars.config import options


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
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
                                        '--level', 'debug',
                                        '--cpu-procs', '2',
                                        '--cache-mem', '10m',
                                        '--schedulers', '127.0.0.1:' + scheduler_port,
                                        '--ignore-avail-mem'])
        proc_scheduler = subprocess.Popen([sys.executable, '-m', 'mars.scheduler',
                                           '--nproc', '1',
                                           '--level', 'debug',
                                           '-H', '127.0.0.1',
                                           '-p', scheduler_port,
                                           '--format', '%(asctime)-15s %(message)s'])

        self.scheduler_port = scheduler_port
        self.proc_worker = proc_worker
        self.proc_scheduler = proc_scheduler

        actor_client = new_client()
        time.sleep(2)
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
            if self.proc_scheduler.poll() is not None:
                raise SystemError('Scheduler not started. exit code %s' % self.proc_scheduler.poll())
            if self.proc_worker.poll() is not None:
                raise SystemError('Worker not started. exit code %s' % self.proc_worker.poll())
            if time.time() - check_time > 20:
                raise SystemError('Check meta_timestamp timeout')

            if not content:
                time.sleep(0.5)
            else:
                break

        web_port = str(get_next_port())
        self.web_port = web_port
        proc_web = subprocess.Popen([sys.executable, '-m', 'mars.web',
                                    '-H', '127.0.0.1',
                                     '--level', 'debug',
                                     '--ui-port', web_port,
                                     '-s', '127.0.0.1:' + self.scheduler_port])
        self.proc_web = proc_web

        service_ep = 'http://127.0.0.1:' + self.web_port
        check_time = time.time()
        while True:
            if time.time() - check_time > 30:
                raise SystemError('Wait for service start timeout')
            try:
                resp = requests.get(service_ep + '/api', timeout=1)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(1)
                continue
            if resp.status_code >= 400:
                time.sleep(1)
                continue
            break

        self.exceptions = gevent.hub.Hub.NOT_ERROR
        gevent.hub.Hub.NOT_ERROR = (Exception,)

    def tearDown(self):
        procs = (self.proc_web, self.proc_worker, self.proc_scheduler)
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

    def testApi(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        with new_session(service_ep) as sess:
            self.assertEqual(sess.count_workers(), 1)
            a = mt.ones((100, 100), chunks=30)
            b = mt.ones((100, 100), chunks=30)
            c = a.dot(b)
            value = sess.run(c)
            assert_array_equal(value, np.ones((100, 100)) * 100)

            # todo this behavior may change when eager mode is introduced
            with self.assertRaises(SystemError):
                sess.run(c)

            va = np.random.randint(0, 10000, (100, 100))
            vb = np.random.randint(0, 10000, (100, 100))
            a = mt.array(va, chunks=30)
            b = mt.array(vb, chunks=30)
            c = a.dot(b)
            value = sess.run(c, timeout=120)
            assert_array_equal(value, va.dot(vb))

            # test test multiple outputs
            a = mt.random.rand(10, 10)
            U, s, V, raw = sess.run(list(mt.linalg.svd(a)) + [a])
            np.testing.assert_allclose(U.dot(np.diag(s).dot(V)), raw)

            # check web UI requests
            res = requests.get(service_ep)
            self.assertEqual(res.status_code, 200)

            res = requests.get(service_ep + '/task')
            self.assertEqual(res.status_code, 200)

            res = requests.get(service_ep + '/worker')
            self.assertEqual(res.status_code, 200)


class TestWithMockServer(unittest.TestCase):
    def setUp(self):
        self._executor = Executor('numpy')

        # create scheduler pool with needed actor
        scheduler_address = '127.0.0.1:' + str(get_next_port())
        self._scheduler_address = scheduler_address
        pool = create_actor_pool(address=scheduler_address, n_process=1, backend='gevent')
        pool.create_actor(ClusterInfoActor, [scheduler_address], uid=ClusterInfoActor.default_name())
        pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
        pool.create_actor(SessionManagerActor, uid=SessionManagerActor.default_name())
        self._kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
        self._pool = pool

        self.start_web(scheduler_address)

    def tearDown(self):
        self._web.stop()
        self._pool.stop()

    def start_web(self, scheduler_address):
        import gevent.monkey
        gevent.monkey.patch_all(thread=False)

        web_port = str(get_next_port())
        mars_web = MarsWeb(port=int(web_port), scheduler_ip=scheduler_address)
        mars_web.start()
        service_ep = 'http://127.0.0.1:' + web_port
        self._service_ep = service_ep
        self._web = mars_web

        check_time = time.time()
        while True:
            if time.time() - check_time > 30:
                raise SystemError('Wait for service start timeout')
            try:
                resp = requests.get(self._service_ep + '/api', timeout=1)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(1)
                continue
            if resp.status_code >= 400:
                time.sleep(1)
                continue
            break

    @mock.patch(GraphActor.__module__ + '.GraphActor.execute_graph')
    @mock.patch(GraphActor.__module__ + '.ResultReceiverActor.fetch_tensor')
    def testApi(self, mock_fetch_tensor, _):
        with new_session(self._service_ep) as sess:
            self._kv_store_ref.write('/workers/meta/%s' % 'mock_endpoint', 'mock_meta')
            self.assertEqual(sess.count_workers(), 1)

            a = mt.ones((100, 100), chunks=30)
            b = mt.ones((100, 100), chunks=30)
            c = a.dot(b)
            graph_key = sess.run(c, timeout=120, wait=False)
            self._kv_store_ref.write('/sessions/%s/graph/%s/state' % (sess.session_id, graph_key), 'SUCCEEDED')
            graph_url = '%s/api/session/%s/graph/%s' % (self._service_ep, sess.session_id, graph_key)
            graph_state = json.loads(requests.get(graph_url).text)
            self.assertEqual(graph_state['state'], 'success')
            mock_fetch_tensor.return_value = dumps(self._executor.execute_tensor(c, concat=True)[0])
            data_url = graph_url + '/data/' + c.key
            data = loads(requests.get(data_url).content)
            assert_array_equal(data, np.ones((100, 100)) * 100)

            # test web session endpoint setter
            self._web.stop()
            self.start_web(self._scheduler_address)
            sess.endpoint = self._service_ep
            graph_url = '%s/api/session/%s/graph/%s' % (self._service_ep, sess.session_id, graph_key)
            data_url = graph_url + '/data/' + c.key
            data = loads(requests.get(data_url).content)
            assert_array_equal(data, np.ones((100, 100)) * 100)
