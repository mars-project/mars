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

import requests
import json
import unittest
import os
import sys
import signal
import subprocess
import time
import uuid

import gevent
import numpy as np
from numpy.testing import assert_array_equal

from mars import tensor as mt
from mars.actors import new_client
from mars.config import options
from mars.scheduler import ResourceActor
from mars.session import new_session
from mars.serialize.dataserializer import dumps
from mars.tests.core import mock
from mars.utils import get_next_port


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

    def wait_scheduler_worker_start(self):
        old_not_errors = gevent.hub.Hub.NOT_ERROR
        gevent.hub.Hub.NOT_ERROR = (Exception,)

        actor_client = new_client()
        time.sleep(1)
        check_time = time.time()
        while True:
            try:
                resource_ref = actor_client.actor_ref(
                    ResourceActor.default_uid(), address='127.0.0.1:' + self.scheduler_port)
                if actor_client.has_actor(resource_ref):
                    break
                else:
                    raise SystemError('Check meta_timestamp timeout')
            except:  # noqa: E722
                if time.time() - check_time > 10:
                    raise
                time.sleep(0.1)

        check_time = time.time()
        while not resource_ref.get_worker_count():
            if self.proc_scheduler.poll() is not None:
                raise SystemError('Scheduler not started. exit code %s' % self.proc_scheduler.poll())
            if self.proc_worker.poll() is not None:
                raise SystemError('Worker not started. exit code %s' % self.proc_worker.poll())
            if time.time() - check_time > 20:
                raise SystemError('Check meta_timestamp timeout')

            time.sleep(0.1)

        gevent.hub.Hub.NOT_ERROR = old_not_errors

    def setUp(self):
        worker_port = self.worker_port = str(get_next_port())
        scheduler_port = self.scheduler_port = str(get_next_port())
        proc_worker = subprocess.Popen([sys.executable, '-m', 'mars.worker',
                                        '-a', '127.0.0.1',
                                        '-p', worker_port,
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

        self.proc_worker = proc_worker
        self.proc_scheduler = proc_scheduler

        self.wait_scheduler_worker_start()

        web_port = self.web_port = str(get_next_port())
        proc_web = subprocess.Popen([sys.executable, '-m', 'mars.web',
                                    '-H', '127.0.0.1',
                                     '--level', 'debug',
                                     '-p', web_port,
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
                time.sleep(0.1)
                continue
            if resp.status_code >= 400:
                time.sleep(0.1)
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

    def testWebApi(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            self.assertEqual(sess.count_workers(), 1)
            a = mt.ones((100, 100), chunk_size=30)
            b = mt.ones((100, 100), chunk_size=30)
            c = a.dot(b)
            value = sess.run(c, timeout=timeout)
            assert_array_equal(value, np.ones((100, 100)) * 100)

            # check resubmission
            value2 = sess.run(c, timeout=timeout)
            assert_array_equal(value, value2)

            # check when local compression libs are missing
            from mars.serialize import dataserializer
            try:
                a = mt.ones((10, 10), chunk_size=30)
                b = mt.ones((10, 10), chunk_size=30)
                c = a.dot(b)
                value = sess.run(c, timeout=timeout)
                assert_array_equal(value, np.ones((10, 10)) * 10)

                dataserializer.decompressors[dataserializer.CompressType.LZ4] = None
                dataserializer.decompressobjs[dataserializer.CompressType.LZ4] = None
                dataserializer.compress_openers[dataserializer.CompressType.LZ4] = None

                assert_array_equal(sess.fetch(c), np.ones((10, 10)) * 10)
            finally:
                dataserializer.decompressors[dataserializer.CompressType.LZ4] = dataserializer.lz4_decompress
                dataserializer.decompressobjs[dataserializer.CompressType.LZ4] = dataserializer.lz4_decompressobj
                dataserializer.compress_openers[dataserializer.CompressType.LZ4] = dataserializer.lz4_open

            va = np.random.randint(0, 10000, (100, 100))
            vb = np.random.randint(0, 10000, (100, 100))
            a = mt.array(va, chunk_size=30)
            b = mt.array(vb, chunk_size=30)
            c = a.dot(b)
            value = sess.run(c, timeout=timeout)
            assert_array_equal(value, va.dot(vb))

            graphs = sess.get_graph_states()

            # make sure status got uploaded
            time.sleep(1.5)

            # check web UI requests
            res = requests.get(service_ep)
            self.assertEqual(res.status_code, 200)

            res = requests.get('%s/scheduler' % (service_ep,))
            self.assertEqual(res.status_code, 200)
            res = requests.get('%s/scheduler/127.0.0.1:%s' % (service_ep, self.scheduler_port))
            self.assertEqual(res.status_code, 200)

            res = requests.get('%s/worker' % (service_ep,))
            self.assertEqual(res.status_code, 200)
            res = requests.get('%s/worker/127.0.0.1:%s' % (service_ep, self.worker_port))
            self.assertEqual(res.status_code, 200)

            res = requests.get('%s/session' % (service_ep,))
            self.assertEqual(res.status_code, 200)
            task_id = next(iter(graphs.keys()))
            res = requests.get('%s/session/%s/graph/%s' % (service_ep, sess._session_id, task_id))
            self.assertEqual(res.status_code, 200)
            res = requests.get('%s/session/%s/graph/%s/running_nodes' % (service_ep, sess._session_id, task_id))
            self.assertEqual(res.status_code, 200)

            from mars.web.task_pages import PROGRESS_APP_NAME
            res = requests.get('%s/%s?session_id=%s&task_id=%s'
                               % (service_ep, PROGRESS_APP_NAME, sess._session_id, task_id))
            self.assertEqual(res.status_code, 200)


class MockResponse:
    def __init__(self, status_code, json_text=None, data=None):
        self._json_text = json_text
        self._content = data
        self._status_code = status_code

    @property
    def text(self):
        return json.dumps(self._json_text)

    @property
    def content(self):
        return self._content

    @property
    def status_code(self):
        return self._status_code


class MockedServer(object):
    def __init__(self):
        self._data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @staticmethod
    def mocked_requests_get(*arg, **_):
        url = arg[0]
        if url.endswith('worker'):
            return MockResponse(200, json_text=1)
        if url.split('/')[-2] == 'graph':
            return MockResponse(200, json_text={"state": 'success'})
        elif url.split('/')[-2] == 'data':
            data = dumps(np.ones((100, 100)) * 100)
            return MockResponse(200, data=data)

    @staticmethod
    def mocked_requests_post(*arg, **_):
        url = arg[0]
        if url.endswith('session'):
            return MockResponse(200, json_text={"session_id": str(uuid.uuid4())})
        elif url.endswith('graph'):
            return MockResponse(200, json_text={"graph_key": str(uuid.uuid4())})
        else:
            return MockResponse(404)

    @staticmethod
    def mocked_requests_delete(*_):
        return MockResponse(200)


class TestWithMockServer(unittest.TestCase):
    def setUp(self):
        self._service_ep = 'http://mock.com'

    @mock.patch('requests.Session.get', side_effect=MockedServer.mocked_requests_get)
    @mock.patch('requests.Session.post', side_effect=MockedServer.mocked_requests_post)
    @mock.patch('requests.Session.delete', side_effect=MockedServer.mocked_requests_delete)
    def testApi(self, *_):
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(self._service_ep) as sess:
            self.assertEqual(sess.count_workers(), 1)

            a = mt.ones((100, 100), chunk_size=30)
            b = mt.ones((100, 100), chunk_size=30)
            c = a.dot(b)

            result = sess.run(c, timeout=timeout)
            assert_array_equal(result, np.ones((100, 100)) * 100)

            d = a * 100
            self.assertIsNone(sess.run(d, fetch=False, timeout=120))
            assert_array_equal(sess.run(d, timeout=120), np.ones((100, 100)) * 100)
