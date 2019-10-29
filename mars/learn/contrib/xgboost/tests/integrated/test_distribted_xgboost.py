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

import subprocess
import sys
import time
import os
import signal
import unittest

import requests
import gevent

import mars.tensor as mt
from mars.utils import get_next_port
from mars.actors import new_client
from mars.scheduler import ResourceActor
from mars.session import new_session
from mars.learn.contrib.xgboost import XGBClassifier, XGBRegressor

try:
    import xgboost
except ImportError:
    xgboost = None


@unittest.skipIf(xgboost is None, 'xgboost not installed')
class Test(unittest.TestCase):
    def setUp(self):
        super(Test, self).setUp()

        n_rows = 1000
        n_columns = 10
        chunk_size = 20
        rs = mt.random.RandomState(0)
        self.X = rs.rand(n_rows, n_columns, chunk_size=chunk_size)
        self.y = rs.rand(n_rows, chunk_size=chunk_size)

        self.n_workers = 2
        self.start_distributed_env(n_workers=self.n_workers)

    def start_distributed_env(self, n_workers=2):
        scheduler_port = self.scheduler_port = str(get_next_port())
        self.proc_workers = []
        for _ in range(n_workers):
            worker_port = str(get_next_port())
            proc_worker = subprocess.Popen([sys.executable, '-m', 'mars.worker',
                                            '-a', '127.0.0.1',
                                            '-p', worker_port,
                                            '--cpu-procs', '2',
                                            '--cache-mem', '10m',
                                            '--schedulers', '127.0.0.1:' + scheduler_port,
                                            '--log-level', 'debug',
                                            '--log-format', 'WOR %(asctime)-15s %(message)s',
                                            '--ignore-avail-mem'])

            self.proc_workers.append(proc_worker)

        proc_scheduler = subprocess.Popen([sys.executable, '-m', 'mars.scheduler',
                                           '--nproc', '1',
                                           '-H', '127.0.0.1',
                                           '-p', scheduler_port,
                                           '-Dscheduler.default_cpu_usage=0',
                                           '--log-level', 'debug',
                                           '--log-format', 'SCH %(asctime)-15s %(message)s'])
        self.proc_scheduler = proc_scheduler

        self.wait_scheduler_worker_start()

        web_port = self.web_port = str(get_next_port())
        proc_web = subprocess.Popen([sys.executable, '-m', 'mars.web',
                                    '-H', '127.0.0.1',
                                     '--log-level', 'debug',
                                     '--log-format', 'WEB %(asctime)-15s %(message)s',
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
        while resource_ref.get_worker_count() < self.n_workers:
            if self.proc_scheduler.poll() is not None:
                raise SystemError('Scheduler not started. exit code %s' % self.proc_scheduler.poll())
            for proc_worker in self.proc_workers:
                if proc_worker.poll() is not None:
                    raise SystemError('Worker not started. exit code %s' % self.proc_worker.poll())
            if time.time() - check_time > 20:
                raise SystemError('Check meta_timestamp timeout')

            time.sleep(0.1)

        gevent.hub.Hub.NOT_ERROR = old_not_errors

    def tearDown(self):
        procs = [self.proc_web, self.proc_scheduler] + self.proc_workers
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

    def testDistributedXGBClassifier(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        timeout = 120 if 'CI' in os.environ else -1
        with new_session(service_ep) as sess:
            run_kwargs = {'timeout': timeout}

            X, y = self.X, self.y
            y = (y * 10).astype(mt.int32)
            classifier = XGBClassifier(verbosity=1, n_estimators=2)
            classifier.fit(X, y, eval_set=[(X, y)], session=sess, run_kwargs=run_kwargs)
            prediction = classifier.predict(X, session=sess, run_kwargs=run_kwargs)

            self.assertEqual(prediction.ndim, 1)
            self.assertEqual(prediction.shape[0], len(self.X))

            history = classifier.evals_result()

            self.assertIsInstance(prediction, mt.Tensor)
            self.assertIsInstance(history, dict)

            self.assertEqual(list(history)[0], 'validation_0')
            self.assertEqual(list(history['validation_0'])[0], 'merror')
            self.assertEqual(len(history['validation_0']), 1)
            self.assertEqual(len(history['validation_0']['merror']), 2)
