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
import logging
import signal
import subprocess
import sys
import time
import unittest

import requests

from mars.actors import new_client
from mars.scheduler import ResourceActor
from mars.utils import get_next_port

logger = logging.getLogger()


class ProcessRequirementUnmetError(RuntimeError):
    pass


class LearnIntegrationTestBase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.n_workers = 2
        asyncio.get_event_loop().run_until_complete(
            self.start_distributed_env(n_workers=self.n_workers))

    async def start_distributed_env(self, *args, **kwargs):
        fail_count = 0
        while True:
            try:
                await self._start_distributed_env(*args, **kwargs)
                break
            except ProcessRequirementUnmetError:
                fail_count += 1
                if fail_count >= 3:
                    raise
                logger.error('Failed to start service, retrying')
                self.terminate_processes()

    async def _start_distributed_env(self, n_workers=2):
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

        await self.wait_scheduler_worker_start()

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
                raise ProcessRequirementUnmetError('Wait for service start timeout')
            try:
                resp = requests.get(service_ep + '/api', timeout=1)
            except (requests.ConnectionError, requests.Timeout):
                await asyncio.sleep(0.1)
                continue
            if resp.status_code >= 400:
                await asyncio.sleep(0.1)
                continue
            break

    async def wait_scheduler_worker_start(self):
        actor_client = new_client()
        await asyncio.sleep(1)
        check_time = time.time()
        while True:
            try:
                resource_ref = actor_client.actor_ref(
                    ResourceActor.default_uid(), address='127.0.0.1:' + self.scheduler_port)
                if await actor_client.has_actor(resource_ref):
                    break
                else:
                    raise ProcessRequirementUnmetError('Check meta_timestamp timeout')
            except:  # noqa: E722
                if time.time() - check_time > 10:
                    raise
                await asyncio.sleep(0.1)

        check_time = time.time()
        while await resource_ref.get_worker_count() < self.n_workers:
            if self.proc_scheduler.poll() is not None:
                raise ProcessRequirementUnmetError(
                    'Scheduler not started. exit code %s' % self.proc_scheduler.poll())
            for proc_worker in self.proc_workers:
                if proc_worker.poll() is not None:
                    raise ProcessRequirementUnmetError(
                        'Worker not started. exit code %s' % self.proc_worker.poll())
            if time.time() - check_time > 20:
                raise ProcessRequirementUnmetError('Check meta_timestamp timeout')

            await asyncio.sleep(0.1)

    def terminate_processes(self):
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

    def tearDown(self):
        self.terminate_processes()
