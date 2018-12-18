#!/usr/bin/env python
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

import multiprocessing
import signal
import os

from ...utils import get_next_port
from ...resource import cpu_count
from ...scheduler.service import SchedulerService
from ...worker.service import WorkerService
from ...actors import create_actor_pool
from ...session import new_session
from ...compat import six
from ...lib import gipc
from ...config import options
from .distributor import gen_distributor


class LocalDistributedCluster(object):

    # at least 2 process are required by scheduler and worker
    MIN_SCHEDULER_N_PROCESS = 2
    MIN_WORKER_N_PROCESS = 2

    def __init__(self, endpoint, n_process=None,
                 scheduler_n_process=None, worker_n_process=None,
                 shared_memory=None):
        self._endpoint = endpoint

        self._started = False
        self._stopped = False

        if shared_memory is not None:
            options.worker.cache_memory_limit = shared_memory

        self._pool = None
        self._scheduler_service = SchedulerService()
        self._worker_service = WorkerService()

        self._scheduler_n_process, self._worker_n_process = \
            self._calc_scheduler_worker_n_process(n_process,
                                                  scheduler_n_process,
                                                  worker_n_process)

    @property
    def pool(self):
        return self._pool

    @classmethod
    def _calc_scheduler_worker_n_process(cls, n_process, scheduler_n_process, worker_n_process,
                                         calc_cpu_count=cpu_count):
        n_scheduler, n_worker = scheduler_n_process, worker_n_process

        if n_scheduler is None and n_worker is None:
            n_scheduler = cls.MIN_SCHEDULER_N_PROCESS
            n_process = n_process if n_process is not None else calc_cpu_count() + n_scheduler
            n_worker = max(n_process - n_scheduler, cls.MIN_WORKER_N_PROCESS)
        elif n_scheduler is None or n_worker is None:
            # one of scheduler and worker n_process provided
            if n_scheduler is None:
                n_process = n_process if n_process is not None else calc_cpu_count()
                n_scheduler = max(n_process - n_worker, cls.MIN_SCHEDULER_N_PROCESS)
            else:
                assert n_worker is None
                n_process = n_process if n_process is not None else calc_cpu_count() + n_scheduler
                n_worker = max(n_process - n_scheduler, cls.MIN_WORKER_N_PROCESS)

        return n_scheduler, n_worker

    def _make_sure_scheduler_ready(self):
        while True:
            workers_meta = self._scheduler_service._resource_ref.get_workers_meta()
            if not workers_meta:
                # wait for worker to report status
                self._pool.sleep(.5)
            else:
                break

    def start_service(self):
        if self._started:
            return
        self._started = True

        # start plasma
        self._worker_service.start_plasma(self._worker_service.cache_memory_limit())

        # start actor pool
        n_process = self._scheduler_n_process + self._worker_n_process
        distributor = gen_distributor(self._scheduler_n_process, self._worker_n_process)
        self._pool = create_actor_pool(self._endpoint, n_process, distributor=distributor)

        # start scheduler first
        self._scheduler_service.start(self._endpoint, self._pool)

        # start worker next
        self._worker_service.start_local(self._endpoint, self._pool, self._scheduler_n_process)

        # make sure scheduler is ready
        self._make_sure_scheduler_ready()

    def stop_service(self):
        if self._stopped:
            return

        self._stopped = True
        try:
            self._scheduler_service.stop(self._pool)
            self._worker_service.stop()
        finally:
            self._pool.stop()

    def serve_forever(self):
        try:
            self._pool.join()
        finally:
            self.stop_service()

    def __enter__(self):
        self.start_service()
        return self

    def __exit__(self, *_):
        self.stop_service()


def gen_endpoint(address):
    port = None
    tries = 5  # retry for 5 times

    for i in range(tries):
        try:
            port = get_next_port()
            break
        except SystemError:
            if i < tries - 1:
                continue
            raise

    return '{0}:{1}'.format(address, port)


def _start_cluster(endpoint, event, n_process=None, shared_memory=None, **kw):
    cluster = LocalDistributedCluster(endpoint, n_process=n_process,
                                      shared_memory=shared_memory, **kw)
    cluster.start_service()
    event.set()
    try:
        cluster.serve_forever()
    finally:
        cluster.stop_service()


def _start_web(scheduler_address, ui_port, event):
    import gevent.monkey
    gevent.monkey.patch_all(thread=False)

    from ...web import MarsWeb

    web = MarsWeb(ui_port, scheduler_address)
    try:
        web.start(event=event, block=True)
    finally:
        web.stop()


class LocalDistributedClusterClient(object):
    def __init__(self, endpoint, web_endpoint, cluster_process, web_process):
        self._cluster_process = cluster_process
        self._web_process = web_process
        self._endpoint = endpoint
        self._web_endpoint = web_endpoint
        self._session = new_session(endpoint).as_default()

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def web_endpoint(self):
        return self._web_endpoint

    @property
    def session(self):
        return self._session

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self):
        if self._cluster_process.is_alive():
            os.kill(self._cluster_process.pid, signal.SIGINT)
            self._cluster_process.join(3)
            if self._cluster_process.is_alive():
                self._cluster_process.terminate()
        if self._web_process is not None and self._web_process.is_alive():
            os.kill(self._web_process.pid, signal.SIGINT)
            self._web_process.join(3)
            if self._web_process.is_alive():
                self._web_process.terminate()


def new_cluster(address='0.0.0.0', web=False, n_process=None, shared_memory=None, **kw):
    endpoint = gen_endpoint(address)
    web_endpoint = None
    if web is True:
        web_endpoint = gen_endpoint('0.0.0.0')
    elif isinstance(web, six.string_types):
        if ':' in web:
            web_endpoint = web
        else:
            web_endpoint = gen_endpoint(web)

    event = multiprocessing.Event()
    kw['n_process'] = n_process
    kw['shared_memory'] = shared_memory or options.worker.cache_memory_limit or '20%'
    process = gipc.start_process(_start_cluster, args=(endpoint, event), kwargs=kw)

    while True:
        event.wait(5)
        if not event.is_set():
            # service not started yet
            continue
        if not process.is_alive():
            raise SystemError('New local cluster failed')
        else:
            break

    web_process = None
    if web_endpoint:
        web_event = multiprocessing.Event()
        ui_port = int(web_endpoint.rsplit(':', 1)[1])
        web_process = gipc.start_process(_start_web, args=(endpoint, ui_port, web_event), daemon=True)

        while True:
            web_event.wait(5)
            if not web_event.is_set():
                # web not started yet
                continue
            if not web_process.is_alive():
                raise SystemError('New web interface failed')
            else:
                break

    return LocalDistributedClusterClient(endpoint, web_endpoint, process, web_process)
