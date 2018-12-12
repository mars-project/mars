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

import threading

from ...utils import get_next_port
from ...resource import cpu_count
from ...scheduler.service import SchedulerService
from ...worker.service import WorkerService
from ...actors import create_actor_pool
from .distributor import gen_distributor


class LocalDistributedCluster(object):

    # at least 2 process are required by scheduler and worker
    MIN_SCHEDULER_N_PROCESS = 2
    MIN_WORKER_N_PROCESS = 2

    def __init__(self, address, web=False, n_process=None,
                 scheduler_n_process=None, worker_n_process=None):
        if ':' in address:
            self._endpoint = address
        else:
            # if no port provided, will try to generate one
            self._endpoint = self._gen_endpoint(address)

        self._stopped = threading.Event()

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
    def _calc_scheduler_worker_n_process(cls, n_process, scheduler_n_process, worker_n_process):
        n_scheduler, n_worker = scheduler_n_process, worker_n_process

        if n_scheduler is None and n_worker is None:
            n_process = n_process if n_process is not None else cpu_count()

            n_scheduler = cls.MIN_SCHEDULER_N_PROCESS
            n_worker = max(n_process - n_scheduler, cls.MIN_WORKER_N_PROCESS)
        elif n_scheduler is None or n_worker is None:
            # one of scheduler and worker n_process provided
            n_process = n_process if n_process is not None else cpu_count()
            if n_scheduler is None:
                n_scheduler = max(n_process - n_worker, cls.MIN_SCHEDULER_N_PROCESS)
            else:
                assert n_worker is None
                n_worker = max(n_process - n_scheduler, cls.MIN_WORKER_N_PROCESS)

        return n_scheduler, n_worker

    @classmethod
    def _gen_endpoint(cls, address):
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

    def start_service(self):
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

    def stop_service(self):
        if self._stopped.is_set():
            return

        self._stopped.set()
        try:
            self._scheduler_service.stop(self._pool)
            self._worker_service.stop()
        finally:
            self._pool.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop_service()


def new_cluster(address='0.0.0.0', web=False, n_process=None, **kw):
    cluster = LocalDistributedCluster(address, web=web, n_process=n_process, **kw)
    cluster.start_service()
    return cluster
