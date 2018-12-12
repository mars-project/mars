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

import os
import logging
import time

from pyarrow import plasma

from ..config import options
from .. import resource, kvstore
from ..utils import readable_size
from ..compat import six
from ..cluster_info import ClusterInfoActor
from .status import StatusActor
from .quota import QuotaActor, MemQuotaActor
from .chunkholder import ChunkHolderActor
from .dispatcher import DispatchActor
from .execution import ExecutionActor
from .calc import CpuCalcActor
from .transfer import ReceiverActor, SenderActor
from .prochelper import ProcessHelperActor
from .transfer import ResultSenderActor
from .spill import SpillActor


logger = logging.getLogger(__name__)


class WorkerService(object):
    service_logger = logger

    def __init__(self):
        self._plasma_store = None

        self._chunk_holder_ref = None
        self._mem_quota_ref = None
        self._dispatch_ref = None
        self._status_ref = None
        self._execution_ref = None

        self._cluster_info_ref = None
        self._cpu_calc_actors = []
        self._sender_actors = []
        self._receiver_actors = []
        self._spill_actors = []
        self._process_helper_actors = []
        self._result_sender_ref = None

    def start_plasma(self, mem_limit, one_mapped_file=False):
        self._plasma_store = plasma.start_plasma_store(
            int(mem_limit), use_one_memory_mapped_file=one_mapped_file
        )
        options.worker.plasma_socket, _ = self._plasma_store.__enter__()

    @classmethod
    def _calc_soft_memory_limit(cls):
        """
        Calc memory limit for MemQuotaActor given configured percentage. Cache size and memory
        already been used will be excluded from calculated size.
        """
        mem_status = resource.virtual_memory()

        phy_soft_limit = options.worker.physical_memory_limit_soft
        cache_limit = options.worker.cache_memory_limit
        actual_used = mem_status.total - mem_status.available
        quota_soft_limit = phy_soft_limit - cache_limit - actual_used

        if quota_soft_limit < 512 * 1024 ** 2:
            raise MemoryError('Memory not sufficient. soft_limit=%s, cache_limit=%s, used=%s'
                              % tuple(readable_size(k) for k in (phy_soft_limit, cache_limit, actual_used)))

        cls.service_logger.info('Setting soft limit to %s.', readable_size(quota_soft_limit))
        return quota_soft_limit

    def start(self, endpoint, schedulers, pool, ignore_avail_mem=False, create_cluster_info=True):
        if schedulers:
            if isinstance(schedulers, six.string_types):
                schedulers = [schedulers]
            service_discover_addr = None
        else:
            schedulers = None
            service_discover_addr = options.kv_store

        # create ClusterInfoActor
        self._cluster_info_ref = pool.create_actor(ClusterInfoActor, schedulers=schedulers,
                                                   service_discover_addr=service_discover_addr,
                                                   uid=ClusterInfoActor.default_name())
        # create StatusActor
        port_str = endpoint.rsplit(':', 1)[-1]
        self._status_ref = pool.create_actor(
            StatusActor, options.worker.advertise_addr + ':' + port_str,
            uid=StatusActor.default_name())

        if ignore_avail_mem:
            # start a QuotaActor instead of MemQuotaActor to avoid memory size detection
            # for debug purpose only, DON'T USE IN PRODUCTION
            self._mem_quota_ref = pool.create_actor(
                QuotaActor, options.worker.physical_memory_limit_soft, uid=MemQuotaActor.default_name())
        else:
            self._mem_quota_ref = pool.create_actor(
                MemQuotaActor, self._calc_soft_memory_limit(),
                options.worker.physical_memory_limit_hard, uid=MemQuotaActor.default_name())

        # create ChunkHolderActor
        self._chunk_holder_ref = pool.create_actor(
            ChunkHolderActor, options.worker.cache_memory_limit, uid=ChunkHolderActor.default_name())
        # create DispatchActor
        self._dispatch_ref = pool.create_actor(DispatchActor, uid=DispatchActor.default_name())
        # create ExecutionActor
        self._execution_ref = pool.create_actor(ExecutionActor, uid=ExecutionActor.default_name())

        # create CpuCalcActor
        for cpu_id in range(options.worker.cpu_process_count):
            uid = 'w:%d:mars-calc-%d-%d' % (cpu_id + 1, os.getpid(), cpu_id)
            actor = pool.create_actor(CpuCalcActor, uid=uid)
            self._cpu_calc_actors.append(actor)

        # create SenderActor and ReceiverActor
        start_pid = 1 + options.worker.cpu_process_count
        for sender_id in range(options.worker.io_process_count):
            uid = 'w:%d:mars-sender-%d-%d' % (start_pid + sender_id, os.getpid(), sender_id)
            actor = pool.create_actor(SenderActor, uid=uid)
            self._sender_actors.append(actor)
        for receiver_id in range(2 * options.worker.io_process_count):
            uid = 'w:%d:mars-receiver-%d-%d' % (start_pid + receiver_id // 2, os.getpid(), receiver_id)
            actor = pool.create_actor(ReceiverActor, uid=uid)
            self._receiver_actors.append(actor)

        # create ProcessHelperActor
        for proc_id in range(pool.cluster_info.n_process):
            uid = 'w:%d:mars-process-helper-%d-%d' % (proc_id, os.getpid(), proc_id)
            actor = pool.create_actor(ProcessHelperActor, uid=uid)
            self._process_helper_actors.append(actor)

        # create ResultSenderActor
        self._result_sender_ref = pool.create_actor(ResultSenderActor, uid=ResultSenderActor.default_name())

        # create SpillActor
        start_pid = 1 + options.worker.cpu_process_count + options.worker.io_process_count
        if options.worker.spill_directory:
            for spill_id in range(len(options.worker.spill_directory) * 2):
                uid = 'w:%d:mars-spill-%d-%d' % (start_pid, os.getpid(), spill_id)
                actor = pool.create_actor(SpillActor, uid=uid)
                self._spill_actors.append(actor)

        kv_store = kvstore.get(options.kv_store)
        if isinstance(kv_store, kvstore.EtcdKVStore):
            kv_store.write('/workers/meta_timestamp', str(int(time.time())))

    def stop(self):
        if self._result_sender_ref:
            self._result_sender_ref.destroy()
        if self._status_ref:
            self._status_ref.destroy()
        if self._chunk_holder_ref:
            self._chunk_holder_ref.destroy()
        if self._dispatch_ref:
            self._dispatch_ref.destroy()
        if self._execution_ref:
            self._execution_ref.destroy()

        for actor in (self._cpu_calc_actors + self._sender_actors
                      + self._receiver_actors + self._spill_actors + self._process_helper_actors):
            actor.destroy()

        self._plasma_store.__exit__(None, None, None)
