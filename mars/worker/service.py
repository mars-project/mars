# -*- coding: utf-8 -*-
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

import logging
import os
import sys

from ..config import options
from .. import resource
from ..utils import parse_readable_size, readable_size
from .status import StatusActor
from .quota import QuotaActor, MemQuotaActor
from .dispatcher import DispatchActor
from .events import EventsActor
from .execution import ExecutionActor
from .calc import CpuCalcActor, CudaCalcActor
from .transfer import SenderActor, ReceiverManagerActor, ReceiverWorkerActor, ResultSenderActor
from .prochelper import ProcessHelperActor
from .storage import IORunnerActor, StorageManagerActor, SharedHolderActor, \
    InProcHolderActor, CudaHolderActor
from .utils import WorkerClusterInfoActor


logger = logging.getLogger(__name__)


class WorkerService(object):
    def __init__(self, **kwargs):
        self._plasma_store = None

        self._storage_manager_ref = None
        self._shared_holder_ref = None
        self._task_queue_ref = None
        self._mem_quota_ref = None
        self._dispatch_ref = None
        self._events_ref = None
        self._status_ref = None
        self._execution_ref = None
        self._daemon_ref = None
        self._receiver_manager_ref = None

        self._cluster_info_ref = None
        self._cpu_calc_actors = []
        self._inproc_holder_actors = []
        self._inproc_io_runner_actors = []
        self._cuda_calc_actors = []
        self._cuda_holder_actors = []
        self._sender_actors = []
        self._receiver_actors = []
        self._spill_actors = []
        self._process_helper_actors = []
        self._result_sender_ref = None

        self._advertise_addr = kwargs.pop('advertise_addr', None)

        cuda_devices = kwargs.pop('cuda_devices', None) or os.environ.get('CUDA_VISIBLE_DEVICES')
        if not cuda_devices:
            self._n_cuda_process = 0
        else:
            cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in cuda_devices)
            if cuda_devices:
                logger.info('Started Mars worker with CUDA cards %s', cuda_devices)
            self._n_cuda_process = resource.cuda_count()

        self._n_cpu_process = int(kwargs.pop('n_cpu_process', None) or resource.cpu_count())
        self._n_net_process = int(kwargs.pop('n_net_process', None) or '4')

        self._spill_dirs = kwargs.pop('spill_dirs', None)
        if self._spill_dirs:
            if isinstance(self._spill_dirs, str):
                from .utils import parse_spill_dirs
                self._spill_dirs = options.worker.spill_directory = parse_spill_dirs(self._spill_dirs)
            else:
                options.worker.spill_directory = self._spill_dirs
        else:
            self._spill_dirs = options.worker.spill_directory = []

        options.worker.disk_compression = kwargs.pop('disk_compression', None) or \
            options.worker.disk_compression
        options.worker.transfer_compression = kwargs.pop('transfer_compression', None) or \
            options.worker.transfer_compression
        options.worker.io_parallel_num = kwargs.pop('io_parallel_num', None) or False
        options.worker.recover_dead_process = not (kwargs.pop('disable_proc_recover', None) or False)
        options.worker.write_shuffle_to_disk = kwargs.pop('write_shuffle_to_disk', None) or False

        self._total_mem = kwargs.pop('total_mem', None)
        self._cache_mem_limit = kwargs.pop('cache_mem_limit', None)
        self._soft_mem_limit = kwargs.pop('soft_mem_limit', None) or '80%'
        self._hard_mem_limit = kwargs.pop('hard_mem_limit', None) or '90%'
        self._ignore_avail_mem = kwargs.pop('ignore_avail_mem', None) or False
        self._min_mem_size = kwargs.pop('min_mem_size', None) or 128 * 1024 ** 2

        self._plasma_dir = kwargs.pop('plasma_dir', None)
        self._use_ext_plasma_dir = kwargs.pop('use_ext_plasma_dir', None) or False

        self._soft_quota_limit = self._soft_mem_limit

        self._calc_memory_limits()

        if kwargs:  # pragma: no cover
            raise TypeError(f'Keyword arguments {kwargs!r} cannot be recognized.')

    @property
    def n_process(self):
        return 1 + self._n_cpu_process + self._n_cuda_process + self._n_net_process \
               + (1 if self._spill_dirs else 0)

    def _get_plasma_limit(self):
        if sys.platform == 'win32':  # pragma: no cover
            return
        elif sys.platform == 'darwin':
            default_plasma_dir = '/tmp'
        else:
            default_plasma_dir = '/dev/shm'

        fd = os.open(self._plasma_dir or default_plasma_dir, os.O_RDONLY)
        stats = os.fstatvfs(fd)
        os.close(fd)
        size = stats.f_bsize * stats.f_bavail
        # keep some safety margin for allocator fragmentation
        return 8 * size // 10

    def _calc_memory_limits(self):
        def _calc_size_limit(limit_str, total_size):
            if limit_str is None:
                return None
            if isinstance(limit_str, int):
                return limit_str
            mem_limit, is_percent = parse_readable_size(limit_str)
            if is_percent:
                return int(total_size * mem_limit)
            else:
                return int(mem_limit)

        mem_stats = resource.virtual_memory()

        if self._total_mem:
            self._total_mem = _calc_size_limit(self._total_mem, mem_stats.total)
        else:
            self._total_mem = mem_stats.total

        self._min_mem_size = _calc_size_limit(self._min_mem_size, self._total_mem)
        self._hard_mem_limit = _calc_size_limit(self._hard_mem_limit, self._total_mem)

        self._cache_mem_limit = _calc_size_limit(self._cache_mem_limit, self._total_mem)
        if self._cache_mem_limit is None:
            self._cache_mem_limit = mem_stats.free // 2

        plasma_limit = self._get_plasma_limit()
        if plasma_limit is not None:
            self._cache_mem_limit = min(plasma_limit, self._cache_mem_limit)

        self._soft_mem_limit = _calc_size_limit(self._soft_mem_limit, self._total_mem)
        actual_used = self._total_mem - mem_stats.available
        if self._ignore_avail_mem:
            self._soft_quota_limit = self._soft_mem_limit
        else:
            used_cache_size = 0 if self._use_ext_plasma_dir else self._cache_mem_limit
            self._soft_quota_limit = self._soft_mem_limit - used_cache_size - actual_used
            if self._soft_quota_limit < self._min_mem_size:
                raise MemoryError(
                    f'Memory not enough. soft_limit={readable_size(self._soft_mem_limit)}, '
                    f'cache_limit={readable_size(self._cache_mem_limit)}, '
                    f'used={readable_size(actual_used)}')

        logger.info('Setting soft limit to %s.', readable_size(self._soft_quota_limit))

    def start_plasma(self):
        from pyarrow import plasma
        self._plasma_store = plasma.start_plasma_store(
            self._cache_mem_limit, plasma_directory=self._plasma_dir)
        options.worker.plasma_socket, _ = self._plasma_store.__enter__()

    def start(self, endpoint, pool, distributed=True, discoverer=None, process_start_index=0):
        # create plasma key mapper
        from .storage import PlasmaKeyMapActor
        pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())

        # create vineyard key mapper
        if options.vineyard.socket:  # pragma: no cover
            from .storage import VineyardKeyMapActor
            pool.create_actor(VineyardKeyMapActor, uid=VineyardKeyMapActor.default_uid())

        # create WorkerClusterInfoActor
        self._cluster_info_ref = pool.create_actor(
            WorkerClusterInfoActor, discoverer, distributed=distributed,
            uid=WorkerClusterInfoActor.default_uid())

        if distributed:
            # create process daemon
            from .daemon import WorkerDaemonActor
            actor_holder = self._daemon_ref = pool.create_actor(
                WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())

            # create StatusActor
            if ':' not in self._advertise_addr:
                self._advertise_addr += ':' + endpoint.rsplit(':', 1)[-1]

            self._status_ref = pool.create_actor(
                StatusActor, self._advertise_addr, uid=StatusActor.default_uid())
        else:
            # create StatusActor
            self._status_ref = pool.create_actor(
                StatusActor, endpoint, with_gpu=self._n_cuda_process > 0, uid=StatusActor.default_uid())

            actor_holder = pool

        if self._ignore_avail_mem:
            # start a QuotaActor instead of MemQuotaActor to avoid memory size detection
            # for debug purpose only, DON'T USE IN PRODUCTION
            self._mem_quota_ref = pool.create_actor(
                QuotaActor, self._soft_mem_limit, uid=MemQuotaActor.default_uid())
        else:
            self._mem_quota_ref = pool.create_actor(
                MemQuotaActor, self._soft_quota_limit, self._hard_mem_limit, uid=MemQuotaActor.default_uid())

        # create StorageManagerActor
        self._storage_manager_ref = pool.create_actor(
            StorageManagerActor, uid=StorageManagerActor.default_uid())
        # create SharedHolderActor
        self._shared_holder_ref = pool.create_actor(
            SharedHolderActor, self._cache_mem_limit, uid=SharedHolderActor.default_uid())
        # create DispatchActor
        self._dispatch_ref = pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
        # create EventsActor
        self._events_ref = pool.create_actor(EventsActor, uid=EventsActor.default_uid())
        # create ReceiverNotifierActor
        self._receiver_manager_ref = pool.create_actor(ReceiverManagerActor, uid=ReceiverManagerActor.default_uid())
        # create ExecutionActor
        self._execution_ref = pool.create_actor(ExecutionActor, uid=ExecutionActor.default_uid())

        # create CpuCalcActor and InProcHolderActor
        if not distributed:
            self._n_cpu_process = pool.cluster_info.n_process - 1 - process_start_index

        for cpu_id in range(self._n_cpu_process):
            uid = f'w:{cpu_id + 1}:mars-cpu-calc'
            actor = actor_holder.create_actor(CpuCalcActor, uid=uid)
            self._cpu_calc_actors.append(actor)

            uid = f'w:{cpu_id + 1}:mars-inproc-holder'
            actor = actor_holder.create_actor(InProcHolderActor, uid=uid)
            self._inproc_holder_actors.append(actor)

            actor = actor_holder.create_actor(
                IORunnerActor, dispatched=False, uid=IORunnerActor.gen_uid(cpu_id + 1))
            self._inproc_io_runner_actors.append(actor)

        start_pid = 1 + self._n_cpu_process

        stats = resource.cuda_card_stats() if self._n_cuda_process else []
        for cuda_id, stat in enumerate(stats):
            for thread_no in range(options.worker.cuda_thread_num):
                uid = f'w:{start_pid + cuda_id}:mars-cuda-calc-{cuda_id}-{thread_no}'
                actor = actor_holder.create_actor(CudaCalcActor, uid=uid)
                self._cuda_calc_actors.append(actor)

            uid = f'w:{start_pid + cuda_id}:mars-cuda-holder-{cuda_id}'
            actor = actor_holder.create_actor(
                CudaHolderActor, stat.fb_mem_info.total, device_id=stat.index, uid=uid)
            self._cuda_holder_actors.append(actor)

            actor = actor_holder.create_actor(
                IORunnerActor, dispatched=False, uid=IORunnerActor.gen_uid(start_pid + cuda_id))
            self._inproc_io_runner_actors.append(actor)

        start_pid += self._n_cuda_process

        if distributed:
            # create SenderActor and ReceiverActor
            for sender_id in range(self._n_net_process):
                uid = f'w:{start_pid + sender_id}:mars-sender-{sender_id}'
                actor = actor_holder.create_actor(SenderActor, uid=uid)
                self._sender_actors.append(actor)

        # Mutable requires ReceiverActor (with ClusterSession)
        for receiver_id in range(2 * self._n_net_process):
            uid = f'w:{start_pid + receiver_id // 2}:mars-receiver-{receiver_id}'
            actor = actor_holder.create_actor(ReceiverWorkerActor, uid=uid)
            self._receiver_actors.append(actor)

        # create ProcessHelperActor
        for proc_id in range(pool.cluster_info.n_process - process_start_index):
            uid = f'w:{proc_id}:mars-process-helper'
            actor = actor_holder.create_actor(ProcessHelperActor, uid=uid)
            self._process_helper_actors.append(actor)

        # create ResultSenderActor
        self._result_sender_ref = pool.create_actor(ResultSenderActor, uid=ResultSenderActor.default_uid())

        # create SpillActor
        start_pid = pool.cluster_info.n_process - 1
        if options.worker.spill_directory:
            for spill_id in range(len(options.worker.spill_directory)):
                uid = f'w:{start_pid}:mars-global-io-runner-{spill_id}'
                actor = actor_holder.create_actor(IORunnerActor, uid=uid)
                self._spill_actors.append(actor)

        # worker can be registered when everything is ready
        self._status_ref.enable_status_upload(_tell=True)

    def handle_process_down(self, pool, proc_indices):
        proc_id_to_pid = dict((proc_id, pool.processes[proc_id].pid) for proc_id in proc_indices)
        exit_codes = [pool.processes[proc_id].exitcode for proc_id in proc_indices]
        logger.warning('Process %r halt, exitcodes=%r. Trying to recover.', proc_id_to_pid, exit_codes)
        for proc_id in proc_indices:
            pool.restart_process(proc_id)
        self._daemon_ref.handle_process_down(proc_indices)

    def stop(self):
        try:
            destroy_futures = []
            for actor in (self._cpu_calc_actors + self._sender_actors + self._inproc_holder_actors
                          + self._inproc_io_runner_actors + self._cuda_calc_actors
                          + self._cuda_holder_actors + self._receiver_actors + self._spill_actors
                          + self._process_helper_actors):
                if actor and actor.ctx:
                    destroy_futures.append(actor.destroy(wait=False))

            if self._result_sender_ref:
                destroy_futures.append(self._result_sender_ref.destroy(wait=False))
            if self._status_ref:
                destroy_futures.append(self._status_ref.destroy(wait=False))
            if self._shared_holder_ref:
                destroy_futures.append(self._shared_holder_ref.destroy(wait=False))
            if self._storage_manager_ref:
                destroy_futures.append(self._storage_manager_ref.destroy(wait=False))
            if self._events_ref:
                destroy_futures.append(self._events_ref.destroy(wait=False))
            if self._dispatch_ref:
                destroy_futures.append(self._dispatch_ref.destroy(wait=False))
            if self._execution_ref:
                destroy_futures.append(self._execution_ref.destroy(wait=False))
            [f.result(5) for f in destroy_futures]
        finally:
            self._plasma_store.__exit__(None, None, None)
