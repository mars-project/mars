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
import shutil
import sys
import tempfile

from .. import resource
from ..config import options
from ..utils import readable_size, calc_size_by_str
from .calc import CpuCalcActor, CudaCalcActor
from .custom_log import CustomLogFetchActor
from .dispatcher import DispatchActor
from .events import EventsActor
from .execution import ExecutionActor
from .prochelper import ProcessHelperActor
from .quota import QuotaActor, MemQuotaActor
from .status import StatusActor
from .storage import IORunnerActor, StorageManagerActor, SharedHolderActor, \
    InProcHolderActor, CudaHolderActor, DiskFileMergerActor
from .transfer import SenderActor, ReceiverManagerActor, ReceiverWorkerActor, ResultSenderActor
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
        self._custom_log_fetch_actors = []
        self._result_sender_ref = None
        self._file_merger_ref = None

        self._distributed = distributed = kwargs.pop('distributed', True)
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

        options.worker.disk_compression = kwargs.pop('disk_compression', None) \
            or options.worker.disk_compression
        options.worker.transfer_compression = kwargs.pop('transfer_compression', None) \
            or options.worker.transfer_compression
        options.worker.io_parallel_num = kwargs.pop('io_parallel_num', None) \
            or options.worker.io_parallel_num
        options.worker.recover_dead_process = not (kwargs.pop('disable_proc_recover', None)
                                                   or not options.worker.recover_dead_process)
        options.worker.write_shuffle_to_disk = kwargs.pop('write_shuffle_to_disk', None) \
            or options.worker.write_shuffle_to_disk

        min_cache_mem_size = kwargs.pop('min_cache_mem_size', None)
        min_cache_mem_size = min_cache_mem_size if min_cache_mem_size is not None \
            else options.worker.min_cache_mem_size
        options.worker.min_cache_mem_size = min_cache_mem_size

        plasma_limit = kwargs.pop('plasma_limit', None)
        plasma_limit = plasma_limit if plasma_limit is not None \
            else options.worker.plasma_limit
        options.worker.plasma_limit = plasma_limit

        if distributed and options.custom_log_dir is None:
            # gen custom_log_dir for distributed only
            options.custom_log_dir = tempfile.mkdtemp(prefix='mars-custom-log')
            self._clear_custom_log_dir = True
        else:
            self._clear_custom_log_dir = False

        self._total_mem = kwargs.pop('total_mem', None)
        if self._total_mem:
            os.environ['MARS_MEMORY_TOTAL'] = str(self._total_mem)
        self._cache_mem_size = kwargs.pop('cache_mem_size', None)
        self._cache_mem_scale = float(kwargs.pop('cache_mem_scale', None) or 1)
        self._soft_mem_limit = kwargs.pop('soft_mem_limit', None) or '80%'
        self._hard_mem_limit = kwargs.pop('hard_mem_limit', None) or '90%'
        self._ignore_avail_mem = kwargs.pop('ignore_avail_mem', None) or False
        self._min_mem_size = kwargs.pop('min_mem_size', None) or 128 * 1024 ** 2

        if sys.platform == 'win32':  # pragma: no cover
            raise NotImplementedError('Mars worker cannot start under Windows')
        elif sys.platform == 'darwin':
            default_plasma_dir = '/tmp'
        else:
            default_plasma_dir = '/dev/shm'
        options.worker.plasma_dir = kwargs.pop('plasma_dir', None) \
            or options.worker.plasma_dir or default_plasma_dir

        self._use_ext_plasma_dir = kwargs.pop('use_ext_plasma_dir', None) or False

        self._soft_quota_limit = self._soft_mem_limit

        self._calc_memory_limits()

        if kwargs:  # pragma: no cover
            raise TypeError(f'Keyword arguments {kwargs!r} cannot be recognized.')

    @property
    def n_process(self):
        return 1 + self._n_cpu_process + self._n_cuda_process + self._n_net_process \
               + (1 if self._spill_dirs else 0)

    @staticmethod
    def _get_plasma_size():
        fd = os.open(options.worker.plasma_dir, os.O_RDONLY)
        stats = os.fstatvfs(fd)
        os.close(fd)
        size = stats.f_bsize * stats.f_bavail
        # keep some safety margin for allocator fragmentation
        return 8 * size // 10

    def _calc_memory_limits(self):
        mem_stats = resource.virtual_memory()

        if self._total_mem:
            self._total_mem = calc_size_by_str(self._total_mem, mem_stats.total)
        else:
            self._total_mem = mem_stats.total

        self._min_mem_size = calc_size_by_str(self._min_mem_size, self._total_mem)
        self._hard_mem_limit = calc_size_by_str(self._hard_mem_limit, self._total_mem)

        raw_cache_mem_size = self._cache_mem_size = \
            calc_size_by_str(self._cache_mem_size, self._total_mem)
        if self._cache_mem_size is None:
            raw_cache_mem_size = self._cache_mem_size = mem_stats.free // 2

        plasma_size = self._get_plasma_size()
        if plasma_size is not None:
            self._cache_mem_size = min(plasma_size, self._cache_mem_size)
        self._cache_mem_size = int(self._cache_mem_size * self._cache_mem_scale)

        self._soft_mem_limit = calc_size_by_str(self._soft_mem_limit, self._total_mem)
        actual_used = self._total_mem - mem_stats.available
        if self._ignore_avail_mem:
            self._soft_quota_limit = self._soft_mem_limit
        else:
            used_cache_size = 0 if self._use_ext_plasma_dir else self._cache_mem_size
            self._soft_quota_limit = self._soft_mem_limit - used_cache_size - actual_used
            if self._soft_quota_limit < self._min_mem_size:
                raise MemoryError(
                    f'Memory not enough. soft_limit={readable_size(self._soft_mem_limit)}, '
                    f'cache_limit={readable_size(self._cache_mem_size)}, '
                    f'used={readable_size(actual_used)}')

        if options.worker.min_cache_mem_size:
            min_cache_mem_size = calc_size_by_str(options.worker.min_cache_mem_size, self._total_mem)
            if min(min_cache_mem_size, raw_cache_mem_size) > self._cache_mem_size:
                raise MemoryError(f'Cache memory size ({self._cache_mem_size}) smaller than '
                                  f'minimal size ({min_cache_mem_size}), worker cannot start')

        logger.info('Setting soft limit to %s.', readable_size(self._soft_quota_limit))

    def start_plasma(self):
        from pyarrow import plasma
        self._plasma_store = plasma.start_plasma_store(
            self._cache_mem_size, plasma_directory=options.worker.plasma_dir)
        options.worker.plasma_socket, _ = self._plasma_store.__enter__()

    def start(self, endpoint, pool, discoverer=None, process_start_index=0):
        distributed = self._distributed
        # create plasma key mapper
        from .storage import PlasmaKeyMapActor
        pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())

        # create WorkerClusterInfoActor
        self._cluster_info_ref = pool.create_actor(
            WorkerClusterInfoActor, discoverer, distributed=distributed,
            uid=WorkerClusterInfoActor.default_uid())

        # create DiskFileMergerActor
        if options.worker.filemerger.enabled and options.worker.spill_directory:
            self._file_merger_ref = pool.create_actor(
                DiskFileMergerActor, uid=DiskFileMergerActor.default_uid())

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
        start_timeout = int(os.environ.get('MARS_SHARED_HOLDER_START_TIMEOUT', None) or 60)
        start_future = pool.create_actor(
            SharedHolderActor, self._cache_mem_size, uid=SharedHolderActor.default_uid(), wait=False)
        self._shared_holder_ref = start_future.result(start_timeout)
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

        if options.custom_log_dir is not None:
            for custom_log_fetch_id in range(self._n_net_process):
                uid = f'w:{start_pid + custom_log_fetch_id}:mars-custom-log-fetch'
                actor = actor_holder.create_actor(CustomLogFetchActor, uid=uid)
                self._custom_log_fetch_actors.append(actor)

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

            if self._clear_custom_log_dir:
                custom_dir = options.custom_log_dir
                shutil.rmtree(custom_dir, ignore_errors=True)
                options.custom_log_dir = None
        finally:
            self._plasma_store.__exit__(None, None, None)
