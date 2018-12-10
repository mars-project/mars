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

import logging
import os
import time

from .. import resource, kvstore
from ..config import options
from ..utils import parse_memory_limit, readable_size
from ..compat import six
from ..base_app import BaseApplication
from ..errors import StartArgumentError
from .distributor import WorkerDistributor

logger = logging.getLogger(__name__)


class WorkerApplication(BaseApplication):
    """
    Main function class of Mars Worker
    """
    service_description = 'Mars Worker'
    service_logger = logger

    def __init__(self):
        super(WorkerApplication, self).__init__()
        self._plasma_helper = None
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

    @staticmethod
    def _calc_size_limit(limit_str, total_size):
        """
        Calculate limitation size when it is represented in percentage or prettified format
        :param limit_str: percentage or prettified format
        :param total_size: total size of the container
        :return: actual size in bytes
        """
        if isinstance(limit_str, int):
            return limit_str
        mem_limit, is_percent = parse_memory_limit(limit_str)
        if is_percent:
            return int(total_size * mem_limit)
        else:
            return int(mem_limit)

    def config_args(self, parser):
        parser.add_argument('-s', '--schedulers', help='scheduler addresses')
        parser.add_argument('--cpu-procs', help='number of processes used for cpu')
        parser.add_argument('--io-procs', help='number of processes used for io')
        parser.add_argument('--phy-mem', help='physical memory size limit')
        parser.add_argument('--ignore-avail-mem', action='store_true', help='ignore available memory')
        parser.add_argument('--cache-mem', help='cache memory size limit')
        parser.add_argument('--disk', help='disk size limit')
        parser.add_argument('--spill-dir', help='spill directory')
        parser.add_argument('--plasma-socket', help='path of Plasma UNIX socket')
        parser.add_argument('--plasma-one-mapped-file', action='store_true',
                            help='path of Plasma UNIX socket')

    def validate_arguments(self):
        if not self.args.schedulers and not self.args.kv_store:
            raise StartArgumentError('either schedulers or url of kv store is required.')
        if not self.args.advertise:
            raise StartArgumentError('advertise address is required.')

    def start_plasma(self, mem_limit, **kwargs):
        from ..utils import PlasmaProcessHelper
        self._plasma_helper = PlasmaProcessHelper(size=int(mem_limit),
                                                  socket=options.worker.plasma_socket,
                                                  **kwargs)
        self._plasma_helper.run()

    def create_pool(self, *args, **kwargs):
        # here we create necessary actors on worker
        # and distribute them over processes
        mem_stats = resource.virtual_memory()

        options.worker.cpu_process_count = int(self.args.cpu_procs or resource.cpu_count())
        options.worker.io_process_count = int(self.args.io_procs or '1')
        options.worker.physical_memory_limit_hard = self._calc_size_limit(
            self.args.phy_mem or options.worker.physical_memory_limit_hard, mem_stats.total
        )
        options.worker.physical_memory_limit_soft = self._calc_size_limit(
            self.args.phy_mem or options.worker.physical_memory_limit_soft, mem_stats.total
        )
        options.worker.cache_memory_limit = self._calc_size_limit(
            self.args.cache_mem or options.worker.cache_memory_limit, mem_stats.total
        )
        options.worker.disk_limit = self.args.disk
        if self.args.spill_dir:
            from .spill import parse_spill_dirs
            options.worker.spill_directory = parse_spill_dirs(self.args.spill_dir)
            spill_dir_count = 1
        else:
            options.worker.spill_directory = None
            spill_dir_count = 0
        options.worker.plasma_socket = self.args.plasma_socket or '/tmp/plasma.sock'
        options.worker.advertise_addr = self.args.advertise

        self.n_process = 1 + options.worker.cpu_process_count + options.worker.io_process_count + spill_dir_count

        kwargs['distributor'] = WorkerDistributor(self.n_process)
        return super(WorkerApplication, self).create_pool(*args, **kwargs)

    @staticmethod
    def _calc_soft_memory_limit():
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

        logger.info('Setting soft limit to %s.', readable_size(quota_soft_limit))
        return quota_soft_limit

    def start_service(self):
        cache_mem_limit = options.worker.cache_memory_limit
        self.start_plasma(cache_mem_limit, one_mapped_file=self.args.plasma_one_mapped_file or False)
        time.sleep(1)

        from ..cluster_info import ClusterInfoActor

        if self.args.schedulers:
            if isinstance(self.args.schedulers, six.string_types):
                schedulers = [self.args.schedulers]
            else:
                schedulers = self.args.schedulers
            service_discover_addr = None
        else:
            schedulers = None
            service_discover_addr = self.args.kv_store
        self._cluster_info_ref = self.pool.create_actor(ClusterInfoActor, schedulers=schedulers,
                                                        service_discover_addr=service_discover_addr,
                                                        uid=ClusterInfoActor.default_name())

        from .status import StatusActor
        port_str = self.endpoint.rsplit(':', 1)[-1]
        self._status_ref = self.pool.create_actor(StatusActor, options.worker.advertise_addr + ':' + port_str,
                                                  uid='StatusActor')

        from .quota import QuotaActor, MemQuotaActor
        if self.args.ignore_avail_mem:
            # start a QuotaActor instead of MemQuotaActor to avoid memory size detection
            # for debug purpose only, DON'T USE IN PRODUCTION
            self._mem_quota_ref = self.pool.create_actor(
                QuotaActor, options.worker.physical_memory_limit_soft, uid='MemQuotaActor')
        else:
            self._mem_quota_ref = self.pool.create_actor(
                MemQuotaActor, self._calc_soft_memory_limit(),
                options.worker.physical_memory_limit_hard, uid='MemQuotaActor')

        from .chunkholder import ChunkHolderActor
        self._chunk_holder_ref = self.pool.create_actor(ChunkHolderActor, cache_mem_limit, uid='ChunkHolderActor')

        from .dispatcher import DispatchActor
        self._dispatch_ref = self.pool.create_actor(DispatchActor, uid='DispatchActor')

        from .execution import ExecutionActor
        self._execution_ref = self.pool.create_actor(ExecutionActor, uid='ExecutionActor')

        from .calc import CpuCalcActor
        for cpu_id in range(options.worker.cpu_process_count):
            uid = 'w:%d:mars-calc-%d-%d' % (cpu_id + 1, os.getpid(), cpu_id)
            actor = self.pool.create_actor(CpuCalcActor, uid=uid)
            self._cpu_calc_actors.append(actor)

        from .transfer import ReceiverActor, SenderActor
        start_pid = 1 + options.worker.cpu_process_count
        for sender_id in range(options.worker.io_process_count):
            uid = 'w:%d:mars-sender-%d-%d' % (start_pid + sender_id, os.getpid(), sender_id)
            actor = self.pool.create_actor(SenderActor, uid=uid)
            self._sender_actors.append(actor)
        for receiver_id in range(2 * options.worker.io_process_count):
            uid = 'w:%d:mars-receiver-%d-%d' % (start_pid + receiver_id // 2, os.getpid(), receiver_id)
            actor = self.pool.create_actor(ReceiverActor, uid=uid)
            self._receiver_actors.append(actor)

        from .prochelper import ProcessHelperActor
        for proc_id in range(self.n_process):
            uid = 'w:%d:mars-process-helper-%d-%d' % (proc_id, os.getpid(), proc_id)
            actor = self.pool.create_actor(ProcessHelperActor, uid=uid)
            self._process_helper_actors.append(actor)

        from .transfer import ResultSenderActor
        self._result_sender_ref = self.pool.create_actor(ResultSenderActor, uid='ResultSenderActor')

        start_pid = 1 + options.worker.cpu_process_count + options.worker.io_process_count
        if options.worker.spill_directory:
            from .spill import SpillActor
            for spill_id in range(len(options.worker.spill_directory) * 2):
                uid = 'w:%d:mars-spill-%d-%d' % (start_pid, os.getpid(), spill_id)
                actor = self.pool.create_actor(SpillActor, uid=uid)
                self._spill_actors.append(actor)

        kv_store = kvstore.get(options.kv_store)
        if isinstance(kv_store, kvstore.EtcdKVStore):
            kv_store.write('/workers/meta_timestamp', str(int(time.time())))

    def stop_service(self):
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

        if self._plasma_helper:
            self._plasma_helper.stop()


main = WorkerApplication()
if __name__ == '__main__':
    main()
