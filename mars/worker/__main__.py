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

from ..base_app import BaseApplication
from ..distributor import MarsDistributor
from ..errors import StartArgumentError
from ..config import options
from ..serialize.dataserializer import CompressType
from .service import WorkerService

logger = logging.getLogger(__name__)


class WorkerApplication(BaseApplication):
    """
    Main function class of Mars Worker
    """
    service_description = 'Mars Worker'
    service_logger = logger

    def __init__(self):
        super(WorkerApplication, self).__init__()
        self._service = None

    def config_args(self, parser):
        parser.add_argument('--cpu-procs', help='number of processes used for cpu')
        parser.add_argument('--io-procs', help='number of processes used for io')
        parser.add_argument('--phy-mem', help='physical memory size limit')
        parser.add_argument('--ignore-avail-mem', action='store_true', help='ignore available memory')
        parser.add_argument('--cache-mem', help='cache memory size limit')
        parser.add_argument('--min-mem', help='minimal free memory required to start worker')
        parser.add_argument('--spill-dir', help='spill directory')

        compress_types = ', '.join(v.value for v in CompressType.__members__.values())
        parser.add_argument('--disk-compression',
                            default=options.worker.disk_compression,
                            help='compression type used for disks, '
                                 'can be selected from %s. %s by default'
                                 % (compress_types, options.worker.disk_compression))
        parser.add_argument('--transfer-compression',
                            default=options.worker.transfer_compression,
                            help='compression type used for network transfer, '
                                 'can be selected from %s. %s by default'
                                 % (compress_types, options.worker.transfer_compression))

    def validate_arguments(self):
        if not self.args.schedulers and not self.args.kv_store:
            raise StartArgumentError('either schedulers or url of kv store is required.')
        if not self.args.advertise:
            raise StartArgumentError('advertise address is required.')

        compress_types = set(v.value for v in CompressType.__members__.values())
        if self.args.disk_compression.lower() not in compress_types:
            raise StartArgumentError('illegal disk compression config %s.' % self.args.disk_compression)
        if self.args.transfer_compression.lower() not in compress_types:
            raise StartArgumentError('illegal transfer compression config %s.' % self.args.transfer_compression)

    def create_pool(self, *args, **kwargs):
        # here we create necessary actors on worker
        # and distribute them over processes
        self._service = WorkerService(
            advertise_addr=self.args.advertise,
            n_cpu_process=self.args.cpu_procs,
            n_io_process=self.args.io_procs,
            spill_dirs=self.args.spill_dir,
            total_mem=self.args.phy_mem,
            cache_mem_limit=self.args.cache_mem,
            ignore_avail_mem=self.args.ignore_avail_mem,
            min_mem_size=self.args.min_mem,
            disk_compression=self.args.disk_compression.lower(),
            transfer_compression=self.args.transfer_compression.lower(),
        )
        # start plasma
        self._service.start_plasma()

        self.n_process = self._service.n_process
        kwargs['distributor'] = MarsDistributor(self.n_process, 'w:0:')
        return super(WorkerApplication, self).create_pool(*args, **kwargs)

    def start(self):
        self._service.start(self.endpoint, self.pool, schedulers=self.args.schedulers)

    def handle_process_down(self, proc_indices):
        self._service.handle_process_down(self.pool, proc_indices)

    def stop(self):
        self._service.stop()


main = WorkerApplication()
if __name__ == '__main__':
    main()
