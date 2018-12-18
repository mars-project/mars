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

from .. import resource
from ..config import options
from ..base_app import BaseApplication
from ..errors import StartArgumentError
from .distributor import WorkerDistributor
from .service import WorkerService

logger = logging.getLogger(__name__)


class WorkerApplication(BaseApplication, WorkerService):
    """
    Main function class of Mars Worker
    """
    service_description = 'Mars Worker'
    service_logger = logger

    def __init__(self):
        super(BaseApplication, self).__init__()
        super(WorkerService, self).__init__()

    def config_args(self, parser):
        parser.add_argument('-s', '--schedulers', help='scheduler addresses')
        parser.add_argument('--cpu-procs', help='number of processes used for cpu')
        parser.add_argument('--io-procs', help='number of processes used for io')
        parser.add_argument('--phy-mem', help='physical memory size limit')
        parser.add_argument('--ignore-avail-mem', action='store_true', help='ignore available memory')
        parser.add_argument('--cache-mem', help='cache memory size limit')
        parser.add_argument('--disk', help='disk size limit')
        parser.add_argument('--spill-dir', help='spill directory')
        parser.add_argument('--plasma-one-mapped-file', action='store_true',
                            help='path of Plasma UNIX socket')

    def validate_arguments(self):
        if not self.args.schedulers and not self.args.kv_store:
            raise StartArgumentError('either schedulers or url of kv store is required.')
        if not self.args.advertise:
            raise StartArgumentError('advertise address is required.')

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
            self.args.phy_mem or options.worker.physical_memory_limit_soft or '48%', mem_stats.total
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
        options.worker.advertise_addr = self.args.advertise

        self.n_process = 1 + options.worker.cpu_process_count + options.worker.io_process_count + spill_dir_count

        # start plasma
        self.start_plasma(options.worker.cache_memory_limit,
                          one_mapped_file=options.worker.plasma_one_mapped_file or False)

        kwargs['distributor'] = WorkerDistributor(self.n_process)
        return super(WorkerApplication, self).create_pool(*args, **kwargs)

    def start_service(self):
        super(WorkerApplication, self).start(self.endpoint, self.args.schedulers,
                                             self.pool, ignore_avail_mem=self.args.ignore_avail_mem)

    def stop_service(self):
        super(WorkerApplication, self).stop()


main = WorkerApplication()
if __name__ == '__main__':
    main()
