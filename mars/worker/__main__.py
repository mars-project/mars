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
from ..errors import StartArgumentError
from .distributor import WorkerDistributor
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
        self._service = WorkerService(
            advertise_addr=self.args.advertise,
            n_cpu_process=self.args.cpu_procs,
            n_io_process=self.args.io_procs,
            spill_dirs=self.args.spill_dir,
            total_mem=self.args.phy_mem,
            cache_mem_limit=self.args.cache_mem,
            ignore_avail_mem=self.args.ignore_avail_mem,
        )
        # start plasma
        self._service.start_plasma(one_mapped_file=self.args.plasma_one_mapped_file or False)

        self.n_process = self._service.n_process
        kwargs['distributor'] = WorkerDistributor(self.n_process)
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
