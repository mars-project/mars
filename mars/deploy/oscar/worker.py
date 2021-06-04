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

import os

from ...resource import cpu_count, cuda_count
from ...utils import get_next_port
from .cmdline import OscarCommandRunner
from .local import start_worker, stop_worker
from .pool import create_worker_actor_pool


class WorkerCommandRunner(OscarCommandRunner):
    command_description = 'Mars Worker'

    def __init__(self):
        super().__init__()
        self.band_to_slot = dict()
        self.n_io_process = 1

    def config_args(self, parser):
        super().config_args(parser)
        parser.add_argument('--n-cpu', help='num of CPU to use', default='auto')
        parser.add_argument('--n-gpu', help='num of GPU to use', default='auto')
        parser.add_argument('--n-io-process', help='num of IO processes', default='1')

    def parse_args(self, parser, argv, environ=None):
        environ = environ or os.environ
        args = super().parse_args(parser, argv, environ=environ)

        if self.config.get('cluster', {}).get('backend', 'fixed') == 'fixed' \
                and not args.supervisors:  # pragma: no cover
            raise ValueError('--supervisors is needed to start Mars Worker')

        if args.endpoint is None:
            args.endpoint = f'0.0.0.0:{get_next_port()}'
        self.n_io_process = int(args.n_io_process)

        n_cpu = cpu_count() if args.n_cpu == 'auto' else args.n_cpu
        n_gpu = cuda_count() if args.n_gpu == 'auto' else args.n_gpu

        self.band_to_slot = band_to_slot = dict()
        band_to_slot['numa-0'] = n_cpu
        for i in range(n_gpu):  # pragma: no cover
            band_to_slot[f'gpu-{i}'] = 1

        storage_config = self.config['storage'] = self.config.get('storage', {})
        plasma_config = storage_config['plasma'] = storage_config.get('plasma', {})
        if 'MARS_CACHE_MEM_SIZE' in environ:
            plasma_config['store_memory'] = environ['MARS_CACHE_MEM_SIZE']
        if 'MARS_PLASMA_DIRS' in environ:
            plasma_config['plasma_directory'] = environ['MARS_PLASMA_DIRS']

        return args

    async def create_actor_pool(self):
        return await create_worker_actor_pool(
            self.args.endpoint, self.band_to_slot, ports=self.ports,
            n_io_process=self.n_io_process, modules=list(self.args.load_modules),
            subprocess_start_method='forkserver' if os.name == 'nt' else 'spawn'
        )

    async def start_services(self):
        return await start_worker(
            self.pool.external_address, self.args.supervisors,
            self.band_to_slot, list(self.args.load_modules), self.config)

    async def stop_services(self):
        return await stop_worker(self.pool.external_address, self.config)


main = WorkerCommandRunner()

if __name__ == '__main__':  # pragma: no branch
    main()
