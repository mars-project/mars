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
from ..base_app import BaseApplication
from ..distributor import MarsDistributor
from .service import SchedulerService

logger = logging.getLogger(__name__)


class SchedulerApplication(BaseApplication):
    service_description = 'Mars Scheduler'
    service_logger = logger

    def __init__(self):
        super(SchedulerApplication, self).__init__()
        self._service = None

    def config_args(self, parser):
        parser.add_argument('--nproc', help='number of processes')

    def create_pool(self, *args, **kwargs):
        self._service = SchedulerService()
        self.n_process = int(self.args.nproc or resource.cpu_count())
        kwargs['distributor'] = MarsDistributor(self.n_process, 's:h1:')
        return super(SchedulerApplication, self).create_pool(*args, **kwargs)

    def start_service(self):
        self._service.start(self.endpoint, self.args.schedulers, self.pool)

    def stop_service(self):
        self._service.stop(self.pool)


main = SchedulerApplication()

if __name__ == '__main__':
    main()
