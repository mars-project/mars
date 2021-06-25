# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from ..oscar.worker import WorkerCommandRunner
from .config import MarsWorkerConfig
from .core import YarnServiceMixin


class YarnWorkerCommandRunner(YarnServiceMixin, WorkerCommandRunner):
    service_name = MarsWorkerConfig.service_name

    def __call__(self, *args, **kwargs):
        os.environ['MARS_CONTAINER_IP'] = self.get_container_ip()
        return super().__call__(*args, **kwargs)

    async def start_services(self):
        from ..oscar.worker import start_worker
        from ...services.cluster import ClusterAPI

        self.register_endpoint()

        await start_worker(
            self.pool.external_address, self.args.supervisors,
            self.band_to_slot, list(self.args.load_modules), self.config,
            mark_ready=False)
        await self.wait_all_supervisors_ready()

        cluster_api = await ClusterAPI.create(self.args.endpoint)
        await cluster_api.mark_node_ready()


main = YarnWorkerCommandRunner()

if __name__ == '__main__':   # pragma: no branch
    main()
