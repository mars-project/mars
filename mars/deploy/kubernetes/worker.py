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

import logging

from ..oscar.worker import WorkerCommandRunner
from .core import K8SServiceMixin

logger = logging.getLogger(__name__)


class K8SWorkerCommandRunner(K8SServiceMixin, WorkerCommandRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def start_services(self):
        from ..oscar.worker import start_worker
        from ...services.cluster import ClusterAPI

        self.write_pid_file()
        await start_worker(
            self.pool.external_address, self.args.supervisors,
            self.band_to_slot, list(self.args.load_modules), self.config,
            mark_ready=False)
        await self.wait_all_supervisors_ready()

        cluster_api = await ClusterAPI.create(self.args.endpoint)
        await cluster_api.mark_node_ready()

        await self.start_readiness_server()

    async def stop_services(self):
        await self.stop_readiness_server()
        await super().stop_services()


main = K8SWorkerCommandRunner()

if __name__ == '__main__':   # pragma: no branch
    main()
