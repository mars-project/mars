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

from ..oscar.supervisor import SupervisorCommandRunner
from .core import K8SServiceMixin


class K8SSupervisorCommandRunner(K8SServiceMixin, SupervisorCommandRunner):
    async def start_services(self):
        await super().start_services()
        await self.start_readiness_server()

    async def stop_services(self):
        await self.stop_readiness_server()
        await super().stop_services()


main = K8SSupervisorCommandRunner()

if __name__ == '__main__':   # pragma: no branch
    main()
