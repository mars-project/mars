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

from ..oscar.supervisor import SupervisorCommandRunner
from .config import MarsSupervisorConfig
from .core import YarnServiceMixin


class YarnSupervisorCommandRunner(YarnServiceMixin, SupervisorCommandRunner):
    service_name = MarsSupervisorConfig.service_name
    web_service_name = MarsSupervisorConfig.web_service_name

    def __call__(self, *args, **kwargs):
        os.environ['MARS_CONTAINER_IP'] = self.get_container_ip()
        return super().__call__(*args, **kwargs)

    async def start_services(self):
        self.register_endpoint()

        await super().start_services()

        from ...services.web import OscarWebAPI
        web_api = await OscarWebAPI.create(self.args.endpoint)
        web_endpoint = await web_api.get_web_address()
        self.register_endpoint(self.web_service_name, web_endpoint)


main = YarnSupervisorCommandRunner()

if __name__ == '__main__':   # pragma: no branch
    main()
