# -*- coding: utf-8 -*-
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

from ...scheduler.__main__ import SchedulerApplication
from .config import MarsSchedulerConfig
from .core import YarnServiceMixin


class YarnSchedulerApplication(YarnServiceMixin, SchedulerApplication):
    service_name = MarsSchedulerConfig.service_name

    def __call__(self, *args, **kwargs):
        os.environ['MARS_CONTAINER_IP'] = self.get_container_ip()
        return super().__call__(*args, **kwargs)

    def start(self):
        self.start_daemon()

        super().start()
        self.register_node()


main = YarnSchedulerApplication()

if __name__ == '__main__':   # pragma: no branch
    main()
