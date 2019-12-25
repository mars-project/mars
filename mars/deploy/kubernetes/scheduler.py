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

from ...scheduler.__main__ import SchedulerApplication
from .core import K8SServiceMixin, ReadinessActor


class K8SSchedulerApplication(K8SServiceMixin, SchedulerApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._readiness_ref = None

    def start(self):
        self.write_pid_file()
        super().start()
        self._readiness_ref = self.pool.create_actor(ReadinessActor, uid=ReadinessActor.default_uid())

    def stop(self):
        self._readiness_ref.destroy()
        super().stop()


main = K8SSchedulerApplication()

if __name__ == '__main__':   # pragma: no branch
    main()
