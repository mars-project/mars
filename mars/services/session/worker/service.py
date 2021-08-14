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

from .... import oscar as mo
from ...core import AbstractService
from .custom_log import CustomLogActor


class SessionWorkerService(AbstractService):
    """
    Session service on worker.

    Service Configuration
    ---------------------
    {
        "session" : {
        }
    }
    """
    async def start(self):
        session_config = self._config.get('session', dict())
        custom_log_dir = session_config.get('custom_log_dir')
        await mo.create_actor(CustomLogActor, custom_log_dir,
                              address=self._address, uid=CustomLogActor.default_uid())

    async def stop(self):
        await mo.destroy_actor(mo.create_actor_ref(
            uid=CustomLogActor.default_uid(), address=self._address))
