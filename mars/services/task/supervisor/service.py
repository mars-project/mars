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
from .manager import TaskConfigurationActor, TaskManagerActor


class TaskSupervisorService(AbstractService):
    """
    Task service on supervisor.

    Service Configuration
    ---------------------
    {
        "task": {
            "default_config": {
                "optimize_tileable_graph": True,
                "optimize_chunk_graph": True,
                "fuse_enabled": True
            }
        }
    }
    """
    async def start(self):
        task_config = self._config.get('task', dict())
        options = task_config.get('default_config', dict())
        task_preprocessor_cls = task_config.get('task_preprocessor_cls')
        await mo.create_actor(TaskConfigurationActor, options,
                              task_preprocessor_cls=task_preprocessor_cls,
                              address=self._address,
                              uid=TaskConfigurationActor.default_uid())

    async def stop(self):
        await mo.destroy_actor(mo.create_actor_ref(
            uid=TaskConfigurationActor.default_uid(), address=self._address))

    async def create_session(self, session_id: str):
        await mo.create_actor(
            TaskManagerActor, session_id, address=self._address,
            uid=TaskManagerActor.gen_uid(session_id))

    async def destroy_session(self, session_id: str):
        task_manager_ref = await mo.actor_ref(
            self._address, TaskManagerActor.gen_uid(session_id))
        return await mo.destroy_actor(task_manager_ref)
