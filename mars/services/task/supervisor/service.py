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

from typing import Dict

from .... import oscar as mo
from .task_manager import TaskConfigurationActor


async def start(config: Dict, address: str):
    """
    Start task service on supervisor.

    Parameters
    ----------
    config
        service config.
        {
            "task": {
                "default_config": {
                    "optimize_tileable_graph": True,
                    "optimize_chunk_graph": True,
                    "fuse_enabled": True
                }
            }
        }
    address : str
        Actor pool address.
    """
    task_config = config.get('task', dict())
    options = task_config.get('default_config', dict())
    task_processor_cls = task_config.get('task_processor_cls')
    await mo.create_actor(TaskConfigurationActor, options,
                          task_processor_cls=task_processor_cls,
                          address=address,
                          uid=TaskConfigurationActor.default_uid())
