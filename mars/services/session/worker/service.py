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
from .custom_log import CustomLogActor


async def start(config: Dict, address: str):
    """
    Start meta service on worker.

    Parameters
    ----------
    config : dict
        service config.
        {
            "session" : {
            }
        }
    address : str
        Actor pool address.
    """
    session_config = config.get('session', dict())
    custom_log_dir = session_config.get('custom_log_dir')
    await mo.create_actor(CustomLogActor, custom_log_dir,
                          address=address, uid=CustomLogActor.default_uid())
