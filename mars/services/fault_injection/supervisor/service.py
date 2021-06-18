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
from .fault_injection_manager import FaultInjectionManagerActor


async def start(config: Dict, address: str):
    """
    Start fault injection service on supervisor.

    Parameters
    ----------
    config
        service config.
        {
            "fault_injection" : {
            }
        }
    address : str
        Actor pool address.
    """
    await mo.create_actor(FaultInjectionManagerActor, address=address,
                          uid=FaultInjectionManagerActor.default_uid())
