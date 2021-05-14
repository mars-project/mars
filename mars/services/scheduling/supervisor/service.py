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


async def start(config: Dict, address: None):
    """
    Start meta service on supervisor.

    Parameters
    ----------
    config : dict
        service config.
        {
            "scheduling" : {
                "submit_period": 1
            }
        }
    address : str
        Actor pool address.
    """
    from .globalslot import GlobalSlotManagerActor
    await mo.create_actor(
        GlobalSlotManagerActor, uid=GlobalSlotManagerActor.default_uid(),
        address=address)
