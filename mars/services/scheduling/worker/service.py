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
from ....utils import calc_size_by_str
from .workerslot import WorkerSlotManagerActor
from .quota import WorkerQuotaManagerActor
from .execution import SubtaskExecutionActor


async def start(config: Dict, address: None):
    """
    Start scheduling service on worker.

    Parameters
    ----------
    config : dict
        service config.
    address : str
        Actor pool address.
    """
    from .... import resource as mars_resource
    scheduling_config = config.get('scheduling', {})

    total_mem = mars_resource.virtual_memory().total
    mem_quota_size = calc_size_by_str(
        scheduling_config.get('mem_quota_size', '80%'), total_mem)

    await mo.create_actor(WorkerSlotManagerActor,
                          uid=WorkerSlotManagerActor.default_uid(),
                          address=address)
    await mo.create_actor(WorkerQuotaManagerActor,
                          default_config=dict(soft_limit=mem_quota_size),
                          uid=WorkerQuotaManagerActor.default_uid(),
                          address=address)
    await mo.create_actor(SubtaskExecutionActor,
                          uid=SubtaskExecutionActor.default_uid(),
                          address=address)
