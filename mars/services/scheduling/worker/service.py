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

from typing import Dict

from .... import oscar as mo
from ....utils import calc_size_by_str
from .workerslot import WorkerSlotManagerActor
from .quota import WorkerQuotaManagerActor
from .execution import SubtaskExecutionActor


async def start(config: Dict, address: str):
    """
    Start scheduling service on worker.

    Parameters
    ----------
    config : dict
        service config.
        {
            "scheduling": {
                "mem_quota_size": "80%",
                "mem_hard_limit": "95%",
                "enable_kill_slot": true,
            }
        }
    address : str
        Actor pool address.
    """
    from .... import resource as mars_resource
    scheduling_config = config.get('scheduling', {})

    total_mem = mars_resource.virtual_memory().total
    mem_quota_size = calc_size_by_str(
        scheduling_config.get('mem_quota_size', '80%'), total_mem)
    mem_hard_limit = calc_size_by_str(
        scheduling_config.get('mem_hard_limit', '95%'), total_mem)
    enable_kill_slot = scheduling_config.get('enable_kill_slot', True)

    await mo.create_actor(WorkerSlotManagerActor,
                          uid=WorkerSlotManagerActor.default_uid(),
                          address=address)
    await mo.create_actor(WorkerQuotaManagerActor,
                          default_config=dict(quota_size=mem_quota_size,
                                              hard_limit=mem_hard_limit,
                                              enable_kill_slot=enable_kill_slot),
                          uid=WorkerQuotaManagerActor.default_uid(),
                          address=address)
    await mo.create_actor(SubtaskExecutionActor,
                          enable_kill_slot=enable_kill_slot,
                          uid=SubtaskExecutionActor.default_uid(),
                          address=address)


async def stop(config: dict, address: str):
    await mo.destroy_actor(mo.create_actor_ref(
        uid=SubtaskExecutionActor.default_uid(), address=address))
    await mo.destroy_actor(mo.create_actor_ref(
        uid=WorkerQuotaManagerActor.default_uid(), address=address))
    await mo.destroy_actor(mo.create_actor_ref(
        uid=WorkerSlotManagerActor.default_uid(), address=address))
