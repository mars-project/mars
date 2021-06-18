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

from typing import Union, Dict

from ... import oscar as mo
from .supervisor.fault_injection_manager import FaultInjectionManagerActor


class FaultInjectionAPI:
    def __init__(self, fault_injection_manager_ref: Union[FaultInjectionManagerActor, mo.ActorRef]):
        self._fault_injection_manager_ref = fault_injection_manager_ref

    @classmethod
    async def create(cls, address: str) -> "FaultInjectionAPI":
        """
        Create Fault Injection API.

        Parameters
        ----------
        address : str
            Supervisor address.

        Returns
        -------
        fault_injection_api
            Fault Injection API.
        """
        # Currently, the fault injection manager is global.
        fault_injection_manager_ref = await mo.actor_ref(
                FaultInjectionManagerActor.default_uid(),
                address=address)
        return FaultInjectionAPI(fault_injection_manager_ref)

    async def set_options(self, options: Dict):
        """
        Set the options for the fault injection manager.

        Parameters
        ----------
        options: Dict
        {
            "fault_count": 1
        }
        """
        return await self._fault_injection_manager_ref.set_options(options)

    async def on_execute_operand(self):
        """
        Be called on executing an operand.

        Returns
        -------
        fault : bool
            True if fault else False
        """
        return await self._fault_injection_manager_ref.on_execute_operand()
