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

from typing import Any, Dict

from mars.core import OperandType
from mars.services.subtask.worker.subtask import SubtaskProcessor

from .api import FaultInjectionAPI
from ...lib.aio import alru_cache


class FaultInjectionSubtaskProcessor(SubtaskProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._fault_injection_api = None

    @staticmethod
    @alru_cache(cache_exceptions=False)
    async def _get_fault_injection_api(supervisor_address: str) -> FaultInjectionAPI:
        return await FaultInjectionAPI.create(supervisor_address)

    async def run(self):
        self._fault_injection_api = await self._get_fault_injection_api(self._supervisor_address)
        return await super().run()

    async def _async_execute_operand(self,
                                     loop,
                                     executor,
                                     ctx: Dict[str, Any],
                                     op: OperandType):
        fault = await self._fault_injection_api.on_execute_operand()
        if fault:
            raise RuntimeError("Fault Injection")

        return await super()._async_execute_operand(loop, executor, ctx, op)
