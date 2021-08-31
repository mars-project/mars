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

from typing import Union

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...tests.core import patch_cls, patch_super as super
from ..session import SessionAPI
from ..scheduling.worker.execution import SubtaskExecutionActor
from ..subtask import Subtask
from ..tests.fault_injection_manager import (
    AbstractFaultInjectionManager,
    ExtraConfigKey,
    FaultPosition,
    handle_fault,
)


@patch_cls(SubtaskExecutionActor)
class FaultInjectedSubtaskExecutionActor(SubtaskExecutionActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_api = None
        self._fault_injection_manager_ref = None

    @alru_cache(cache_exceptions=False)
    async def _get_fault_injection_manager_ref(self, supervisor_address: str, session_id: str, name: str) \
            -> Union[mo.ActorRef, AbstractFaultInjectionManager]:
        session_api = await self._get_session_api(supervisor_address)
        return await session_api.get_remote_object(session_id, name)

    @staticmethod
    @alru_cache(cache_exceptions=False)
    async def _get_session_api(supervisor_address: str):
        return await SessionAPI.create(supervisor_address)

    async def internal_run_subtask(self, subtask: Subtask, band_name: str):
        # fault injection
        if subtask.extra_config:
            fault_injection_manager_name = subtask.extra_config.get(
                    ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME)
            if fault_injection_manager_name is not None:
                subtask_info = self._subtask_info[subtask.subtask_id]
                fault_injection_manager = await self._get_fault_injection_manager_ref(
                        subtask_info.supervisor_address, subtask.session_id, fault_injection_manager_name)
                fault = await fault_injection_manager.get_fault(
                        FaultPosition.ON_RUN_SUBTASK, {'subtask': subtask})
                handle_fault(fault)
        return super().internal_run_subtask(subtask, band_name)
