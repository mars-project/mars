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

import sys
from typing import Any, Callable, Dict, Tuple, Union

from ... import oscar as mo
from ...core import OperandType
from ...lib.aio import alru_cache
from ...tests.core import patch_cls, patch_super as super
from ..session import SessionAPI
from ..scheduling.worker.execution import SubtaskExecutionActor, SubtaskExecutionInfo
from ..subtask.worker.processor import SubtaskProcessor
from ..tests.fault_injection_manager import (
    AbstractFaultInjectionManager,
    ExtraConfigKey,
    FaultPosition,
    handle_fault,
)


@patch_cls(SubtaskExecutionActor)
class FaultInjectedSubtaskExecutionActor(SubtaskExecutionActor):
    @alru_cache(cache_exceptions=False)
    async def _get_fault_injection_manager_ref(
        self, supervisor_address: str, session_id: str, name: str
    ) -> Union[mo.ActorRef, AbstractFaultInjectionManager]:
        session_api = await self._get_session_api(supervisor_address)
        return await session_api.get_remote_object(session_id, name)

    @staticmethod
    @alru_cache(cache_exceptions=False)
    async def _get_session_api(supervisor_address: str):
        return await SessionAPI.create(supervisor_address)

    async def _execute_subtask_once(self, subtask_info: SubtaskExecutionInfo):
        try:
            return await super()._execute_subtask_once(subtask_info)
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            exc_info = sys.exc_info()
            subtask = subtask_info.subtask
            if not subtask.retryable:
                unretryable_op = [
                    chunk.op
                    for chunk in subtask.chunk_graph
                    if not getattr(chunk.op, "retryable", True)
                ]
                message = (
                    f"Run subtask failed due to {exc_info[1]}, "
                    f"the subtask {subtask.subtask_id} is not retryable, "
                    f"it contains unretryable op: {unretryable_op!r}"
                )
                _UnretryableException = type(
                    "_UnretryableException", (exc_info[0],), {}
                )
                raise _UnretryableException(message).with_traceback(exc_info[-1])
            else:
                raise

    async def _execute_subtask_with_retry(self, subtask_info: SubtaskExecutionInfo):
        subtask = subtask_info.subtask
        # fault injection
        if subtask.extra_config:
            fault_injection_manager_name = subtask.extra_config.get(
                ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME
            )
            if fault_injection_manager_name is not None:
                fault_injection_manager = await self._get_fault_injection_manager_ref(
                    subtask_info.supervisor_address,
                    subtask.session_id,
                    fault_injection_manager_name,
                )
                fault = await fault_injection_manager.get_fault(
                    FaultPosition.ON_RUN_SUBTASK, {"subtask": subtask}
                )
                handle_fault(fault)
        return await super()._execute_subtask_with_retry(subtask_info)

    @classmethod
    def _log_subtask_retry(
        cls,
        subtask_info: SubtaskExecutionInfo,
        target_func: Callable,
        trial: int,
        exc_info: Tuple,
        retry: bool = True,
    ):
        exc_info = (
            super()._log_subtask_retry(
                subtask_info, target_func, trial, exc_info, retry=retry
            )
            or exc_info
        )

        if retry:
            if trial < subtask_info.max_retries - 1:
                return exc_info
            else:
                _ExceedMaxRerun = type("_ExceedMaxRerun", (exc_info[0],), {})
                return (
                    _ExceedMaxRerun,
                    _ExceedMaxRerun(str(exc_info[1])).with_traceback(exc_info[-1]),
                    exc_info[-1],
                )
        else:
            _UnhandledException = type("_UnhandledException", (exc_info[0],), {})
            return (
                _UnhandledException,
                _UnhandledException(str(exc_info[1])).with_traceback(exc_info[-1]),
                exc_info[-1],
            )


@patch_cls(SubtaskProcessor)
class FaultInjectionSubtaskProcessor(SubtaskProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fault_injection_manager_ref: Union[
            mo.ActorRef, AbstractFaultInjectionManager
        ] = None

    async def run(self):
        if self.subtask.extra_config:
            fault_injection_manager_name = self.subtask.extra_config.get(
                ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME
            )
            if fault_injection_manager_name is not None:
                self._fault_injection_manager_ref = (
                    await self._session_api.get_remote_object(
                        self._session_id, fault_injection_manager_name
                    )
                )
        return await super().run()

    async def _async_execute_operand(self, ctx: Dict[str, Any], op: OperandType):
        if self._fault_injection_manager_ref is not None:
            fault = await self._fault_injection_manager_ref.get_fault(
                FaultPosition.ON_EXECUTE_OPERAND,
                {"subtask": self.subtask, "operand": op},
            )
            handle_fault(fault)
        return await super()._async_execute_operand(ctx, op)
