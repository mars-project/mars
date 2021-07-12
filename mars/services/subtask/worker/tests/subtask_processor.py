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

import os
from typing import Any, Dict

from mars.core import OperandType
from mars.services.subtask.worker.processor import SubtaskProcessor
from mars.tests.core import _check_args, ObjectCheckMixin

from ....tests.fault_injection_manager import FaultType, ExtraConfigKey


class CheckedSubtaskProcessor(ObjectCheckMixin, SubtaskProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        check_options = dict()
        kwargs = self.subtask.extra_config or dict()
        self._operand_executors = operand_executors = \
            kwargs.pop('operand_executors', dict())
        for op, executor in operand_executors.items():
            op.register_executor(executor)
        for key in _check_args:
            check_options[key] = kwargs.get(key, True)
        self._check_options = check_options

    def _execute_operand(self,
                         ctx: Dict[str, Any],
                         op: OperandType):
        super()._execute_operand(ctx, op)
        if self._check_options.get('check_all', True):
            for out in op.outputs:
                if out not in self._chunk_graph.result_chunks:
                    continue
                if out.key not in ctx and \
                        any(k[0] == out.key for k in ctx if isinstance(k, tuple)):
                    # both shuffle mapper and reducer
                    continue
                self.assert_object_consistent(out, ctx[out.key])

    async def done(self):
        await super().done()
        for op in self._operand_executors:
            op.unregister_executor()


_fault_args = [ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME]


class FaultInjectionSubtaskProcessor(SubtaskProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fault_options = dict()
        kwargs = self.subtask.extra_config or dict()
        self._operand_executors = operand_executors = \
            kwargs.pop('operand_executors', dict())
        for op, executor in operand_executors.items():
            op.register_executor(executor)
        for key in _fault_args:
            if key in kwargs:
                fault_options[key] = kwargs[key]
        self._fault_options = fault_options
        self._fault_injection_manager = None

    async def done(self):
        await super().done()
        for op in self._operand_executors:
            op.unregister_executor()

    async def run(self):
        self._fault_injection_manager = await self._session_api.get_remote_object(
                self._session_id,
                self._fault_options[ExtraConfigKey.FAULT_INJECTION_MANAGER_NAME])
        return await super().run()

    async def _async_execute_operand(self,
                                     loop,
                                     executor,
                                     ctx: Dict[str, Any],
                                     op: OperandType):
        fault = await self._fault_injection_manager.on_execute_operand()
        if fault == FaultType.Exception:
            raise RuntimeError("Fault Injection")
        elif fault == FaultType.ProcessExit:
            # used to simulate process crash, no cleanup.
            os._exit(-1)
        assert fault == FaultType.NoFault, \
            f"Got unexpected fault from on_execute_operand: {fault}"

        return await super()._async_execute_operand(loop, executor, ctx, op)
