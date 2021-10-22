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

import enum
import os
import uuid
from abc import ABC, abstractmethod

from ...core.base import MarsError
from ..session import SessionAPI


class ExtraConfigKey:
    FAULT_INJECTION_MANAGER_NAME = "fault_injection_manager_name"


class FaultPosition(enum.Enum):
    ON_EXECUTE_OPERAND = 0
    ON_RUN_SUBTASK = 1


class FaultType(enum.Enum):
    NoFault = 0
    Exception = 1
    UnhandledException = 2
    ProcessExit = 3


class FaultInjectionError(MarsError):
    pass


class FaultInjectionUnhandledError(Exception):
    pass


def handle_fault(fault):
    if fault == FaultType.Exception:
        raise FaultInjectionError("Fault Injection")
    elif fault == FaultType.UnhandledException:
        raise FaultInjectionUnhandledError("Fault Injection Unhandled")
    elif fault == FaultType.ProcessExit:
        # used to simulate process crash, no cleanup.
        os._exit(-1)
    assert fault == FaultType.NoFault, f"Got unexpected fault: {fault}"


class AbstractFaultInjectionManager(ABC):
    """
    The abstract base of fault injection manager for test.
    """

    name = str(uuid.uuid4())

    @abstractmethod
    def get_fault(self, pos: FaultPosition, ctx=None) -> FaultType:
        """
        Get fault at position.

        Parameters
        ----------
        pos
            The fault position.
        ctx
            The fault context.

        Returns
        -------
            The fault type.
        """
        pass

    @classmethod
    async def create(cls, session_id, supervisor_address):
        """
        Create the fault injection manager on supervisor.

        Parameters
        ----------
        session_id
            The session id.
        supervisor_address
            The supervisor address.
        -------
        """
        session_api = await SessionAPI.create(supervisor_address)
        await session_api.create_remote_object(session_id, cls.name, cls)
