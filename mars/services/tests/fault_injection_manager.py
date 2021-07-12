import enum
import uuid
from abc import ABC, abstractmethod

from ...services.session import SessionAPI


class ExtraConfigKey:
    FAULT_INJECTION_MANAGER_NAME = 'fault_injection_manager_name'


class FaultType(enum.Enum):
    NoFault = 0
    Exception = 1
    ProcessExit = 2


class AbstractFaultInjectionManager(ABC):
    """
    The abstract base of fault injection manager for test.
    """
    name = str(uuid.uuid4())

    @abstractmethod
    def on_execute_operand(self) -> FaultType:
        """
        Be called when executing operand on worker.

        Returns
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
        await session_api.create_remote_object(
                session_id, cls.name, cls)
