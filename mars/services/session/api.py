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

from abc import ABC, abstractmethod
from typing import Union, TypeVar
from urllib.parse import urlparse

from ... import oscar as mo
from ...lib.aio import alru_cache
from .supervisor import SessionManagerActor
from mars.services.web.core import ServiceWebAPIBase, get_supervisor_address


APIType = TypeVar('APIType', bound='SessionAPI')


class SessionAPI(ABC):
    @classmethod
    @alru_cache
    async def create(cls, address: str, **kwargs) -> "APIType":
        if urlparse(address).scheme == 'http':
            return await SessionWebAPI.create(address, *kwargs)
        else:
            return await OscarSessionAPI.create(address, **kwargs)

    @abstractmethod
    async def create_session(self, session_id: str) -> str:
        """
        Create session and return address.

        Parameters
        ----------
        session_id : str
            Session ID

        Returns
        -------
        address : str
            Session address.
        """

    @abstractmethod
    async def delete_session(self, session_id: str):
        """
        Delete session.

        Parameters
        ----------
        session_id : str
            Session ID.
        """


class OscarSessionAPI(SessionAPI):
    def __init__(self,
                 address: str,
                 session_manager: Union[SessionManagerActor, mo.ActorRef]):
        self._address = address
        self._session_manager_ref = session_manager

    @classmethod
    @alru_cache
    async def create(cls, address: str, **kwargs) -> "OscarSessionAPI":
        if kwargs:  # pragma: no cover
            raise TypeError(f'OscarSessionAPI.create '
                            f'got unknown arguments: {list(kwargs)}')
        session_manager = await mo.actor_ref(
            address, SessionManagerActor.default_uid())
        return OscarSessionAPI(address, session_manager)

    async def create_session(self, session_id: str) -> str:
        session_actor_ref = \
            await self._session_manager_ref.create_session(session_id)
        return session_actor_ref.address

    async def has_session(self, session_id: str) -> bool:
        """
        Check if session created.

        Parameters
        ----------
        session_id : str
            Session ID.

        Returns
        -------
        if_exists : bool
        """
        return await self._session_manager_ref.has_session(session_id)

    async def delete_session(self, session_id: str):
        await self._session_manager_ref.delete_session(session_id)

    async def get_session_address(self, session_id: str) -> str:
        """
        Get session address.

        Parameters
        ----------
        session_id : str
            Session ID.

        Returns
        -------
        address : str
            Session address.
        """
        return (await self._session_manager_ref.get_session_ref(session_id)).address

    async def last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        """
        Get session last idle time.

        Parameters
        ----------
        session_id : str, None
            Session ID. None for all sessions.

        Returns
        -------
        last_idle_time: str
            The last idle time if the session(s) is idle else None.
        """
        return await self._session_manager_ref.last_idle_time(session_id)


class MockSessionAPI(OscarSessionAPI):
    @classmethod
    async def create(cls,
                     address: str, **kwargs) -> "SessionAPI":
        session_id = kwargs.pop('session_id')
        if kwargs:  # pragma: no cover
            raise TypeError(f'SessionAPI.create '
                            f'got unknown arguments: {list(kwargs)}')

        session_manager = await mo.create_actor(
            SessionManagerActor, address=address,
            uid=SessionManagerActor.default_uid())
        if session_id:
            await session_manager.create_session(
                session_id, create_services=False)
        return MockSessionAPI(address, session_manager)


class SessionWebAPI(ServiceWebAPIBase, SessionAPI):
    _service_name = 'session'

    @classmethod
    async def create(cls, address: str, **kwargs):
        supervisor_address = await get_supervisor_address(address)
        return SessionWebAPI(address, 'create', supervisor_address, **kwargs)

    async def create_session(self, session_id: str) -> str:
        return await self._call_method({}, 'create_session', session_id)

    async def delete_session(self, session_id: str):
        return await self._call_method({}, 'delete_session', session_id)
