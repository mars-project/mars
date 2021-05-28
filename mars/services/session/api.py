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
from typing import Any, Union

from ... import oscar as mo
from ...lib.aio import alru_cache
from .supervisor import CustomLogMetaActor, SessionManagerActor, SessionActor
from .worker import CustomLogActor


class AbstractSessionAPI(ABC):

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

    @abstractmethod
    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
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


class SessionAPI(AbstractSessionAPI):
    def __init__(self,
                 address: str,
                 session_manager: Union[SessionManagerActor, mo.ActorRef]):
        self._address = address
        self._session_manager_ref = session_manager

    @classmethod
    @alru_cache
    async def create(cls, address: str, **kwargs) -> "SessionAPI":
        if kwargs:  # pragma: no cover
            raise TypeError(f'SessionAPI.create '
                            f'got unknown arguments: {list(kwargs)}')
        session_manager = await mo.actor_ref(
            address, SessionManagerActor.default_uid())
        return SessionAPI(address, session_manager)

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

    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        return await self._session_manager_ref.get_last_idle_time(session_id)

    @alru_cache(cache_exceptions=False)
    async def _get_session_ref(self, session_id: str) -> Union[SessionActor, mo.ActorRef]:
        return await self._session_manager_ref.get_session_ref(session_id)

    async def create_remote_object(self,
                                   session_id: str,
                                   name: str,
                                   object_cls,
                                   *args, **kwargs):
        session = await self._get_session_ref(session_id)
        return await session.create_remote_object(name, object_cls, *args, **kwargs)

    async def get_remote_object(self,
                                session_id: str,
                                name: str):
        session = await self._get_session_ref(session_id)
        return await session.get_remote_object(name)

    async def destroy_remote_object(self,
                                    session_id: str,
                                    name: str):
        session = await self._get_session_ref(session_id)
        return await session.destroy_remote_object(name)

    @alru_cache(cache_exceptions=False)
    async def _get_custom_log_meta_ref(self, session_id: str) -> \
            Union[CustomLogMetaActor, mo.ActorRef]:
        session = await self._get_session_ref(session_id)
        return await mo.actor_ref(
            mo.ActorRef(session.address,
                        CustomLogMetaActor.gen_uid(session_id)))

    async def register_custom_log_path(self,
                                       session_id: str,
                                       tileable_op_key: str,
                                       chunk_op_key: str,
                                       worker_address: str,
                                       log_path: str):
        custom_log_meta_ref = await self._get_custom_log_meta_ref(session_id)
        return await custom_log_meta_ref.register_custom_log_path(
            tileable_op_key, chunk_op_key, worker_address, log_path)

    @classmethod
    async def new_custom_log_dir(cls, address: str, session_id: str):
        ref = await mo.actor_ref(mo.ActorRef(
            address, CustomLogActor.default_uid()))
        return await ref.new_custom_log_dir(session_id)


class MockSessionAPI(SessionAPI):
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
