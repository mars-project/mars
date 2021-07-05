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

from typing import Dict, List, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....utils import parse_readable_size
from ..core import SessionInfo
from ..supervisor import CustomLogMetaActor, SessionManagerActor, SessionActor
from ..worker import CustomLogActor
from .core import AbstractSessionAPI


class SessionAPI(AbstractSessionAPI):
    def __init__(self,
                 address: str,
                 session_manager: Union[SessionManagerActor, mo.ActorRef]):
        self._address = address
        self._session_manager_ref = session_manager

    @classmethod
    @alru_cache(cache_exceptions=False)
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

    async def get_sessions(self) -> List[SessionInfo]:
        return await self._session_manager_ref.get_sessions()

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
        try:
            ref = await mo.actor_ref(mo.ActorRef(
                address, CustomLogActor.default_uid()))
        except mo.ActorNotExist:
            return
        return await ref.new_custom_log_dir(session_id)

    async def fetch_tileable_op_logs(self,
                                     session_id: str,
                                     tileable_op_key: str,
                                     chunk_op_key_to_offsets: Dict[str, List[int]],
                                     chunk_op_key_to_sizes: Dict[str, List[int]]) -> Dict:
        custom_log_meta_ref = await self._get_custom_log_meta_ref(session_id)
        chunk_op_key_to_arr_paths = \
            await custom_log_meta_ref.get_tileable_op_log_paths(tileable_op_key)
        if chunk_op_key_to_arr_paths is None:
            return
        worker_to_kwds = dict()
        for chunk_op_key, addr_path in chunk_op_key_to_arr_paths.items():
            worker_address, log_path = addr_path
            if isinstance(chunk_op_key_to_offsets, dict):
                offset = chunk_op_key_to_offsets.get(chunk_op_key, 0)
            elif isinstance(chunk_op_key_to_offsets, str):
                offset = int(parse_readable_size(chunk_op_key_to_offsets)[0])
            elif isinstance(chunk_op_key_to_offsets, int):
                offset = chunk_op_key_to_offsets
            else:
                offset = 0
            if isinstance(chunk_op_key_to_sizes, dict):
                size = chunk_op_key_to_sizes.get(chunk_op_key, -1)
            elif isinstance(chunk_op_key_to_sizes, str):
                size = int(parse_readable_size(chunk_op_key_to_sizes)[0])
            elif isinstance(chunk_op_key_to_sizes, int):
                size = chunk_op_key_to_sizes
            else:
                size = -1
            if worker_address not in worker_to_kwds:
                worker_to_kwds[worker_address] = {
                    'chunk_op_keys': [],
                    'log_paths': [],
                    'offsets': [],
                    'sizes': []
                }
            kwds = worker_to_kwds[worker_address]
            kwds['chunk_op_keys'].append(chunk_op_key)
            kwds['log_paths'].append(log_path)
            kwds['offsets'].append(offset)
            kwds['sizes'].append(size)
        result = dict()
        for worker, kwds in worker_to_kwds.items():
            custom_log_ref = await mo.actor_ref(
                mo.ActorRef(worker, CustomLogActor.default_uid()))
            chunk_op_keys = kwds.pop('chunk_op_keys')
            logs = await custom_log_ref.fetch_logs(**kwds)
            for chunk_op_key, log_result in zip(chunk_op_keys, logs):
                result[chunk_op_key] = log_result
        return result


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
