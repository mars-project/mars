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

import asyncio
import functools
from typing import Dict, List, Optional

from .... import oscar as mo
from ....utils import to_binary
from ...cluster import ClusterAPI
from ...core import NodeRole, create_service_session, \
    destroy_service_session
from ..core import SessionInfo


class SessionManagerActor(mo.Actor):
    def __init__(self, service_config: Optional[Dict] = None):
        self._session_refs: Dict[str, mo.ActorRef] = dict()
        self._cluster_api: Optional[ClusterAPI] = None
        self._service_config = service_config or dict()

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)

    async def __pre_destroy__(self):
        await asyncio.gather(*[
            mo.destroy_actor(ref) for ref in self._session_refs.values()
        ])

    async def create_session(self,
                             session_id: str,
                             create_services: bool = True):
        if session_id in self._session_refs:
            raise mo.Return(self._session_refs[session_id])

        [address] = await self._cluster_api.get_supervisors_by_keys([session_id])
        session_actor_ref = await mo.create_actor(
            SessionActor, session_id, self._service_config,
            address=address,
            uid=SessionActor.gen_uid(session_id),
            allocate_strategy=mo.allocate_strategy.Random())
        self._session_refs[session_id] = session_actor_ref

        # sync ref to other managers
        for supervisor_address in await self._cluster_api.get_supervisors():
            if supervisor_address == self.address:
                continue
            session_manager_ref = await mo.actor_ref(
                supervisor_address, SessionManagerActor.default_uid())
            await session_manager_ref.add_session_ref(
                session_id, session_actor_ref)

        # let session actor create session-related services
        if create_services:
            yield session_actor_ref.create_services()

        raise mo.Return(session_actor_ref)

    def get_sessions(self) -> List[SessionInfo]:
        return [SessionInfo(session_id=session_id) for session_id in self._session_refs.keys()]

    def get_session_ref(self,
                        session_id: str):
        return self._session_refs[session_id]

    def add_session_ref(self,
                        session_id: str,
                        session_actor_ref: mo.ActorRef):
        self._session_refs[session_id] = session_actor_ref

    def remove_session_ref(self, session_id: str):
        del self._session_refs[session_id]

    def has_session(self, session_id: str):
        return session_id in self._session_refs

    async def delete_session(self, session_id):
        session_actor_ref = self._session_refs.pop(session_id)
        await mo.destroy_actor(session_actor_ref)

        # sync removing to other managers
        for supervisor_address in await self._cluster_api.get_supervisors():
            if supervisor_address == self.address:
                continue
            session_manager_ref = await mo.actor_ref(
                supervisor_address, SessionManagerActor.default_uid())
            await session_manager_ref.remove_session_ref(session_id)

    async def get_last_idle_time(self, session_id=None):
        if session_id is not None:
            session = self._session_refs[session_id]
            raise mo.Return(await session.get_last_idle_time())
        else:
            all_last_idle_time = yield asyncio.gather(
                *[session.get_last_idle_time() for session in self._session_refs.values()])
            if any(last_idle_time is None for last_idle_time in all_last_idle_time):
                raise mo.Return(None)
            else:
                raise mo.Return(max(all_last_idle_time))


class SessionActor(mo.Actor):
    def __init__(self, session_id: str,
                 service_config: Dict):
        self._session_id = session_id

        self._meta_api = None
        self._lifecycle_api = None
        self._task_api = None
        self._scheduling_api = None

        self._service_config = service_config

        self._custom_log_meta_ref = None

    @classmethod
    def gen_uid(cls, session_id):
        return f'{session_id}_session_actor'

    async def __post_create__(self):
        from .custom_log import CustomLogMetaActor

        self._custom_log_meta_ref = await mo.create_actor(
            CustomLogMetaActor, self._session_id,
            address=self.address,
            uid=CustomLogMetaActor.gen_uid(self._session_id))

    async def __pre_destroy__(self):
        await destroy_service_session(
            NodeRole.SUPERVISOR, self._service_config, self._session_id, self.address)
        await mo.destroy_actor(self._custom_log_meta_ref)

    async def create_services(self):
        from ...task import TaskAPI
        await create_service_session(
            NodeRole.SUPERVISOR, self._service_config, self._session_id, self.address)
        if 'task' in self._service_config['services']:
            self._task_api = await TaskAPI.create(
                session_id=self._session_id, address=self.address)

    async def get_last_idle_time(self):
        if self._task_api is None:
            return None
        return await self._task_api.get_last_idle_time()

    async def create_remote_object(self, name: str,
                                   object_cls, *args, **kwargs):
        return await mo.create_actor(
            RemoteObjectActor, object_cls, args, kwargs,
            address=self.address, uid=to_binary(name))

    async def get_remote_object(self, name: str):
        return await mo.actor_ref(mo.ActorRef(self.address, to_binary(name)))

    async def destroy_remote_object(self, name: str):
        return await mo.destroy_actor(mo.ActorRef(self.address, to_binary(name)))


class RemoteObjectActor(mo.Actor):
    def __init__(self, object_cls, args, kwargs):
        self._object = object_cls(*args, **kwargs)

    def __getattr__(self, attr):
        func = getattr(self._object, attr)
        if not callable(func):  # pragma: no cover
            return object.__getattribute__(self._object, attr)

        @functools.wraps(func)
        async def wrap(*args, **kwargs):
            # return coroutine to not block current actor
            if asyncio.iscoroutinefunction(func):
                return func(*args, **kwargs)
            else:
                # for sync call, running in thread
                return asyncio.to_thread(func, *args, **kwargs)

        return wrap
