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

import asyncio
import functools
from typing import Dict, Optional

from .... import oscar as mo
from ...cluster import ClusterAPI


class SessionManagerActor(mo.Actor):
    def __init__(self, service_config: Optional[Dict] = None):
        self._session_refs: Dict[str, mo.ActorRef] = dict()
        self._cluster_api: Optional[ClusterAPI] = None
        self._service_config = service_config or dict()

    async def __post_create__(self):
        self._cluster_api = await ClusterAPI.create(self.address)

    async def create_session(self,
                             session_id: str,
                             create_services: bool = True):
        if session_id in self._session_refs:
            raise mo.Return(self._session_refs[session_id])

        address = (await self._cluster_api.get_supervisors_by_keys([session_id]))[0]
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

    async def create_services(self):
        from ...meta import MetaAPI
        from ...lifecycle import LifecycleAPI
        from ...scheduling import SchedulingAPI
        from ...task import TaskAPI

        services = set(self._service_config['services'])

        self._meta_api = self._scheduling_api = self._task_api = None

        if services and 'meta' in services:
            self._meta_api = await MetaAPI.create_session(
                self._session_id, self.address)
        if services and 'lifecycle' in services:
            self._lifecycle_api = await LifecycleAPI.create_session(
                self._session_id, self.address)
        if services and 'scheduling' in services:
            self._scheduling_api = await SchedulingAPI.create_session(
                self._session_id, self.address, self._service_config)
        if services and 'task' in services:
            self._task_api = await TaskAPI.create_session(
                self._session_id, self.address)

    async def get_last_idle_time(self):
        if self._task_api is None:
            return None
        return await self._task_api.get_last_idle_time()

    async def create_remote_object(self, name: str,
                                   object_cls, *args, **kwargs):
        return await mo.create_actor(
            RemoteObjectActor, object_cls, args, kwargs,
            address=self.address, uid=name)

    async def get_remote_object(self, name: str):
        return await mo.actor_ref(mo.ActorRef(self.address, name))

    async def destroy_remote_object(self, name: str):
        return await mo.destroy_actor(mo.ActorRef(self.address, name))

    async def __pre_destroy__(self):
        from ...meta import MetaAPI
        from ...lifecycle import LifecycleAPI
        from ...scheduling import SchedulingAPI
        from ...task import TaskAPI

        if self._task_api:
            await TaskAPI.destroy_session(self._session_id, self.address)
        if self._lifecycle_api:
            await LifecycleAPI.destroy_session(self._session_id, self.address)
        if self._meta_api:
            await MetaAPI.destroy_session(self._session_id, self.address)
        if self._scheduling_api:
            await SchedulingAPI.destroy_session(self._session_id, self.address)

            await mo.destroy_actor(self._custom_log_meta_ref)


class RemoteObjectActor(mo.Actor):
    def __init__(self, object_cls, args, kwargs):
        self._object = object_cls(*args, **kwargs)

    def __getattr__(self, attr):
        func = getattr(self._object, attr)
        if not callable(func):  # pragma: no cover
            return object.__getattribute__(self._object, attr)

        @functools.wraps(func)
        async def wrap(*args, **kwargs):
            part_func = functools.partial(func, *args, **kwargs)
            loop = asyncio.get_running_loop()
            return loop.run_in_executor(None, part_func)

        return wrap
