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
from ...core cimport _Actor, ActorRef
from ...context cimport BaseActorContext
from ...errors import ActorNotExist, ActorAlreadyExist
# TODO(fyrestone): Move lazy_import to oscar
from ....utils import lazy_import

ray = lazy_import("ray")


cdef class RayActorContext(BaseActorContext):
    def __init__(self, str address=None, index=0):
        pass

    async def create_actor(self, object actor_cls, *args, object uid=None, object address=None, **kwargs):
        uid = str(uid)
        try:
            # TODO(fyrestone): We should make the actor dead when current job is dropped.
            actor_handle = ray.remote(actor_cls).options(
                name=uid, lifetime="detached").remote(*args, **kwargs)
        except ValueError:
            raise ActorAlreadyExist(f'Actor {uid} already exist, cannot create')
        await asyncio.gather(
            actor_handle._set_uid.remote(uid),
            actor_handle._set_address.remote(address))
        await actor_handle.__post_create__.remote()
        return ActorRef(address, uid)

    async def has_actor(self, ActorRef actor_ref):
        try:
            # Maybe poor performance.
            ray.get_actor(actor_ref.uid)
            return True
        except Exception:
            return False

    async def destroy_actor(self, ActorRef actor_ref):
        try:
            actor_handle = ray.get_actor(actor_ref.uid)
        except (KeyError, ValueError):
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist')

        await actor_handle.__pre_destroy__.remote()
        ray.kill(actor_handle)
        return actor_ref.uid

    async def send(self, ActorRef actor_ref, object message, bint wait_response=True):
        try:
            actor_handle = ray.get_actor(actor_ref.uid)
        except ValueError:
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist')
        ret = await actor_handle.__on_receive__.remote(message)
        return ret if wait_response else None

    async def actor_ref(self, ActorRef ref):
        return ref
