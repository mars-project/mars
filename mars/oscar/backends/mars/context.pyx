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

from ...core cimport _Actor, ActorRef
from ...context cimport BaseActorContext
from ...errors import ActorNotExist, ActorAlreadyExist


cdef class MessageDispatcher:
    pass


cdef class MarsActorContext(BaseActorContext):
    cdef public dict actors

    def __init__(self, str address=None, index=0):
        self.actors = dict()

    async def create_actor(self, object actor_cls, *args, object uid=None, object address=None, **kwargs):
        cdef _Actor actor

        if uid in self.actors:
            raise ActorAlreadyExist(f'Actor {uid} already exist, cannot create')

        actor = actor_cls(*args, **kwargs)
        actor.uid = uid
        actor.address = address
        self.actors[uid] = actor
        await actor.__post_create__()
        return ActorRef(address, uid)

    async def has_actor(self, ActorRef actor_ref):
        return actor_ref.uid in self.actors

    async def destroy_actor(self, ActorRef actor_ref):
        cdef _Actor actor

        try:
            actor = self.actors[actor_ref.uid]
        except KeyError:
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist')

        await actor.__pre_destroy__()
        del self.actors[actor_ref.uid]
        return actor_ref.uid

    async def send(self, ActorRef actor_ref, object message, bint wait_response=True):
        try:
            actor = self.actors[actor_ref.uid]
        except KeyError:
            raise ActorNotExist(f'Actor {actor_ref.uid} does not exist')
        ret = await actor.__on_receive__(message)
        return ret if wait_response else None

