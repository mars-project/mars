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
from typing import Tuple, Union, Type

from ....utils import to_binary
from ...api import Actor
from ...core import ActorRef
from ...context import BaseActorContext
from ...utils import create_actor_ref
from .allocate_strategy import AllocateStrategy, AddressSpecified
from .core import ActorCaller
from .message import DEFAULT_PROTOCOL, new_message_id, _MessageBase, \
    ResultMessage, ErrorMessage, CreateActorMessage, HasActorMessage, \
    DestroyActorMessage, ActorRefMessage, SendMessage
from .router import Router


class MarsActorContext(BaseActorContext):
    __slots__ = '_address', '_caller'

    support_allocate_strategy = True

    def __init__(self, address: str = None):
        self._address = address
        self._caller = ActorCaller()

    def __del__(self):
        self._caller.cancel_tasks()

    async def _call(self, address: str, message: _MessageBase):
        return await self._caller.call(Router.get_instance_or_empty(),
                                       address, message)

    @staticmethod
    def _process_result_message(message: Union[ResultMessage, ErrorMessage]):
        if isinstance(message, ResultMessage):
            return message.result
        else:
            raise message.error.with_traceback(message.traceback)

    async def create_actor(self, actor_cls: Type[Actor], *args, uid=None,
                           address: str = None, **kwargs) -> ActorRef:
        router = Router.get_instance_or_empty()
        address = address or self._address or router.external_address
        allocate_strategy = kwargs.get('allocate_strategy', None)
        if isinstance(allocate_strategy, AllocateStrategy):
            allocate_strategy = kwargs.pop('allocate_strategy')
        else:
            allocate_strategy = AddressSpecified(address)
        create_actor_message = CreateActorMessage(
            new_message_id(), actor_cls, to_binary(uid),
            args, kwargs, allocate_strategy,
            protocol=DEFAULT_PROTOCOL
        )
        result = await self._call(address, create_actor_message)
        return self._process_result_message(result)

    async def has_actor(self, actor_ref: ActorRef) -> bool:
        message = HasActorMessage(
            new_message_id(), actor_ref,
            protocol=DEFAULT_PROTOCOL)
        result = await self._call(actor_ref.address, message)
        return self._process_result_message(result)

    async def destroy_actor(self, actor_ref: ActorRef):
        message = DestroyActorMessage(
            new_message_id(), actor_ref,
            protocol=DEFAULT_PROTOCOL)
        result = await self._call(actor_ref.address, message)
        return self._process_result_message(result)

    async def actor_ref(self, *args, **kwargs):
        actor_ref = create_actor_ref(*args, **kwargs)
        message = ActorRefMessage(
            new_message_id(), actor_ref,
            protocol=DEFAULT_PROTOCOL)
        result = await self._call(actor_ref.address, message)
        return self._process_result_message(result)

    async def send(self,
                   actor_ref: ActorRef,
                   message: Tuple,
                   wait_response: bool = True):
        message = SendMessage(
            new_message_id(), actor_ref,
            message, protocol=DEFAULT_PROTOCOL)
        call = self._call(actor_ref.address, message)
        if wait_response:
            return self._process_result_message(await call)
        else:
            future = asyncio.create_task(call)
            await asyncio.sleep(0)
            return future
