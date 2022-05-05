# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from enum import Enum
from types import TracebackType
from typing import Any, Type

from ..core import ActorRef

DEFAULT_PROTOCOL: int = 0

class MessageType(Enum):
    control = 0
    result = 1
    error = 2
    create_actor = 3
    destroy_actor = 4
    has_actor = 5
    actor_ref = 6
    send = 7
    tell = 8
    cancel = 9

class ControlMessageType(Enum):
    stop = 0
    restart = 1
    sync_config = 2
    get_config = 3
    wait_pool_recovered = 4
    add_sub_pool_actor = 5

class _MessageBase:
    message_type: MessageType
    protocol: int
    message_id: bytes
    message_trace: list
    profiling_context: Any

    def __init__(
        self,
        message_id: bytes = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
        profiling_context: Any = None,
    ): ...
    def __repr__(self): ...

class ControlMessage(_MessageBase):
    message_type = MessageType.control

    address: str
    control_message_type: ControlMessageType
    content: Any

    def __init__(
        self,
        message_id: bytes = None,
        address: str = None,
        control_message_type: ControlMessageType = None,
        content: Any = None,
        protocol: int = DEFAULT_PROTOCOL,
    ): ...

class ResultMessage(_MessageBase):
    message_type = MessageType.result

    result: Any

    def __init__(
        self,
        message_id: bytes = None,
        result: Any = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
        profiling_context: Any = None,
    ): ...

class ErrorMessage(_MessageBase):
    message_type = MessageType.error

    address: str
    pid: int
    error_type: Type
    error: BaseException
    traceback: TracebackType

    def __init__(
        self,
        message_id: bytes = None,
        address: str = None,
        pid: int = -1,
        error_type: Type[BaseException] = None,
        error: BaseException = None,
        traceback: TracebackType = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...
    def as_instanceof_cause(self) -> BaseException: ...

class CreateActorMessage(_MessageBase):
    message_type = MessageType.create_actor

    actor_cls: Type
    actor_id: bytes
    args: tuple
    kwargs: dict
    allocate_strategy: Any
    from_main: bool

    def __init__(
        self,
        message_id: bytes = None,
        actor_cls: Type = None,
        actor_id: bytes = None,
        args: tuple = None,
        kwargs: dict = None,
        allocate_strategy: Any = None,
        from_main: bool = False,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...

class DestroyActorMessage(_MessageBase):
    message_type = MessageType.destroy_actor

    actor_ref: ActorRef
    from_main: bool

    def __init__(
        self,
        message_id: bytes = None,
        actor_ref: ActorRef = None,
        from_main: bool = False,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...

class HasActorMessage(_MessageBase):
    message_type = MessageType.has_actor

    actor_ref: ActorRef

    def __init__(
        self,
        message_id: bytes = None,
        actor_ref: ActorRef = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...

class ActorRefMessage(_MessageBase):
    message_type = MessageType.actor_ref

    actor_ref: ActorRef

    def __init__(
        self,
        message_id: bytes = None,
        actor_ref: ActorRef = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...

class SendMessage(_MessageBase):
    message_type = MessageType.send

    actor_ref: ActorRef
    content: Any

    def __init__(
        self,
        message_id: bytes = None,
        actor_ref: ActorRef = None,
        content: object = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
        profiling_context: Any = None,
    ): ...

class TellMessage(SendMessage):
    message_type = MessageType.tell

class CancelMessage(_MessageBase):
    message_type = MessageType.cancel

    address: str
    cancel_message_id: bytes

    def __init__(
        self,
        message_id: bytes = None,
        address: str = None,
        cancel_message_id: bytes = None,
        protocol: int = DEFAULT_PROTOCOL,
        message_trace: list = None,
    ): ...

class DeserializeMessageFailed(RuntimeError):
    def __init__(self, message_id: bytes): ...
    def __str__(self): ...

def reset_random_seed(): ...
def new_message_id() -> bytes: ...
