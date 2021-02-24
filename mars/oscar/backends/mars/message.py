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
from enum import Enum
from types import TracebackType
from typing import Any, Type, Tuple, Dict, List

import numpy as np

from ....lib.tblib import pickling_support
from ....serialization.core import Serializer, pickle, pickle_buffers, unpickle_buffers
from ....utils import classproperty, implements
from ...core import ActorRef


# make sure traceback can be pickled
pickling_support.install()


DEFAULT_PROTOCOL = 0


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


class ControlMessageType(Enum):
    stop = 0
    restart = 1
    sync_config = 2
    get_config = 3


class _MessageBase(ABC):
    __slots__ = 'protocol', 'message_id', 'scoped_message_ids'

    def __init__(self,
                 message_id: bytes,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        self.message_id = message_id
        # A message can be in the scope of other messages,
        # this is mainly used for detecting deadlocks,
        # e.g. Actor `A` sent a message(id: 1) to actor `B`,
        # in the processing of `B`, it sent back a message(id: 2) to `A`,
        # deadlock happens, because `A` is still waiting for reply from `B`.
        # In this case, the `scoped_message_ids` will be [1, 2],
        # `A` will find that id:1 already exists in inbox,
        # thus deadlock detected.
        self.scoped_message_ids = scoped_message_ids
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        self.protocol = protocol

    @classproperty
    @abstractmethod
    def message_type(self) -> MessageType:
        """
        Message type.

        Returns
        -------
        message_type: MessageType
            message type.
        """


class ControlMessage(_MessageBase):
    __slots__ = 'control_message_type', 'content'

    def __init__(self,
                 message_id: bytes,
                 control_message_type: ControlMessageType,
                 content: Any,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.control_message_type = control_message_type
        self.content = content

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.control


class ResultMessage(_MessageBase):
    __slots__ = 'result',

    def __init__(self,
                 message_id: bytes,
                 result: Any,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.result = result

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.result


class ErrorMessage(_MessageBase):
    __slots__ = 'error_type', 'error', 'traceback'

    def __init__(self,
                 message_id: bytes,
                 error_type: Type[BaseException],
                 error: BaseException,
                 traceback: TracebackType,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.error_type = error_type
        self.error = error
        self.traceback = traceback

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.error


class CreateActorMessage(_MessageBase):
    __slots__ = 'actor_cls', 'actor_id', 'args', 'kwargs', 'allocate_strategy'

    def __init__(self,
                 message_id: bytes,
                 actor_cls: Type,
                 actor_id: bytes,
                 args: Tuple,
                 kwargs: Dict,
                 allocate_strategy,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.actor_cls = actor_cls
        self.actor_id = actor_id
        self.args = args
        self.kwargs = kwargs
        self.allocate_strategy = allocate_strategy

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.create_actor


class DestroyActorMessage(_MessageBase):
    __slots__ = 'actor_ref', 'from_main'

    def __init__(self,
                 message_id: bytes,
                 actor_ref: ActorRef,
                 from_main: bool = False,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.actor_ref = actor_ref
        self.from_main = from_main

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.destroy_actor


class HasActorMessage(_MessageBase):
    __slots__ = 'actor_ref',

    def __init__(self,
                 message_id: bytes,
                 actor_ref: ActorRef,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.actor_ref = actor_ref

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.has_actor


class ActorRefMessage(_MessageBase):
    __slots__ = 'actor_ref',

    def __init__(self,
                 message_id: bytes,
                 actor_ref: ActorRef,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.actor_ref = actor_ref

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.actor_ref


class SendMessage(_MessageBase):
    __slots__ = 'actor_ref', 'content',

    def __init__(self,
                 message_id: bytes,
                 actor_ref: ActorRef,
                 content: Any,
                 scoped_message_ids: List[bytes] = None,
                 protocol: int = None):
        super().__init__(message_id,
                         scoped_message_ids=scoped_message_ids,
                         protocol=protocol)
        self.actor_ref = actor_ref
        self.content = content

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.send


class TellMessage(SendMessage):
    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.tell


class DesrializeMessageFailed(Exception):
    def __init__(self, message_id):
        self.message_id = message_id

    def __str__(self):
        return f'Deserialize {self.message_id} failed'


class MessageSerializer(Serializer):
    serializer_name = 'actor_message'

    def serialize(self, obj: _MessageBase):
        assert obj.protocol == 0, 'only support protocol 0 for now'
        # use pickle for protocol 0
        return {'protocol': 0, 'message_id': obj.message_id}, \
               pickle_buffers(obj)

    def deserialize(self, header: Dict, buffers: List):
        protocol = header['protocol']
        assert protocol == 0, 'only support protocol 0 for now'
        message_id = header['message_id']
        try:
            return unpickle_buffers(buffers)
        except pickle.UnpicklingError as e:  # pragma: no cover
            raise DesrializeMessageFailed(message_id) from e


# register message serializer
MessageSerializer.register(_MessageBase)


def new_message_id():
    return np.random.bytes(32)
