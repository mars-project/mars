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

from libc.stdint cimport uint8_t, int32_t

from ..lib.tblib import pickling_support

pickling_support.install()

# Internal message types includes:
# 1) create actor
# 2) destroy actor
# 3) has actor
# 4) result that return back by the actor in a different process or machine
# 5) error that throws back by the actor in a different process or machine
# 6) send all, the send actor message
# 7) tell all, the tell actor message
cpdef uint8_t CREATE_ACTOR = MessageType.create_actor
cpdef uint8_t DESTROY_ACTOR = MessageType.destroy_actor
cpdef uint8_t HAS_ACTOR = MessageType.has_actor
cpdef uint8_t RESULT = MessageType.result
cpdef uint8_t ERROR = MessageType.error
cpdef uint8_t SEND_ALL = MessageType.send_all
cpdef uint8_t TELL_ALL = MessageType.tell_all


cpdef enum MessageSerialType:
    # Internal message serialized type
    null = 0
    raw_bytes = 1
    pickle = 2


cdef uint8_t NONE = MessageSerialType.null
cdef uint8_t RAW_BYTES = MessageSerialType.raw_bytes
cdef uint8_t PICKLE = MessageSerialType.pickle


cdef int32_t DEFAULT_PROTOCOL = 0


cdef class CREATE_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, int32_t message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, ActorRef actor_ref=None,
                 object actor_cls=None, tuple args=None, dict kwargs=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref
        self.actor_cls = actor_cls
        self.args = args
        self.kwargs = kwargs


cdef class DESTROY_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, int32_t message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, object actor_ref=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref


cdef class HAS_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, int32_t message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, ActorRef actor_ref=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref


cdef class RESULT_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, int32_t message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, object result=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.result = result


cdef class ERROR_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, int32_t message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, object error_type=None,
                 object error=None, object traceback=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.error_type = error_type
        self.error = error
        self.traceback = traceback


cdef class SEND_MESSAGE(_BASE_ACTOR_MESSAGE):
    def __init__(self, object message_type=-1, bytes message_id=None,
                 int32_t from_index=0, int32_t to_index=0, ActorRef actor_ref=None,
                 object message=None):
        self.message_type = message_type
        self.message_id = message_id
        self.from_index = from_index
        self.to_index = to_index
        self.actor_ref = actor_ref
        self.message = message
