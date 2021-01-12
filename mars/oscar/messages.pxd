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

from libc.stdint cimport int32_t

from .core cimport ActorRef


cpdef enum MessageType:
    create_actor = 0
    destroy_actor = 1
    has_actor = 2
    result = 3
    error = 4
    send_all = 5
    tell_all = 6
    send_chunk_start = 7
    send_chunk = 8
    send_chunk_end = 9
    tell_chunk_start = 10
    tell_chunk = 11
    tell_chunk_end = 12


cdef class _BASE_ACTOR_MESSAGE:
    cdef public int32_t message_type
    cdef public bytes message_id
    cdef public int32_t from_index
    cdef public int32_t to_index


cdef class CREATE_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref
    cdef public object actor_cls
    cdef public tuple args
    cdef public dict kwargs


cdef class DESTROY_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref


cdef class HAS_ACTOR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref


cdef class RESULT_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public object result


cdef class ERROR_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public object error_type
    cdef public object error
    cdef public object traceback


cdef class SEND_MESSAGE(_BASE_ACTOR_MESSAGE):
    cdef public ActorRef actor_ref
    cdef public object message
