#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from ..core cimport ActorRef


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


cpdef int32_t unpack_message_type_value(bytes binary)
cpdef object unpack_message_type(bytes binary)
cpdef bytes unpack_message_id(bytes binary)
cpdef bytes pack_actor_ref(ActorRef actor_ref)
cpdef void unpack_actor_ref(bytes binary, ActorRef actor_ref)
cpdef object unpack_send_message(bytes message)
cpdef object pack_send_message(int32_t from_index, int32_t to_index, ActorRef actor_ref, object message,
                               object write=*, int32_t protocol=*)
cpdef object unpack_send_message(bytes message)
cpdef object pack_tell_message(int32_t from_index, int32_t to_index, ActorRef actor_ref, object message,
                               object write=*, int32_t protocol=*)
cpdef object unpack_tell_message(bytes message)
cpdef tuple get_index(bytes binary, object calc_from_uid)
cpdef object pack_create_actor_message(int32_t from_index, int32_t to_index, ActorRef actor_ref, object actor_cls,
                                       tuple args, dict kw, object write=*, int32_t protocol=*)
cpdef object unpack_create_actor_message(bytes binary)
cpdef object pack_destroy_actor_message(int32_t from_index, int32_t to_index, ActorRef actor_ref,
                                        object write=*, int32_t protocol=*)
cpdef object unpack_destroy_actor_message(bytes binary)
cpdef object pack_has_actor_message(int32_t from_index, int32_t to_index, ActorRef actor_ref,
                                    object write=*, int32_t protocol=*)
cpdef object unpack_has_actor_message(bytes binary)
cpdef object pack_result_message(bytes message_id, int32_t from_index, int32_t to_index, object result,
                                 object write=*, int32_t protocol=*)
cpdef object unpack_result_message(bytes binary, object from_index=*, object to_index=*)
cpdef object pack_error_message(bytes message_id, int32_t from_index, int32_t to_index,
                                object error_type, object error, object tb,
                                object write=*, int32_t protocol=*)
cpdef object unpack_error_message(bytes binary)
