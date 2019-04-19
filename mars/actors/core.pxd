#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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


cdef class ActorRef:
    cdef public str address
    cdef public object uid
    cdef public object _ctx

    cpdef object send(self, object message, bint wait=*, object callback=*)
    cpdef object tell(self, object message, object delay=*, bint wait=*,
                      object callback=*)
    cpdef object destroy(self, bint wait=*, object callback=*)


cdef class Actor:
    cdef str _address
    cdef object _uid
    cdef object _ctx

    cpdef ActorRef ref(self)
    cpdef post_create(self)
    cpdef on_receive(self, message)
    cpdef pre_destroy(self)


cdef class _FunctionActor(Actor):
    cpdef on_receive(self, message)


cpdef object create_actor_pool(str address=*, int n_process=*, object distributor=*,
                               object parallel=*, str backend=*)
cpdef object new_client(object parallel=*, str backend=*)