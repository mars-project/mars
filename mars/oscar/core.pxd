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


cdef class ActorRef:
    cdef public str address
    cdef public object uid
    cdef public object _ctx
    cdef dict _methods

    cpdef object send(self, object message, bint wait=*, object callback=*)
    cpdef object tell(self, object message, object delay=*, bint wait=*,
                      object callback=*)
    cpdef object destroy(self, bint wait=*, object callback=*)


cdef class Actor:
    cdef str _address
    cdef object _uid
    cdef object _ctx

    cpdef ActorRef ref(self)
    cpdef __post_create__(self)
    cpdef __on_receive__(self, tuple action)
    cpdef __pre_destroy__(self)


cdef class ActorEnvironment:
    cdef public dict actor_locks
    cdef public object address
