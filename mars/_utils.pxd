# Copyright 1999-2021 Alibaba Group Holding Ltd.
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


cdef class TypeDispatcher:
    cdef dict _handlers
    cdef dict _lazy_handlers
    cdef dict _inherit_handlers
    cdef object __weakref__

    cpdef void register(self, object type_, object handler)
    cpdef void unregister(self, object type_)
    cdef _reload_lazy_handlers(self)
    cpdef get_handler(self, object type_)


cpdef str to_str(s, encoding=*)
cpdef bytes to_binary(s, encoding=*)
cpdef unicode to_text(s, encoding=*)
cpdef register_tokenizer(cls, handler)
cpdef void reset_id_random_seed() except *
cpdef bytes new_random_id(int byte_len)
