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

from random import getrandbits
from typing import AsyncGenerator

from .._utils cimport to_str
from .core cimport ActorRef, LocalActorRef


cpdef bytes new_actor_id():
    return getrandbits(256).to_bytes(32, "little")


def create_actor_ref(*args, **kwargs):
    """
    Create an actor reference.

    Returns
    -------
    ActorRef
    """

    cdef str address
    cdef object uid

    address = to_str(kwargs.pop('address', None))
    uid = kwargs.pop('uid', None)

    if kwargs:
        raise ValueError('Only `address` or `uid` keywords are supported')

    if len(args) == 2:
        if address:
            raise ValueError('address has been specified')
        address = to_str(args[0])
        uid = args[1]
    elif len(args) == 1:
        tp0 = type(args[0])
        if tp0 is ActorRef or tp0 is LocalActorRef:
            uid = args[0].uid
            address = to_str(address or args[0].address)
        else:
            uid = args[0]

    if uid is None:
        raise ValueError('Actor uid should be provided')

    return ActorRef(address, uid)


cdef set _is_async_generator_typecache = set()


cdef bint is_async_generator(obj):
    cdef type tp = type(obj)
    if tp in _is_async_generator_typecache:
        return True

    if isinstance(obj, AsyncGenerator):
        if len(_is_async_generator_typecache) < 100:
            _is_async_generator_typecache.add(tp)
        return True
    else:
        return False
