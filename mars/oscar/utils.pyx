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

import typing

import numpy as np

from .._utils cimport to_str


cpdef bytes new_actor_id():
    return np.random.bytes(32)


def create_actor_ref(*args, **kwargs):
    """
    Create an actor reference.

    Returns
    -------
    ActorRef
    """
    from .core import ActorRef

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
    elif len(args) == 1 and isinstance(args[0], ActorRef):
        uid = args[0].uid
        address = to_str(address or args[0].address)
    elif len(args) == 1:
        uid = args[0]

    if uid is None:
        raise ValueError('Actor uid should be provided')

    return ActorRef(address, uid)


cdef set _is_async_generator_typecache = set()


cdef bint is_async_generator(obj):
    cdef type tp = type(obj)
    if tp in _is_async_generator_typecache:
        return True

    if isinstance(obj, typing.AsyncGenerator):
        if len(_is_async_generator_typecache) < 100:
            _is_async_generator_typecache.add(tp)
        return True
    else:
        return False
