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

import sys
import asyncio

import numpy as np

from ..._utils cimport to_str
from ...utils import aio_run

aio_run = aio_run

cdef bint _LESS_PY37 = sys.version_info[0:2] < (3, 7)


cdef object get_future_loop(object future):
    if _LESS_PY37:
        return future._loop
    else:
        return future.get_loop()


cpdef bytes new_actor_id():
    return np.random.bytes(32)


async def sleep_and_await(func, *args, **kwargs):
    delay = kwargs.pop('_delay', 0)
    await asyncio.sleep(delay)
    return await func(*args, **kwargs)


async def await_sequence(*awaitables):
    cdef object val = None
    for a in awaitables:
        val = await a
    return val


def done_future():
    f = asyncio.Future()
    f.set_result(None)
    return f


def create_actor_ref(*args, **kwargs):
    from ..core import ActorRef

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
