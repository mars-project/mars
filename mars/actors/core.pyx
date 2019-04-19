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

import multiprocessing

from .cluster cimport ClusterInfo


__all__ = ['ActorRef', 'Actor', 'FunctionActor', 'create_actor_pool', 'new_client',
           'register_actor_implementation', 'unregister_actor_implementation']


cdef class ActorRef:
    def __init__(self, str address, object uid):
        self.uid = uid
        self.address = address

    def _set_ctx(self, ctx):
        self._ctx = ctx

    ctx = property(None, _set_ctx)

    cpdef object send(self, object message, bint wait=True, object callback=None):
        return self._ctx.send(self, message, wait=wait, callback=callback)

    cpdef object tell(self, object message, object delay=None, bint wait=True,
                      object callback=None):
        return self._ctx.tell(self, message, delay=delay, wait=wait, callback=callback)

    cpdef object destroy(self, bint wait=True, object callback=None):
        return self._ctx.destroy_actor(self, wait=wait, callback=callback)

    def __getstate__(self):
        return self.address, self.uid

    def __setstate__(self, state):
        self.address, self.uid = state

    def __reduce__(self):
        return self.__class__, self.__getstate__()

    def __getattr__(self, str item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        def _mt_call(*args, **kwargs):
            wait = kwargs.pop('_wait', True)
            if kwargs.pop('_tell', False):
                delay = kwargs.pop('_delay', None)
                return self.tell((item,) + args + (kwargs,), delay=delay, wait=wait)
            else:
                return self.send((item,) + args + (kwargs,), wait=wait)

        return _mt_call


cdef class Actor:
    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, addr):
        self._address = addr

    cpdef ActorRef ref(self):
        return self._ctx.actor_ref(self._address, self._uid)

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, ctx):
        self._ctx = ctx

    cpdef post_create(self):
        pass

    cpdef on_receive(self, message):
        raise NotImplementedError()

    cpdef pre_destroy(self):
        pass


cdef dict _actor_implementation = dict()


cdef class _FunctionActor(Actor):
    cpdef on_receive(self, message):
        method, args, kwargs = message[0], message[1:-1], message[-1]
        return getattr(self, method)(*args, **kwargs)


class FunctionActor(_FunctionActor):
    def __new__(cls, *args, **kwargs):
        try:
            return _actor_implementation[id(cls)](*args, **kwargs)
        except KeyError:
            return super(FunctionActor, cls).__new__(cls, *args, **kwargs)



cpdef object create_actor_pool(str address=None, int n_process=0, object distributor=None,
                               object parallel=None, str backend='gevent'):
    cdef bint standalone
    cdef ClusterInfo cluster_info

    if backend != 'gevent':
        raise ValueError('Only gevent-based actor pool is supported for now')

    from .pool.gevent_pool import ActorPool

    standalone = address is None
    if n_process <= 0:
        n_process = multiprocessing.cpu_count()

    cluster_info = ClusterInfo(standalone, n_process, address=address)
    pool = ActorPool(cluster_info, distributor=distributor, parallel=parallel)
    pool.run()

    return pool


cpdef object new_client(object parallel=None, str backend='gevent'):
    if backend != 'gevent':
        raise ValueError('Only gevent-based actor pool is supported for now')

    from .pool.gevent_pool import ActorClient

    return ActorClient(parallel=parallel)


def register_actor_implementation(actor_cls, impl_cls):
    _actor_implementation[id(actor_cls)] = impl_cls


def unregister_actor_implementation(actor_cls):
    try:
        del _actor_implementation[id(actor_cls)]
    except KeyError:
        pass

