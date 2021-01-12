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

from threading import local as thread_local
from urllib.parse import urlparse

from gevent.local import local as gevent_local

from .core cimport ActorRef

cdef dict _backend_context_cls = dict()

cdef object _thread_context_local = thread_local()
cdef object _gevent_context_local = gevent_local()


cdef class BaseActorContext:
    def create_actor(self, object actor_cls, *args, object address=None, **kwargs):
        raise NotImplementedError

    cpdef int has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None) except -1:
        raise NotImplementedError

    cpdef destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        raise NotImplementedError

    def actor_ref(self, *args, **kwargs):
        raise NotImplementedError

    cpdef BaseActorContext copy(self):
        raise NotImplementedError


cdef class ClientActorContext(BaseActorContext):
    cdef dict _backend_contexts

    def __init__(self):
        self._backend_contexts = dict()

    cdef inline object _get_backend_context(self, object address):
        try:
            return self._backend_contexts[address]
        except KeyError:
            scheme = urlparse(address).scheme or 'mars'
            context = self._backend_contexts[address] = \
                _backend_context_cls[scheme]()
            return context

    def create_actor(self, object actor_cls, *args, object address=None, **kwargs):
        if address is None:
            raise ValueError('address must be provided')
        context = self._get_backend_context(address)
        return context.create_actor(actor_cls, *args, address=address, **kwargs)

    cpdef int has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None) except -1:
        context = self._get_backend_context(actor_ref.address)

    cpdef destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        context = self._get_backend_context(actor_ref.address)
        return context.destroy_actor(actor_ref, wait=wait, callback=callback)

    def actor_ref(self, *args, **kwargs):
        from .utils import create_actor_ref
        return create_actor_ref(*args, **kwargs)

    cpdef BaseActorContext copy(self):
        return ClientActorContext()


def register_backend_context(scheme, cls):
    assert issubclass(cls, BaseActorContext)
    _backend_context_cls[scheme] = cls


def get_context():
    try:
        return _gevent_context_local.context
    except AttributeError:
        pass

    try:
        return _thread_context_local.context
    except AttributeError:
        context = _thread_context_local.context = ClientActorContext()
        return context
