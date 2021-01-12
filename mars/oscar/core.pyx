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
    def __init__(self, str address, object uid):
        self.uid = uid
        self.address = address
        self._methods = dict()

    def _set_ctx(self, ctx):
        self._ctx = ctx

    ctx = property(lambda self: self._ctx, _set_ctx)

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

        try:
            return self._methods[item]
        except KeyError:
            method = self._methods[item] = ActorRefMethod(item)
            return method


cdef class ActorRefMethod:
    cdef ActorRef ref
    cdef object method_name

    def __init__(self, ref, method_name):
        self.ref = ref
        self.method_name = method_name

    def send(self, *args, **kwargs):
        return self.ref.send((self.method_name,) + args + (kwargs,))

    def tell(self, *args, **kwargs):
        return self.ref.tell((self.method_name,) + args + (kwargs,))

    def async_send(self, *args, **kwargs):
        return self.ref.send((self.method_name,) + args + (kwargs,), wait=False)

    def async_tell(self, *args, **kwargs):
        return self.ref.tell((self.method_name,) + args + (kwargs,), wait=False)

    def delay_tell(self, *args, delay=None, **kwargs):
        return self.ref.tell((self.method_name,) + args + (kwargs,), delay=delay, wait=False)


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

    cpdef __post_create__(self):
        pass

    cpdef __on_receive__(self, tuple action):
        raise NotImplementedError

    cpdef __pre_destroy__(self):
        pass


cdef class ActorEnvironment:
    pass


cdef class ActorFuture:
    def __and__(self, other):
        if not isinstance(other, list):
            other = [other]
        return ActorFutures([self] + other)


class ActorFutures(list):
    def __and__(self, other):
        if not isinstance(other, list):
            other = [other]
        return ActorFutures(self + other)
