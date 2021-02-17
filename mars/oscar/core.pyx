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

import asyncio

from .utils cimport is_async_generator


cdef class ActorRef:
    """
    Reference of an Actor at user side
    """
    def __init__(self, str address, object uid):
        self.uid = uid
        self.address = address
        self._methods = dict()

    cdef __send__(self, object message):
        from .context import get_context
        ctx = get_context()
        return ctx.send(self, message)

    cdef __tell__(self, object message, object delay=None):
        from .context import get_context
        ctx = get_context()
        return ctx.send(self, message, wait_response=False)

    def destroy(self, object callback=None):
        from .context import get_context
        ctx = get_context()
        return ctx.destroy_actor(self)

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
            method = self._methods[item] = ActorRefMethod(self, item)
            return method


cdef class ActorRefMethod:
    """
    Wrapper for an Actor method at client
    """
    cdef ActorRef ref
    cdef object method_name

    def __init__(self, ref, method_name):
        self.ref = ref
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        return self.send(*args, **kwargs)

    def send(self, *args, **kwargs):
        return self.ref.__send__((self.method_name,) + args + (kwargs,))

    def tell(self, *args, **kwargs):
        return self.ref.__tell__((self.method_name,) + args + (kwargs,))

    def tell_delay(self, *args, delay=None, **kwargs):
        async def delay_fun():
            await asyncio.sleep(delay)
            await self.ref.__tell__((self.method_name,) + args + (kwargs,))

        asyncio.create_task(delay_fun())


cdef class _Actor:
    """
    Base Mars actor class, user methods implemented as methods
    """
    def __init__(self):
        self._lock = asyncio.locks.Lock()

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
        from .context import get_context
        return get_context().actor_ref(self._address, self._uid)

    async def _handle_actor_result(self, result):
        cdef tuple res_tuple
        cdef list results

        if asyncio.iscoroutine(result):
            result = await result
        elif is_async_generator(result):
            result = (result,)

        if isinstance(result, tuple):
            res_tuple = result
            results = []
            for res_item in res_tuple:
                if is_async_generator(res_item):
                    results.append(self._run_actor_async_generator(res_item))
                elif asyncio.iscoroutine(result):
                    results.append(result)
                else:
                    results.append(res_item)

            await asyncio.wait([coro for coro in results if asyncio.iscoroutine(coro)])
            result = tuple(await res if asyncio.iscoroutine(res) else res
                           for res in results)
        return result

    async def _run_actor_async_generator(self, gen):
        """
        Run an async generator under Actor lock
        """
        cdef tuple res_tuple
        cdef object res

        try:
            res = None

            while True:
                async with self._lock:
                    res = await gen.asend(res)
                res = await self._handle_actor_result(res)
        except StopAsyncIteration:
            pass
        return res

    async def __post_create__(self):
        """
        Method called after actor creation
        """
        pass

    async def __pre_destroy__(self):
        """
        Method called before actor destroy
        """
        pass

    async def __on_receive__(self, tuple message):
        """
        Handle message from other actors and dispatch them to user methods

        Parameters
        ----------
        message : tuple
            Message shall be (method_name,) + args + (kwargs,)
        """
        method = message[0]
        args = message[1:-1]
        kwargs = message[-1]
        async with self._lock:
            result = getattr(self, method)(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        return await self._handle_actor_result(result)
