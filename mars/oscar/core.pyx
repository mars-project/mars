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
cimport cython
import sys
import weakref

from .utils cimport is_async_generator
from .utils import create_actor_ref


CALL_METHOD_DEFAULT = 0
CALL_METHOD_BATCH = 1


_coro_to_arg = weakref.WeakKeyDictionary()


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

    def batch(self, *calls):
        cdef list args_list = []
        cdef list kwargs_list = []

        last_call_type = last_method = None
        for ref_call in calls:
            call_type, (method, _call_method, args, kwargs) = _coro_to_arg.pop(ref_call)
            if (last_call_type is not None and call_type != last_call_type) \
                    or (last_method is not None and method != last_method):
                raise ValueError('Does not support calling multiple methods in batch')
            last_call_type = call_type
            last_method = method

            args_list.append(args)
            kwargs_list.append(kwargs)

        if last_call_type == 'send':
            return self.__send__((last_method, CALL_METHOD_BATCH, (args_list, kwargs_list), {}))
        else:
            return self.__tell__((last_method, CALL_METHOD_BATCH, (args_list, kwargs_list), {}))

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

    def __hash__(self):
        return hash((type(self), self.address, self.uid))

    def __eq__(self, other):
        return isinstance(other, ActorRef) and \
               other.address == self.address and \
               other.uid == self.uid


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
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        coro = self.ref.__send__(arg_tuple)
        _coro_to_arg[coro] = ('send', arg_tuple)
        return coro

    def tell(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        coro = self.ref.__tell__(arg_tuple)
        _coro_to_arg[coro] = ('tell', arg_tuple)
        return coro

    def tell_delay(self, *args, delay=None, **kwargs):
        async def delay_fun():
            await asyncio.sleep(delay)
            await self.ref.__tell__((self.method_name, CALL_METHOD_DEFAULT, args, kwargs))

        return asyncio.create_task(delay_fun())


# The @cython.binding(True) is for ray getting members.
# The value is True by default after cython >= 3.0.0
@cython.binding(True)
cdef class _Actor:
    """
    Base Mars actor class, user methods implemented as methods
    """
    def __cinit__(self, *args, **kwargs):
        self._lock = asyncio.locks.Lock()

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid
        
    def _set_uid(self, uid):
        self._uid = uid

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, addr):
        self._address = addr

    def _set_address(self, addr):
        self._address = addr

    cpdef ActorRef ref(self):
        return create_actor_ref(self._address, self._uid)

    async def _handle_actor_result(self, result):
        cdef int result_pos
        cdef tuple res_tuple
        cdef list tasks, values
        cdef object coro
        cdef bint extract_tuple = False
        cdef set dones, pending

        if asyncio.iscoroutine(result):
            result = await result
        elif is_async_generator(result):
            result = (result,)
            extract_tuple = True

        if type(result) is tuple:
            res_tuple = result
            tasks = []
            values = []
            for res_item in res_tuple:
                if is_async_generator(res_item):
                    value = asyncio.create_task(self._run_actor_async_generator(res_item))
                    tasks.append(value)
                elif asyncio.iscoroutine(res_item):
                    value = asyncio.create_task(res_item)
                    tasks.append(value)
                else:
                    value = res_item
                values.append(value)

            if len(tasks) > 0:
                try:
                    dones, pending = await asyncio.wait(tasks)
                except asyncio.CancelledError:
                    for task in tasks:
                        task.cancel()
                    # wait till all tasks return cancelled
                    dones, pending = await asyncio.wait(tasks)

                if extract_tuple:
                    result = list(dones)[0].result()
                else:
                    result = tuple(t.result() if t in dones else t for t in values)

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

                try:
                    res = await self._handle_actor_result(res)
                except:
                    async with self._lock:
                        res = await gen.athrow(*sys.exc_info())
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
        method, call_method, args, kwargs = message
        if call_method == CALL_METHOD_DEFAULT:
            func = getattr(self, method)
        elif call_method == CALL_METHOD_BATCH:
            func = getattr(self, method).batch
        else:  # pragma: no cover
            raise ValueError(f'call_method {call_method} not valid')

        async with self._lock:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        return await self._handle_actor_result(result)
