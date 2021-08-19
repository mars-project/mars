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

import asyncio
import inspect
import logging
import sys
cimport cython
from typing import AsyncGenerator

from .errors import Return
from .utils cimport is_async_generator
from .utils import create_actor_ref


CALL_METHOD_DEFAULT = 0
CALL_METHOD_BATCH = 1

logger = logging.getLogger(__name__)

cdef:
    bint _log_unhandled_errors = False
    bint _log_cycle_send = False


def set_debug_options(options):
    global _log_unhandled_errors, _log_cycle_send
    if options is None:
        _log_unhandled_errors = _log_cycle_send = False
    else:
        _log_unhandled_errors = options.log_unhandled_errors
        _log_cycle_send = options.log_cycle_send


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

    def __hash__(self):
        return hash((type(self), self.address, self.uid))

    def __eq__(self, other):
        return isinstance(other, ActorRef) and \
               other.address == self.address and \
               other.uid == self.uid

    def __repr__(self):
        return 'ActorRef(uid={!r}, address={!r})'.format(self.uid, self.address)


cdef class _DelayedArgument:
    cdef readonly tuple arguments

    def __init__(self, tuple arguments):
        self.arguments = arguments


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
        return self.ref.__send__(arg_tuple)

    def tell(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        return self.ref.__tell__(arg_tuple)

    def delay(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        return _DelayedArgument(arg_tuple)

    def batch(self, *delays, send=True):
        cdef:
            list args_list = list()
            list kwargs_list = list()

        last_method = None
        for delay in delays:
            method, _call_method, args, kwargs = delay.arguments
            if last_method is not None and method != last_method:
                raise ValueError('Does not support calling multiple methods in batch')
            last_method = method

            args_list.append(args)
            kwargs_list.append(kwargs)
        if last_method is None:
            last_method = self.method_name

        if send:
            return self.ref.__send__((last_method, CALL_METHOD_BATCH,
                                      (args_list, kwargs_list), {}))
        else:
            return self.ref.__tell__((last_method, CALL_METHOD_BATCH,
                                      (args_list, kwargs_list), {}))

    def tell_delay(self, *args, delay=None, ignore_conn_fail=True, **kwargs):
        async def delay_fun():
            try:
                await asyncio.sleep(delay)
                await self.ref.__tell__((self.method_name, CALL_METHOD_DEFAULT, args, kwargs))
            except Exception as ex:
                if ignore_conn_fail and isinstance(ex, ConnectionRefusedError):
                    return

                logger.error(f'Error {type(ex)} occurred when calling {self.method_name} '
                             f'on {self.ref.uid} at {self.ref.address} with tell_delay')
                raise

        return asyncio.create_task(delay_fun())


cdef class _BaseActor:
    """
    Base Mars actor class, user methods implemented as methods
    """
    def __cinit__(self, *args, **kwargs):
        self._lock = self._create_lock()

    def _create_lock(self):
        raise NotImplementedError

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
        cdef bint cancelled = False
        cdef set dones, pending

        if inspect.isawaitable(result):
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
                elif inspect.isawaitable(res_item):
                    value = asyncio.create_task(res_item)
                    tasks.append(value)
                else:
                    value = res_item
                values.append(value)

            if len(tasks) > 0:
                try:
                    dones, pending = await asyncio.wait(tasks)
                except asyncio.CancelledError:
                    cancelled = True
                    for task in tasks:
                        task.cancel()
                    # wait till all tasks return cancelled
                    dones, pending = await asyncio.wait(tasks)

                if extract_tuple:
                    result = list(dones)[0].result()
                else:
                    result = tuple(t.result() if t in dones else t for t in values)

                if cancelled:
                    # raise in case no CancelledError raised
                    raise asyncio.CancelledError

        return result

    async def _run_actor_async_generator(self, gen: AsyncGenerator):
        """
        Run an async generator under Actor lock
        """
        cdef tuple res_tuple
        cdef bint is_exception = False
        cdef object res
        cdef object message_trace = None, pop_message_trace = None, set_message_trace = None

        from .debug import pop_message_trace, set_message_trace, debug_async_timeout
        try:
            res = None
            while True:
                with debug_async_timeout('actor_lock_timeout',
                                         'async_generator %r hold lock timeout', gen):
                    async with self._lock:
                        if not is_exception:
                            res = await gen.asend(res)
                        else:
                            res = await gen.athrow(*res)
                try:
                    if _log_cycle_send:
                        message_trace = pop_message_trace()

                    res = await self._handle_actor_result(res)
                    is_exception = False
                except:
                    res = sys.exc_info()
                    is_exception = True
                finally:
                    if _log_cycle_send:
                        set_message_trace(message_trace)
        except Return as ex:
            return ex.value
        except StopAsyncIteration as ex:
            return

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
        from .debug import debug_async_timeout
        try:
            method, call_method, args, kwargs = message
            if call_method == CALL_METHOD_DEFAULT:
                func = getattr(self, method)
                with debug_async_timeout('actor_lock_timeout',
                                         "Method %s of actor %s hold lock timeout.",
                                         method, self.uid):
                    async with self._lock:
                        result = func(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
            elif call_method == CALL_METHOD_BATCH:
                func = getattr(self, method)
                with debug_async_timeout('actor_lock_timeout',
                                         "Batch method %s of actor %s hold lock timeout, batch size %s.",
                                         method, self.uid, len(args)):
                    async with self._lock:
                        delays = []
                        for s_args, s_kwargs in zip(*args):
                            delays.append(func.delay(*s_args, **s_kwargs))
                        result = func.batch(*delays)
                        if asyncio.iscoroutine(result):
                            result = await result
            else:  # pragma: no cover
                raise ValueError(f'call_method {call_method} not valid')

            return await self._handle_actor_result(result)
        except Exception as ex:
            if _log_unhandled_errors:
                from .debug import logger as debug_logger
                debug_logger.exception('Got unhandled error when handling message %r'
                                       'in actor %s at %s', message, self.uid, self.address)
            raise ex


# The @cython.binding(True) is for ray getting members.
# The value is True by default after cython >= 3.0.0
@cython.binding(True)
cdef class _Actor(_BaseActor):
    def _create_lock(self):
        return asyncio.locks.Lock()


cdef class _FakeLock:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# The @cython.binding(True) is for ray getting members.
# The value is True by default after cython >= 3.0.0
@cython.binding(True)
cdef class _StatelessActor(_BaseActor):
    def _create_lock(self):
        return _FakeLock()
