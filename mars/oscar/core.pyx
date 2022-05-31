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
import weakref
from typing import AsyncGenerator

cimport cython

from .context cimport get_context
from .errors import Return, ActorNotExist
from .utils cimport is_async_generator


CALL_METHOD_DEFAULT = 0
CALL_METHOD_BATCH = 1

logger = logging.getLogger(__name__)

cdef:
    bint _log_unhandled_errors = False
    bint _log_cycle_send = False
    dict _local_pool_map = dict()
    object _actor_method_wrapper


def set_debug_options(options):
    global _log_unhandled_errors, _log_cycle_send
    if options is None:
        _log_unhandled_errors = _log_cycle_send = False
    else:
        _log_unhandled_errors = options.log_unhandled_errors
        _log_cycle_send = options.log_cycle_send


cdef _get_local_actor(address, uid):
    # Do not expose this method to Python to avoid actor being
    # referenced everywhere.
    #
    # The cycle send detection relies on send message, so we
    # disabled the local actor proxy if the debug option is on.
    if _log_cycle_send:
        return None
    pool_ref = _local_pool_map.get(address)
    pool = None if pool_ref is None else pool_ref()
    if pool is not None:
        actor = pool._actors.get(uid)
        if actor is not None:
            return actor
    return None


def register_local_pool(address, pool):
    """
    Register local actor pool for local actor lookup.
    """
    _local_pool_map[address] = weakref.ref(
        pool, lambda _: _local_pool_map.pop(address, None)
    )


cpdef create_local_actor_ref(address, uid):
    """
    Create a reference to local actor.

    Returns
    -------
    LocalActorRef or None
    """
    actor = _get_local_actor(address, uid)
    if actor is not None:
        return LocalActorRef(actor)
    return None


cpdef create_actor_ref(address, uid):
    """
    Create an actor reference.
    TODO(fyrestone): Remove the create_actor_ref in utils.pyx

    Returns
    -------
    ActorRef or LocalActorRef
    """
    actor = _get_local_actor(address, uid)
    return ActorRef(address, uid) if actor is None else LocalActorRef(actor)


cdef class ActorRef:
    """
    Reference of an Actor at user side
    """
    def __init__(self, str address, object uid):
        if isinstance(uid, str):
            uid = uid.encode()
        self.uid = uid
        self.address = address
        self._methods = dict()

    def destroy(self, object callback=None):
        ctx = get_context()
        return ctx.destroy_actor(self)

    def __reduce__(self):
        return create_actor_ref, (self.address, self.uid)

    def __getattr__(self, item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        try:
            return self._methods[item]
        except KeyError:
            method = self._methods[item] = ActorRefMethod(self, item)
            return method

    def __hash__(self):
        return hash((self.address, self.uid))

    def __eq__(self, other):
        other_type = type(other)
        if other_type is ActorRef or other_type is LocalActorRef:
            return self.address == other.address and self.uid == other.uid
        return False

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
    cdef object _options

    def __init__(self, ref, method_name, options=None):
        self.ref = ref
        self.method_name = method_name
        self._options = options or {}

    def __call__(self, *args, **kwargs):
        return self.send(*args, **kwargs)

    def options(self, **options):
        return ActorRefMethod(self.ref, self.method_name, options)

    def send(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        return get_context().send(self.ref, arg_tuple, **self._options)

    def tell(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        return get_context().send(self.ref, arg_tuple, wait_response=False, **self._options)

    def delay(self, *args, **kwargs):
        arg_tuple = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
        return _DelayedArgument(arg_tuple)

    def batch(self, *delays, send=True):
        cdef:
            long n_delays = len(delays)
            bint has_kw = False
            list args_list
            list kwargs_list
            _DelayedArgument delay

        args_list = [None] * n_delays
        kwargs_list = [None] * n_delays

        last_method = None
        for idx in range(n_delays):
            delay = delays[idx]
            method, _call_method, args, kwargs = delay.arguments
            if last_method is not None and method != last_method:
                raise ValueError('Does not support calling multiple methods in batch')
            last_method = method

            args_list[idx] = args
            kwargs_list[idx] = kwargs
            if kwargs:
                has_kw = True

        if not has_kw:
            kwargs_list = None
        if last_method is None:
            last_method = self.method_name

        message = (last_method, CALL_METHOD_BATCH, (args_list, kwargs_list), None)
        return get_context().send(self.ref, message, wait_response=send, **self._options)

    def tell_delay(self, *args, delay=None, ignore_conn_fail=True, **kwargs):
        async def delay_fun():
            try:
                await asyncio.sleep(delay)
                message = (self.method_name, CALL_METHOD_DEFAULT, args, kwargs)
                await get_context().send(self.ref, message, wait_response=False, **self._options)
            except Exception as ex:
                if ignore_conn_fail and isinstance(ex, ConnectionRefusedError):
                    return

                logger.error(f'Error {type(ex)} occurred when calling {self.method_name} '
                             f'on {self.ref.uid} at {self.ref.address} with tell_delay')
                raise

        return asyncio.create_task(delay_fun())


cdef class LocalActorRef(ActorRef):
    def __init__(self, _BaseActor actor):
        # Make sure the input actor is an instance of _BaseActor.
        super().__init__(actor._address, actor._uid)
        self._actor_weakref = weakref.ref(actor, lambda _: self._methods.clear())

    cdef _weakref_local_actor(self):
        actor = _get_local_actor(self.address, self.uid)
        # Make sure the input actor is an instance of _BaseActor.
        if actor is not None and isinstance(actor, _BaseActor):
            self._actor_weakref = weakref.ref(actor, lambda _: self._methods.clear())
            return actor
        return None

    def __getattr__(self, item):
        try:
            return self._methods[item]
        except KeyError:
            actor = self._actor_weakref() or self._weakref_local_actor()
            if actor is None:
                raise ActorNotExist(f"Actor {self.uid} does not exist") from None
            # For detecting the attribute error.
            getattr(actor, item)
            method = self._methods[item] = LocalActorRefMethod(self, item)
            return method

    def __repr__(self):
        return 'LocalActorRef(uid={!r}, address={!r}), actor_weakref={!r}'.format(
            self.uid, self.address, self._actor_weakref)


async def __pyx_actor_method_wrapper(method, result_handler, lock, args, kwargs):
    async with lock:
        result = method(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
    return await result_handler(result)

# Avoid global lookup.
_actor_method_wrapper = __pyx_actor_method_wrapper


cdef class LocalActorRefMethod:
    cdef LocalActorRef _local_actor_ref
    cdef object _method_name

    def __init__(self, LocalActorRef local_actor_ref, method_name):
        self._local_actor_ref = local_actor_ref
        self._method_name = method_name

    cdef tuple _get_referent(self):
        actor = self._local_actor_ref._actor_weakref() or self._local_actor_ref._weakref_local_actor()
        if actor is None:
            raise ActorNotExist(f"Actor {self._local_actor_ref.uid} does not exist.")
        method = getattr(actor, self._method_name)
        return actor, method

    def __call__(self, *args, **kwargs):
        actor, method = self._get_referent()
        return _actor_method_wrapper(
            method, actor._handle_actor_result, (<_BaseActor>actor)._lock, args, kwargs)

    def options(self, **options):
        return self

    def send(self, *args, **kwargs):
        actor, method = self._get_referent()
        return _actor_method_wrapper(
            method, actor._handle_actor_result, (<_BaseActor>actor)._lock, args, kwargs)

    def tell(self, *args, **kwargs):
        actor, method = self._get_referent()
        coro = _actor_method_wrapper(
            method, actor._handle_actor_result, (<_BaseActor>actor)._lock, args, kwargs)
        asyncio.create_task(coro)
        return asyncio.sleep(0)

    def delay(self, *args, **kwargs):
        actor, method = self._get_referent()
        return method.delay(*args, **kwargs)

    def batch(self, *delays, send=True):
        actor, method = self._get_referent()
        coro = _actor_method_wrapper(
            method.batch, actor._handle_actor_result, (<_BaseActor>actor)._lock, delays, dict())
        if send:
            return coro
        else:
            asyncio.create_task(coro)
            return asyncio.sleep(0)

    def tell_delay(self, *args, delay=None, ignore_conn_fail=True, **kwargs):
        async def delay_fun():
            await asyncio.sleep(delay)
            await self.tell(*args, **kwargs)

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
        cdef int idx
        cdef tuple res_tuple
        cdef list tasks, coros, coro_poses, values
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
            coros = []
            coro_poses = []
            values = []
            for idx, res_item in enumerate(res_tuple):
                if is_async_generator(res_item):
                    value = self._run_actor_async_generator(res_item)
                    coros.append(value)
                    coro_poses.append(idx)
                elif inspect.isawaitable(res_item):
                    value = res_item
                    coros.append(value)
                    coro_poses.append(idx)
                else:
                    value = res_item
                values.append(value)

            # when there is only one coroutine, we do not need to use
            # asyncio.wait as it introduces much overhead
            if len(coros) == 1:
                task_result = await coros[0]
                if extract_tuple:
                    result = task_result
                else:
                    result = tuple(task_result if t is coros[0] else t for t in values)
            elif len(coros) > 0:
                tasks = [asyncio.create_task(t) for t in coros]
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
                    for pos in coro_poses:
                        task = tasks[pos]
                        values[pos] = task.result()
                    result = tuple(values)

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
                async with self._lock:
                    with debug_async_timeout('actor_lock_timeout',
                                             'async_generator %r hold lock timeout', gen):
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
                async with self._lock:
                    with debug_async_timeout('actor_lock_timeout',
                                             "Method %s of actor %s hold lock timeout.",
                                             method, self.uid):
                        result = func(*args, **kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
            elif call_method == CALL_METHOD_BATCH:
                func = getattr(self, method)
                async with self._lock:
                    with debug_async_timeout('actor_lock_timeout',
                                             "Batch method %s of actor %s hold lock timeout, batch size %s.",
                                             method, self.uid, len(args)):
                        args_list, kwargs_list = args
                        if kwargs_list is None:
                            kwargs_list = [{}] * len(args_list)
                        result = func.call_with_lists(args_list, kwargs_list)
                        if asyncio.iscoroutine(result):
                            result = await result
            else:  # pragma: no cover
                raise ValueError(f'call_method {call_method} not valid')

            return await self._handle_actor_result(result)
        except Exception as ex:
            if _log_unhandled_errors:
                from .debug import logger as debug_logger
                # use `%.500` to avoid print too long messages
                debug_logger.exception('Got unhandled error when handling message %.500r '
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
