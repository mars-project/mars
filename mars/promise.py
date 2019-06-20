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

import logging
import struct
import sys
import threading
import weakref

import numpy as np

from .compat import getargspec
from .actors import FunctionActor
from .actors.core import ActorRef
from .errors import PromiseTimeout
from .utils import wraps, build_exc_info

logger = logging.getLogger(__name__)
_promise_pool = dict()


class Promise(object):
    """
    Object representing a promised result
    """
    def __init__(self, resolve=None, reject=None, done=False, failed=False,
                 args=None, kwargs=None):
        # use random promise id
        self._id = struct.pack('<Q', id(self)) + np.random.bytes(32)

        self._accept_handler = self._wrap_handler(resolve)
        self._reject_handler = self._wrap_handler(reject)

        # _bind_item indicates the Promise object whose step_next()
        # should be called when current Promise finishes. For instance,
        # in ``p2 = p1.then(lambda arg: promise_call())``, the value of
        # _bind_item of the Promise object returned by ``promise_call()``
        # is p2, thus when the lambda finishes, subsequent operations in
        # p2 can be executed.
        self._bind_item = None  # type: Promise
        # _next_item indicates the next Promise object to be invoked
        # when the current one returns.
        self._next_item = None  # type: Promise

        # promise results
        if done:
            self._accepted = True
        elif failed:
            self._accepted = False
        else:
            self._accepted = None
            # register in global pool to reject gc collection when calls not finished
            _promise_pool[self._id] = self

        self._args = args or ()
        self._kwargs = kwargs or {}

        self.post_create()

    def __del__(self):
        self.pre_destroy()
        self._clear_result_cache()

    @property
    def id(self):
        return self._id

    def post_create(self):
        pass

    def pre_destroy(self):
        pass

    def _wrap_handler(self, func):
        """
        Wraps a promise handler
        """
        if func is None:
            return None

        @wraps(func)
        def _wrapped(*args, **kwargs):
            _promise_pool.pop(self.id, None)

            result = None
            try:
                result = func(*args, **kwargs)
                if isinstance(result, Promise):
                    # the function itself returns a promise object
                    # bind returned promise object to current promise
                    assert result._next_item is None
                    result._bind_item = self
                    if result._accepted is not None:
                        # promise already done, we move next
                        args = result._args or ()
                        kwargs = result._kwargs or {}
                        result._clear_result_cache()

                        kwargs['_accept'] = result._accepted
                        return result._internal_step_next, args, kwargs
                else:
                    # return non-promise result, we just step next
                    return self._internal_step_next, (result,), {}
            except:  # noqa: E722
                # error occurred when executing func, we reject with exc_info
                logger.exception('Exception met in executing promise.')
                exc = sys.exc_info()
                self._clear_result_cache()
                return self._internal_step_next, exc, dict(_accept=False)
            finally:
                del result
        return _wrapped

    def _get_bind_root(self):
        """
        Get root promise of result promises
        :return: root promise
        :rtype: Promise
        """
        target = self
        while target._bind_item is not None:
            if target.id in _promise_pool:
                # remove binder that will not be used later
                _promise_pool.pop(target.id, None)
            target._clear_result_cache()
            target = target._bind_item
        return target

    @staticmethod
    def _get_handling_promise(p, handler_attr):
        """
        Get a promise object that defines the handler needed
        :param Promise p: promise object
        :param str handler_attr: attribute name of the handler
        :rtype: Promise
        """
        while getattr(p, handler_attr) is None:
            p._accept_handler = p._reject_handler = None

            p = p._get_bind_root()  # type: Promise
            if p and p._next_item is not None:
                _promise_pool.pop(p.id, None)
                p._clear_result_cache()
                p = p._next_item
            else:
                break
        return p

    @staticmethod
    def _log_unexpected_error(args):
        if args and len(args) == 3 and issubclass(args[0], Exception):
            logger.exception('Unhandled exception in promise', exc_info=args)

    def _clear_result_cache(self):
        self._args = ()
        self._kwargs = {}

    def _internal_step_next(self, *args, **kwargs):
        """
        Actual call to step promise call into the next step.
        If there are further steps, the function will return it
        as a (func, args, kwargs) tuple to reduce memory footprint.
        """
        accept = kwargs.pop('_accept', True)
        target_promise = self

        self._accepted = accept

        try:
            root_promise = self._get_bind_root()

            if root_promise:
                _promise_pool.pop(root_promise.id, None)

            target_promise = root_promise._next_item
            root_promise._accepted = accept
            if not target_promise:
                root_promise._args = args
                root_promise._kwargs = kwargs
                if not accept:
                    self._log_unexpected_error(args)
                return
            else:
                root_promise._clear_result_cache()

            next_call = None
            if accept:
                acceptor = self._get_handling_promise(target_promise, '_accept_handler')
                if acceptor and acceptor._accept_handler:
                    # remove the handler in promise in case that
                    # function closure is not freed
                    handler, acceptor._accept_handler = acceptor._accept_handler, None
                    next_call = handler(*args, **kwargs)
                else:
                    acceptor._accepted = accept
                    acceptor._args = args
                    acceptor._kwargs = kwargs
                    _promise_pool.pop(acceptor.id, None)
            else:
                rejecter = self._get_handling_promise(target_promise, '_reject_handler')
                if rejecter and rejecter._reject_handler:
                    # remove the handler in promise in case that
                    # function closure is not freed
                    handler, rejecter._reject_handler = rejecter._reject_handler, None
                    next_call = handler(*args, **kwargs)
                else:
                    rejecter._accepted = accept
                    rejecter._args = args
                    rejecter._kwargs = kwargs
                    _promise_pool.pop(rejecter.id, None)
                    self._log_unexpected_error(args)
            return next_call
        finally:
            del args, kwargs
            if target_promise:
                _promise_pool.pop(target_promise.id, None)

    def step_next(self, args_and_kwargs=None):
        """
        Step into next promise with given args and kwargs
        """
        if args_and_kwargs is None:
            args_and_kwargs = [(), {}]
        step_call = (self._internal_step_next,) + tuple(args_and_kwargs)
        del args_and_kwargs[:]

        while step_call is not None:
            func, args, kwargs = step_call
            step_call = func(*args, **kwargs)

    def then(self, on_fulfilled, on_rejected=None):
        promise = Promise(on_fulfilled, on_rejected)
        assert self._bind_item is None
        self._next_item = promise
        if self._accepted is not None:
            args_and_kwargs = [self._args, self._kwargs]
            args_and_kwargs[1]['_accept'] = self._accepted
            self._clear_result_cache()
            self.step_next(args_and_kwargs)
        return promise

    def catch(self, on_rejected):
        return self.then(None, on_rejected)

    def wait(self, waiter=None, timeout=None):
        """
        Wait when the promise returns. Currently only used in debug.
        :param waiter: wait object
        :param timeout: wait timeout
        :return: accept or reject
        """
        waiter = threading.Event()
        status = []

        def _finish_exec(accept_or_reject):
            waiter.set()
            status.append(accept_or_reject)

        self.then(lambda *_, **__: _finish_exec(True),
                  lambda *_, **__: _finish_exec(False))
        waiter.wait(timeout)
        return status[0]


class PromiseRefWrapper(object):
    """
    Promise wrapper that enables promise call by adding _promise=True
    """
    def __init__(self, ref, caller):
        self._ref = ref
        self._caller = caller  # type: PromiseActor

    def send(self, message):
        return self._ref.send(message)

    def tell(self, message, delay=None):
        return self._ref.tell(message, delay=delay)

    def destroy(self):
        return self._ref.destroy_actor(self)

    @property
    def uid(self):
        return self._ref.uid

    @property
    def address(self):
        return self._ref.address

    def __getattr__(self, item):
        if item.startswith('_'):
            return object.__getattribute__(self, item)

        def _mt_fun(*args, **kwargs):
            ref_fun = getattr(self._ref, item)
            if not kwargs.pop('_promise', False):
                return ref_fun(*args, **kwargs)

            p = Promise()
            self._caller.register_promise(p, self._ref)

            timeout = kwargs.pop('_timeout', 0)

            kwargs['callback'] = ((self._caller.uid, self._caller.address),
                                  'handle_promise', p.id)
            kwargs['_tell'] = True
            ref_fun(*args, **kwargs)

            if timeout and timeout > 0:
                # add a callback that triggers some times later to deal with timeout
                self._caller.ref().handle_promise_timeout(p.id, _tell=True, _delay=timeout)

            return p

        return _mt_fun


def reject_on_exception(func):
    """
    Decorator on actor callback functions that handles exceptions by
    sending it to caller as promise rejections. The function should have
    an argument called ``callback``.
    """
    arg_names = getargspec(func).args
    callback_pos = None
    if arg_names:
        for idx, name in enumerate(arg_names):
            if name == 'callback':
                callback_pos = idx
                break

    @wraps(func)
    def _wrapped(*args, **kwargs):
        callback = None
        if 'callback' in kwargs:
            callback = kwargs['callback']
        elif callback_pos and callback_pos < len(args):
            callback = args[callback_pos]

        try:
            return func(*args, **kwargs)
        except:  # noqa: E722
            actor = args[0]
            logger.exception('Unhandled exception in promise call')
            if callback:
                actor.tell_promise(callback, *sys.exc_info(), **dict(_accept=False))
            else:
                raise
    return _wrapped


class PromiseActor(FunctionActor):
    """
    Actor class providing promise functionality
    """
    def _prepare_promise_registration(self):
        if not hasattr(self, '_promises'):
            self._promises = dict()
            self._promise_ref_keys = dict()
            self._ref_key_promises = dict()

    def promise_ref(self, *args, **kwargs):
        """
        Wraps an existing ActorRef into a promise ref
        """
        self._prepare_promise_registration()

        if not args and not kwargs:
            ref = self.ref()
        elif args and isinstance(args[0], ActorRef):
            ref = self.ctx.actor_ref(args[0].uid, address=args[0].address)
        else:
            ref = self.ctx.actor_ref(*args, **kwargs)
        return PromiseRefWrapper(ref, self)

    def spawn_promised(self, func, *args, **kwargs):
        """
        Run func asynchronously in a pool and returns a promise.
        The running of the function does not block current actor.

        :param func: function to run
        :return: promise
        """
        self._prepare_promise_registration()

        if not hasattr(self, '_async_group'):
            self._async_group = self.ctx.asyncpool()

        p = Promise()
        ref = self.ref()
        self.register_promise(p, ref)

        def _wrapped(*a, **kw):
            try:
                result = func(*a, **kw)
            except:  # noqa: E722
                ref.handle_promise(p.id, *sys.exc_info(), **dict(_accept=False, _tell=True))
            else:
                ref.handle_promise(p.id, result, _tell=True)
            finally:
                del a, kw

        try:
            self._async_group.spawn(_wrapped, *args, **kwargs)
        finally:
            del args, kwargs
        return p

    def register_promise(self, promise, ref):
        """
        Register a promise into the actor with referrer info

        :param Promise promise: promise object to register
        :param ActorRef ref: ref
        """
        promise_id = promise.id

        def _weak_callback(*_):
            self.delete_promise(promise_id)

        self._promises[promise_id] = weakref.ref(promise, _weak_callback)
        ref_key = (ref.uid, ref.address)
        self._promise_ref_keys[promise_id] = ref_key
        try:
            self._ref_key_promises[ref_key].add(promise_id)
        except KeyError:
            self._ref_key_promises[ref_key] = {promise_id}

    def get_promise(self, promise_id):
        """
        Get promise object from weakref.
        """
        obj = self._promises.get(promise_id)
        if obj is None:
            return None
        return obj()

    def delete_promise(self, promise_id):
        if promise_id not in self._promises:
            return
        ref_key = self._promise_ref_keys[promise_id]
        self._ref_key_promises[ref_key].remove(promise_id)
        del self._promises[promise_id]
        del self._promise_ref_keys[promise_id]

    def reject_dead_endpoints(self, dead_endpoints, *args, **kwargs):
        """
        Reject all promises related to given remote address
        :param dead_endpoints: list of dead workers
        """
        dead_refs = []
        for ref_key in self._ref_key_promises.keys():
            uid, addr = ref_key
            if addr in dead_endpoints:
                dead_refs.append(self.ctx.actor_ref(uid, address=addr))
        return self.reject_promise_refs(dead_refs, *args, **kwargs)

    def reject_promise_refs(self, refs, *args, **kwargs):
        """
        Reject all promises related to given actor ref
        :param refs: actor refs to reject
        """
        kwargs['_accept'] = False
        handled_refs = []
        for ref in refs:
            ref_key = (ref.uid, ref.address)
            if ref_key not in self._ref_key_promises:
                continue
            handled_refs.append(ref)
            for promise_id in list(self._ref_key_promises[ref_key]):
                p = self.get_promise(promise_id)
                if p is None:
                    continue
                p.step_next([args, kwargs])
        return handled_refs

    def tell_promise(self, callback, *args, **kwargs):
        """
        Tell promise results to the caller
        :param callback: promise callback
        """
        uid, address = callback[0]
        callback_args = callback[1:] + args + (kwargs, )
        self.ctx.actor_ref(uid, address=address).tell(callback_args)

    def handle_promise(self, promise_id, *args, **kwargs):
        """
        Callback entry for promise results
        :param promise_id: promise key
        """
        p = self.get_promise(promise_id)
        if p is None:
            logger.warning('Promise %r reentered or not registered in %s', promise_id, self.uid)
            return
        self.get_promise(promise_id).step_next([args, kwargs])
        self.delete_promise(promise_id)

    def handle_promise_timeout(self, promise_id):
        """
        Callback entry for promise timeout
        :param promise_id: promise key
        """
        p = self.get_promise(promise_id)
        if p is None or p._accepted is not None:
            # skip promises that are already finished
            return

        self.delete_promise(promise_id)
        p.step_next([build_exc_info(PromiseTimeout), dict(_accept=False)])


def all_(promises):
    """
    Create a promise with promises. Invoked when all referenced promises accepted
    or at least one rejected
    :param promises: collection of promises
    :return: the new promise
    """
    promises = [p for p in promises if isinstance(p, Promise)]
    new_promise = Promise()
    finish_set = set()

    def _build_then(promise):
        def _then(*_, **__):
            finish_set.add(promise.id)
            if all(p.id in finish_set for p in promises):
                new_promise.step_next()
        return _then

    def _handle_reject(*args, **kw):
        if new_promise._accepted is not None:
            return
        for p in promises:
            _promise_pool.pop(p.id, None)
        kw['_accept'] = False
        new_promise.step_next([args, kw])

    for p in promises:
        p.then(_build_then(p), _handle_reject)

    if promises:
        return new_promise
    else:
        new_promise.step_next()
        return new_promise


def get_active_promise_count():
    return len(_promise_pool)


def finished(*args, **kwargs):
    """
    Create a promise already finished
    :return: Promise object if _promise is True, otherwise args[0] will be returned
    """
    promise = kwargs.pop('_promise', True)
    accept = kwargs.pop('_accept', True)
    if not promise:
        return args[0] if args else None
    if accept:
        return Promise(done=True, args=args, kwargs=kwargs)
    else:
        return Promise(failed=True, args=args, kwargs=kwargs)
