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

from .compat import getargspec, six
from .actors import FunctionActor
from .actors.core import ActorRef
from .errors import PromiseTimeout
from .utils import wraps

logger = logging.getLogger(__name__)
_promise_pool = dict()


class Promise(object):
    """
    Object representing a promised result
    """
    def __init__(self, resolve=None, reject=None, done=False):
        # use random promise id
        self._id = struct.pack('<Q', id(self)) + np.random.bytes(32)
        # register in global pool to reject gc collection
        _promise_pool[self._id] = self

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
        self._accepted = None if not done else True
        self._args = ()

        self.post_create()

    def __del__(self):
        self.pre_destroy()

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
            try:
                result = func(*args, **kwargs)
                if isinstance(result, Promise):
                    # the function itself returns a promise object
                    # bind returned promise object to current promise
                    result._bind_item = self
                    if result._accepted is not None:
                        # promise already done, we move next
                        result.step_next(_accept=result._accepted)
                else:
                    # return non-promise result, we just step next
                    self.step_next(result)
            except:
                # error occurred when executing func, we reject with exc_info
                logger.exception('Exception met in executing promise.')
                exc = sys.exc_info()
                self.step_next(*exc, _accept=False)
        return _wrapped

    def _get_bind_root(self):
        """
        Get root promise of result promises
        :return: root promise
        :rtype: Promise
        """
        target = self
        while target._bind_item is not None:
            if target and target.id in _promise_pool:
                # remove binder that will not be used later
                del _promise_pool[target.id]
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
            p = p._get_bind_root()
            if p and p._next_item is not None:
                if p.id in _promise_pool:
                    # remove predecessor that will not be used later
                    del _promise_pool[p.id]
                p = p._next_item
            else:
                break
        return p

    def step_next(self, *args, **kwargs):
        """
        Step into next promise with given args and kwargs
        """
        def _log_unexpected_error():
            if args and len(args) == 3 and issubclass(args[0], Exception):
                try:
                    six.reraise(*args)
                except:
                    logger.exception('Unhandled exception in promise')

        accept = kwargs.pop('_accept', True)
        target_promise = self

        self._accepted = accept
        if not self._accepted:
            # we only preserve exceptions to avoid tracing huge objects
            self._args = args

        try:
            target_promise = self._get_bind_root()

            if target_promise and target_promise.id in _promise_pool:
                del _promise_pool[target_promise.id]

            target_promise = target_promise._next_item
            if not target_promise:
                if not accept:
                    _log_unexpected_error()
                return

            if accept:
                acceptor = self._get_handling_promise(target_promise, '_accept_handler')
                if acceptor and acceptor._accept_handler:
                    acceptor._accept_handler(*args, **kwargs)
            else:
                rejecter = self._get_handling_promise(target_promise, '_reject_handler')
                if rejecter and rejecter._reject_handler:
                    rejecter._reject_handler(*args, **kwargs)
                else:
                    _log_unexpected_error()
        finally:
            if target_promise and target_promise.id in _promise_pool:
                del _promise_pool[target_promise.id]

    def then(self, on_fulfilled, on_rejected=None):
        promise = Promise(on_fulfilled, on_rejected)
        self._next_item = promise
        if self._accepted is not None:
            self.step_next(*self._args, **dict(_accept=self._accepted))
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
        return self._ref.send(self, message)

    def tell(self, message, delay=None):
        return self._ref.tell(self, message, delay=delay)

    def destroy(self):
        return self._ref.destroy_actor(self)

    @property
    def uid(self):
        return self._ref.uid

    @property
    def ctx(self):
        return self._ref.ctx

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
            self._caller._promises[p.id] = p

            timeout = kwargs.pop('_timeout', 0)

            kwargs['callback'] = ((self._caller.uid, self._caller.address),
                                  'handle_promise', p.id)
            kwargs['_tell'] = True
            ref_fun(*args, **kwargs)

            if timeout > 0:
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
        except:
            actor = args[0]
            logger.exception('Unhandled exception in promise call')
            if callback:
                actor.tell_promise(callback, *sys.exc_info(), **dict(_accept=False))
    return _wrapped


class PromiseActor(FunctionActor):
    """
    Actor class providing promise functionality
    """
    def promise_ref(self, *args, **kwargs):
        """
        Wraps an existing ActorRef into a promise ref
        """
        if not hasattr(self, '_promises'):
            self._promises = weakref.WeakValueDictionary()  # type: dict[object, Promise]

        if not args and not kwargs:
            ref = self.ref()
        elif args and isinstance(args[0], ActorRef):
            ref = self.ctx.actor_ref(args[0].uid, address=args[0].address)
        else:
            ref = self.ctx.actor_ref(*args, **kwargs)
        return PromiseRefWrapper(ref, self)

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
        try:
            self._promises[promise_id].step_next(*args, **kwargs)
            if promise_id in self._promises:
                del self._promises[promise_id]
        except KeyError:
            logger.warning('Promise %r reentered in %s', promise_id, self.uid)

    def handle_promise_timeout(self, promise_id):
        """
        Callback entry for promise timeout
        :param promise_id: promise key
        """
        p = self._promises.get(promise_id)
        if not p or p._accepted is not None:
            # skip promises that are already finished
            return

        try:
            del self._promises[promise_id]
        except KeyError:
            pass

        try:
            raise PromiseTimeout
        except PromiseTimeout:
            exc_info = sys.exc_info()
        p.step_next(*exc_info, **dict(_accept=False))


def all_(promises):
    """
    Create a promise with promises. Invoked when all referenced promises accepted
    or at least one rejected
    :param promises: collection of promises
    :return: the new promise
    """
    promises = list(promises)
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
        kw['_accept'] = False
        new_promise.step_next(*args, **kw)

    [p.then(_build_then(p), _handle_reject) for p in promises if isinstance(p, Promise)]
    if promises:
        return new_promise
    else:
        new_promise.step_next()
        return new_promise
