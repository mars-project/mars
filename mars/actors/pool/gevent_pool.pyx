#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import weakref
import sys
import pickle
import random
import errno
import socket
import struct
import itertools

import gevent
import gevent.queue
import gevent.lock
import gevent.event
import gevent.pool
import gevent.socket
import gevent.server
import gevent.subprocess
import gevent.fileobject
from gevent._tblib import _init as gevent_init_tblib
from gevent.threadpool import ThreadPool as GThreadPool

from ...lib import gipc
from ...compat import six, OrderedDict, TimeoutError, BrokenPipeError, ConnectionRefusedError
from ..errors import ActorPoolNotStarted, ActorNotExist, ActorAlreadyExist
from ..distributor cimport Distributor
from ..core cimport ActorRef, Actor
from ..cluster cimport ClusterInfo
from .messages cimport pack_send_message, pack_tell_message, pack_create_actor_message, \
    pack_destroy_actor_message, pack_result_message, unpack_send_message, unpack_tell_message, \
    unpack_create_actor_message, unpack_destroy_actor_message, unpack_result_message, \
    pack_has_actor_message, unpack_has_actor_message, pack_error_message, unpack_error_message, \
    unpack_message_type_value, unpack_message_type, unpack_message_id, read_remote_message, \
    get_index, MessageType
from .messages import write_remote_message
from .utils cimport new_actor_id
from .utils import create_actor_ref

gevent_init_tblib()

cdef int REMOTE_FROM_INDEX = -2
cdef int UNKNOWN_TO_INDEX = -1
cpdef int REMOTE_DEFAULT_PARALLEL = 50  # parallel connection at most
cpdef int REMOTE_MAX_CONNECTION = 200  # most connections

_inaction_encoder = _inaction_decoder = None


cdef class MessageContext:
    cdef public object message
    cdef public object async_result

    def __init__(self, message):
        self.message = message
        self.async_result = gevent.event.AsyncResult()

    cpdef result(self):
        return self.async_result.result()


cdef class ActorExecutionContext:
    cdef public object actor
    cdef public object lock
    cdef public object inbox
    cdef public object comm

    def __init__(self, actor, comm):
        self.actor = weakref.ref(actor)
        self.lock = gevent.lock.RLock()
        self.inbox = gevent.queue.Queue()
        self.comm = comm

    cpdef object fire_run(self):
        cdef MessageContext message_ctx

        with self.lock:
            message_ctx = self.inbox.get()
            actor = self.actor()

            try:
                res = actor.on_receive(message_ctx.message)
                message_ctx.async_result.set_result(res)
                return res
            except:
                t, ex, tb = sys.exc_info()
                message_ctx.async_result.set_exception(ex, exc_info=(t, ex, tb))
                raise

    cpdef void receive(self, MessageContext message_ctx):
        self.inbox.put(message_ctx)


cdef class ThreadPool:
    cdef object _pool

    def __init__(self, num_threads):
        self._pool = GThreadPool(num_threads)

    def submit(self, fn, *args, **kwargs):
        return self._pool.spawn(fn, *args, **kwargs)


cdef class ActorContext:
    """
    Hold by an actor.

    actor = Actor()
    actor.ctx = ActorContext(comm)

    provide methods:

    create_actor
    actor_ref
    """
    cdef object _comm

    def __init__(self, comm):
        self._comm = comm

    @property
    def index(self):
        return self._comm.index

    @property
    def distributor(self):
        return self._comm.distributor

    def create_actor(self, actor_cls, *args, **kwargs):
        cdef bint wait
        cdef object address
        cdef object uid
        cdef object ref

        wait = kwargs.pop('wait', True)
        address = kwargs.pop('address', None)
        uid = kwargs.pop('uid', None)

        if wait:
            ref = self._comm.create_actor(address, uid, actor_cls, *args, **kwargs)
            ref.ctx = self
            return ref

        def callback(ret):
            ret.ctx = self

        kwargs['wait'] = False
        kwargs['callback'] = callback
        return self._comm.create_actor(address, uid, actor_cls, *args, **kwargs)

    def destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        return self._comm.destroy_actor(actor_ref, wait=wait, callback=callback)

    def has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        return self._comm.has_actor(actor_ref, wait=wait, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = self
        return ref

    def send(self, ActorRef actor_ref, object message, bint wait=True, object callback=None):
        return self._comm.send(actor_ref, message, wait=wait, callback=callback)

    def tell(self, ActorRef actor_ref, object message, object delay=None,
             bint wait=True, object callback=None):
        return self._comm.tell(actor_ref, message, delay=delay,
                               wait=wait, callback=callback)

    @staticmethod
    def sleep(seconds):
        gevent.sleep(seconds)

    @staticmethod
    def fileobject(fobj, mode='rb', bufsize=-1, close=True):
        return gevent.fileobject.FileObject(fobj, mode=mode, bufsize=bufsize, close=close)

    @staticmethod
    def popen(*args, **kwargs):
        return gevent.subprocess.Popen(*args, **kwargs)

    @staticmethod
    def threadpool(size):
        return ThreadPool(size)

    @staticmethod
    def asyncpool(size=None):
        return gevent.pool.Pool(size)


cdef class LocalActorPool:
    """
    manage local actors, this pool is not aware of the multi-processes or cluster
    """

    cdef public dict actors
    cdef public object comm_proxy
    cdef public str address
    cdef public int index

    def __init__(self, str address=None, index=0):
        self.actors = dict()
        self.comm_proxy = None
        self.address = address
        self.index = index

    @property
    def comm(self):
        return self.comm_proxy()

    cpdef set_comm(self, object comm):
        self.comm_proxy = weakref.ref(comm)

    cpdef ActorExecutionContext get_actor_execution_ctx(self, object actor_uid):
        try:
            return self.actors[actor_uid][1]
        except KeyError:
            raise ActorNotExist('Actor {0} does not exist'.format(actor_uid))

    def create_actor(self, object actor_cls, object uid, *args, **kw):
        cdef Actor actor

        actor = actor_cls(*args, **kw)
        actor.address = self.address
        actor.uid = uid
        actor_execution_ctx = ActorExecutionContext(actor, self.comm)
        actor.ctx = ActorContext(actor_execution_ctx.comm)
        if uid in self.actors:
            raise ActorAlreadyExist('Actor {0} already exist, cannot create'.format(actor.uid))
        self.actors[uid] = (actor, actor_execution_ctx)
        actor.post_create()
        return ActorRef(self.address, uid)

    cpdef bint has_actor(self, object actor_uid) except -128:
        if actor_uid not in self.actors:
            return False
        return True

    cpdef destroy_actor(self, object actor_uid):
        cdef Actor actor

        try:
            actor, _ = self.actors[actor_uid]
        except KeyError:
            raise ActorNotExist('Actor {0} does not exist'.format(actor_uid))
        actor.pre_destroy()
        del self.actors[actor_uid]
        return actor_uid


cdef class AsyncHandler:
    cdef dict async_results

    def __init__(self):
        self.async_results = dict()

    cpdef object submit(self, bytes unique_id):
        cdef object ar

        ar = self.async_results[unique_id] = gevent.event.AsyncResult()

        # wait for result
        ar.wait()
        del self.async_results[unique_id]

        return ar.result()

    cpdef object future(self, bytes unique_id, object future, object callback=None):
        cdef object ar

        ar = self.async_results[unique_id] = gevent.event.AsyncResult()

        # wait for result
        ar.wait()
        del self.async_results[unique_id]

        if callback and ar.successful():
            callback(ar.result())

        future(ar)

    cpdef object wait(self, bytes unique_id):
        cdef object ar

        ar = self.async_results[unique_id] = gevent.event.AsyncResult()

        # wait for result
        ar.wait()
        del self.async_results[unique_id]

        if ar.successful():
            return ar.result()
        else:
            return ar.exception()

    cpdef void got(self, bytes unique_id, object result):
        # receive event handling
        self.async_results[unique_id].set_result(result)

    cpdef void err(self, bytes unique_id, object t, object ex, object tb):
        self.async_results[unique_id].set_exception(ex, exc_info=(t, ex, tb))


cdef class Connection:
    cdef public object conn
    cdef public object lock

    def __init__(self, conn, lock):
        self.conn = conn
        self.lock = lock

    def __enter__(self):
        return self.conn

    def __exit__(self, *_):
        self.lock.release()

    cpdef void release(self):
        self.lock.release()


class Connections(object):
    global_lock = gevent.lock.Semaphore()
    addrs = 0

    def __init__(self, address):
        if isinstance(address, six.string_types):
            self.address = address.split(':', 1)
        else:
            self.address = address

        with self.global_lock:
            Connections.addrs += 1

        self.lock = gevent.lock.Semaphore()
        self.conn_locks = OrderedDict()

    @property
    def conn(self):
        return [conn_lock[0] for conn_lock in six.itervalues(self.conn_locks)]

    def _connect(self, conn, lock):
        return Connection(conn, lock)

    def got_broken_pipe(self, fd):
        del self.conn_locks[fd]

    def connect(self):
        cdef int maxlen
        cdef object conn
        cdef object lock

        with self.lock:
            for conn, lock in six.itervalues(self.conn_locks):
                # try to reuse the connections before
                locked = lock.acquire(blocking=False)
                if not locked:
                    continue
                return self._connect(conn, lock)

            maxlen = max(REMOTE_MAX_CONNECTION // Connections.addrs, 1)

            if len(self.conn) < maxlen:
                # create a new connection
                lock = gevent.lock.Semaphore()
                lock.acquire()
                try:
                    conn = gevent.socket.create_connection(self.address)
                except socket.error as exc:  # pragma: no cover
                    if exc.errno == errno.ECONNREFUSED:
                        raise ConnectionRefusedError
                    elif exc.errno == errno.ETIMEDOUT:
                        raise TimeoutError
                    else:
                        raise

                self.conn_locks[conn.fileno()] = (conn, lock)
                return self._connect(conn, lock)

            def close(c, lk):
                with lk:
                    c.close()

            ps = [gevent.spawn(close, c, l) for c, l in
                  itertools.islice(six.itervalues(self.conn_locks), maxlen, len(self.conn_locks))]

            i = random.randint(0, maxlen - 1)
            fd = next(itertools.islice(six.iterkeys(self.conn_locks), i, i + 1))
            conn, lock = self.conn_locks[fd]
            lock.acquire()

            # wait for conn finished
            gevent.joinall(ps)
            self.conn_locks = OrderedDict(itertools.islice(six.iteritems(self.conn_locks), maxlen))

            return self._connect(conn, lock)

    def __del__(self):
        for c, _ in self.conn_locks.values():
            try:
                c.close()
            except:  # pragma: no cover
                pass


cdef class ActorRemoteHelper:
    """
    Used to handle remote operations, like deliver create_actor, destroy_actor, send etc to remote,
    and handle the response.

    An ActorRemoteHelper instance can be hold by Communicator and ActorClient.
    """
    index = REMOTE_FROM_INDEX  # means remote

    cdef object _parallel
    cdef object _pool
    cdef dict _connections
    cdef object _lock

    def __init__(self, parallel=None):
        self._parallel = parallel if parallel is not None else REMOTE_DEFAULT_PARALLEL
        self._pool = gevent.pool.Pool(self._parallel)
        self._connections = dict()
        self._lock = gevent.lock.RLock()

    cdef object _new_connection(self, str address):
        with self._lock:
            if address not in self._connections:
                connections = Connections(address)
                self._connections[address] = connections

            return self._connections[address].connect()

    cpdef object _send_remote(self, str address, object binary):
        cdef bytes res_binary
        cdef object message_type

        with self._new_connection(address) as sock:
            try:
                write_remote_message(sock.sendall, *binary)
                res_binary = read_remote_message(sock.recv)
                message_type = unpack_message_type(res_binary)
                if message_type == MessageType.error:
                    error_message = unpack_error_message(res_binary)
                    six.reraise(error_message.error_type, error_message.error, error_message.traceback)
                else:
                    assert message_type == MessageType.result
                    return unpack_result_message(res_binary).result
            except BrokenPipeError:
                self._connections[address].got_broken_pipe(sock.fileno())
                raise
            except socket.error as exc:  # pragma: no cover
                if exc.errno == errno.EPIPE:
                    self._connections[address].got_broken_pipe(sock.fileno())
                    raise BrokenPipeError
                else:
                    raise

    def create_actor(self, str address, object uid, object actor_cls, *args, **kwargs):
        cdef bint wait
        cdef object callback
        cdef tuple binaries
        cdef ActorRef actor_ref
        cdef object future

        wait = kwargs.pop('wait', True)
        callback = kwargs.pop('callback', None)

        try:
            binaries = pack_create_actor_message(
                ActorRemoteHelper.index, UNKNOWN_TO_INDEX,
                ActorRef(address, uid), actor_cls, args, kwargs)[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        if wait:
            actor_ref = self._pool.apply(self._send_remote, (address, binaries))
            actor_ref.ctx = ActorContext(self)
            return actor_ref

        # return future
        future = gevent.event.AsyncResult()

        def on_success(g):
            actor_ref = g.value
            actor_ref.ctx = ActorContext(self)
            if callback is not None:
                callback(actor_ref)
            future.set_result(actor_ref)

        def on_failure(g):
            try:
                g.get()
            except:
                t, ex, tb = sys.exc_info()
                future.set_exception(ex, exc_info=(t, ex, tb))

        p = self._pool.apply_async(self._send_remote, (address, binaries))
        p.link_value(on_success)
        p.link_exception(on_failure)
        return future

    def _async_run(self, str address, object binaries, object callback=None):
        cdef object future

        # return future
        future = gevent.event.AsyncResult()

        def on_success(g):
            ret = g.value
            if callback is not None:
                callback(ret)
            future.set_result(ret)

        def on_failure(g):
            try:
                g.get()
            except:
                t, ex, tb = sys.exc_info()
                future.set_exception(ex, exc_info=(t, ex, tb))

        p = self._pool.apply_async(self._send_remote, (address, binaries))
        p.link_value(on_success)
        p.link_exception(on_failure)
        return future

    cpdef object destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        cdef tuple binaries

        binaries = pack_destroy_actor_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref)[1:]

        if wait:
            return self._pool.apply(self._send_remote, (actor_ref.address, binaries))

        # return future
        return self._async_run(actor_ref.address, binaries, callback=callback)

    cpdef object has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        cdef tuple binaries

        binaries = pack_has_actor_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref)[1:]

        if wait:
            return self._pool.apply(self._send_remote, (actor_ref.address, binaries))

        # return future
        return self._async_run(actor_ref.address, binaries, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = self
        return ref

    cpdef _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                      bint wait=True, object callback=None):
        cdef object func
        cdef list binaries

        try:
            if wait_response:
                binaries = pack_send_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref, message)[1:]
            else:
                binaries = pack_tell_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref, message)[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle message {0}'.format(message))

        if wait:
            return self._pool.apply(self._send_remote, (actor_ref.address, binaries))

        # return future
        return self._async_run(actor_ref.address, binaries, callback=callback)

    cpdef send(self, ActorRef actor_ref, object message, bint wait=True, object callback=None):
        return self._send(actor_ref, message, wait_response=True, wait=wait, callback=callback)

    cpdef tell(self, ActorRef actor_ref, object message, object delay=None,
                      bint wait=True, object callback=None):
        if delay is not None:
            gevent.spawn_later(delay, self._send, actor_ref, message, wait_response=False,
                               wait=wait, callback=callback)
            return

        return self._send(actor_ref, message, wait_response=False, wait=wait, callback=callback)


cdef class Communicator(AsyncHandler):
    """
    Communicator, decide to send message to
    1) local actor pool, just spawn a greenlet to call the ActorExecutionContext.fire_run
    2) local dispatcher, mainly send message to another process, no pipe means no local dispatcher
    3) remote dispatcher which serves a stream handler

    Also, communicator receives message from dispatcher which received from different process or remote.
    """

    cdef public int index
    cdef public Distributor distributor

    cdef object pool
    cdef ClusterInfo cluster_info
    cdef object pipe
    cdef object remote_handler
    cdef object running
    cdef dict _handlers
    cdef object __weakref__

    def __init__(self, pool, ClusterInfo cluster_info, pipe, Distributor distributor=None, parallel=None):
        super(Communicator, self).__init__()
        AsyncHandler.__init__(self)

        self.pool = pool
        self.index = self.pool.index
        self.cluster_info = cluster_info
        self.pipe = pipe
        if distributor is None:
            self.distributor = Distributor(self.cluster_info.n_process)
        else:
            self.distributor = distributor
        self.remote_handler = ActorRemoteHelper(parallel=parallel)

        self.running = gevent.event.Event()
        self.running.set()

        self._handlers = {
            MessageType.send_all: self._on_receive_send,
            MessageType.tell_all: self._on_receive_tell,
            MessageType.create_actor: self._on_receive_create_actor,
            MessageType.destroy_actor: self._on_receive_destroy_actor,
            MessageType.has_actor: self._on_receive_has_actor,
            MessageType.result: self._on_receive_result,
            MessageType.error: self._on_receive_error,
        }

    cpdef _dispatch(self, local_func, redirect_func, remote_func, ActorRef actor_ref, args, kwargs):
        cdef int send_to_index
        if actor_ref.address is None or actor_ref.address == self.cluster_info.address:
            if self.cluster_info.n_process == 1:
                send_to_index = self.index
            else:
                send_to_index = self.distributor.distribute(actor_ref.uid)
            if send_to_index == self.index:
                # send to local actor_pool
                return local_func(*args, **kwargs)
            else:
                return redirect_func(send_to_index, *args, **kwargs)
        else:
            if self.cluster_info.standalone:
                raise ValueError('Not allow to send message to remote in standalone mode')

            return remote_func(*args, **kwargs)

    def _send_local(self, ActorRef actor_ref, object message, bint wait_response=True,
                    bint wait=True, object callback=None):
        cdef ActorExecutionContext actor_ctx
        cdef object p
        cdef object future
        cdef MessageContext message_ctx

        # send to self's actor pool
        actor_ctx = self.pool.get_actor_execution_ctx(actor_ref.uid)
        message_ctx = MessageContext(message)
        actor_ctx.receive(message_ctx)
        gevent.spawn(actor_ctx.fire_run)
        p = gevent.spawn(message_ctx.result)

        if wait:
            if wait_response:
                return p.get()
            else:
                return

        future = gevent.event.AsyncResult()

        if wait_response:
            # send, not wait
            def on_success(g):
                ret = g.value
                if callback is not None:
                    callback(ret)
                future.set_result(ret)

            def on_failure(g):
                try:
                    g.get()
                except:
                    t, ex, tb = sys.exc_info()
                    future.set_exception(ex, exc_info=(t, ex, tb))

            p.link_value(on_success)
            p.link_exception(on_failure)
        else:
            if callback is not None:
                callback(None)
            # tell, directly set future result None
            future.set_result(None)

        return future

    def _send_process(self, int to_index, ActorRef actor_ref, message, bint wait_response=True,
                      bint wait=True, callback=None):
        cdef object func
        cdef list msg
        cdef object future

        try:
            if wait_response:
                msg = pack_send_message(self.index, to_index, actor_ref, message)
            else:
                msg = pack_tell_message(self.index, to_index, actor_ref, message)
            message_id, message = msg[0], msg[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle message {0}'.format(message))

        if wait:
            self.pipe.put(*message)
            return self.submit(message_id)

        future = gevent.event.AsyncResult()

        def check():
            self.pipe.put(*message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef object _send_remote(self, ActorRef actor_ref, object message,
                              bint wait_response=True, bint wait=True, object callback=None):
        if wait_response:
            return self.remote_handler.send(actor_ref, message, wait=wait, callback=callback)
        else:
            return self.remote_handler.tell(actor_ref, message, wait=wait, callback=callback)

    cpdef _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                bint wait=True, object callback=None):
        return self._dispatch(self._send_local, self._send_process, self._send_remote, actor_ref,
                              (actor_ref, message),
                              dict(wait_response=wait_response, wait=wait, callback=callback))

    cpdef send(self, ActorRef actor_ref, object message, bint wait=True, object callback=None):
        return self._send(actor_ref, message, wait_response=True, wait=wait, callback=callback)

    cpdef tell(self, ActorRef actor_ref, object message, object delay=None,
               bint wait=True, object callback=None):
        if delay is not None:
            gevent.spawn_later(delay, self._send, actor_ref, message, wait_response=False,
                               wait=wait, callback=callback)
            return

        return self._send(actor_ref, message, wait_response=False, wait=wait, callback=callback)

    cpdef object _create_local_actor(self, ActorRef actor_ref, actor_cls, args, kwargs):
        cdef bint wait
        cdef object callback
        cdef object future
        cdef object res

        wait = kwargs.pop('wait', True)
        callback = kwargs.pop('callback', None)

        if wait:
            return self.pool.create_actor(actor_cls, actor_ref.uid, *args, **kwargs)

        future = gevent.event.AsyncResult()
        try:
            res = self.pool.create_actor(actor_cls, actor_ref.uid, *args, **kwargs)
            if callback is not None:
                callback(res)
            future.set_result(res)
        except:
            t, ex, tb = sys.exc_info()
            future.set_exception(ex, exc_info=(t, ex, tb))

        return future

    def _create_process_actor(self, int to_index, ActorRef actor_ref, object actor_cls,
                              args, kwargs):
        cdef bint wait
        cdef object callback
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        wait = kwargs.pop('wait', True)
        callback = kwargs.pop('callback', None)

        try:
            message_id, message = pack_create_actor_message(
                self.index, to_index, actor_ref, actor_cls, args, kwargs)
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        if wait:
            self.pipe.put(message)
            return self.submit(message_id)

        # return future
        future = gevent.event.AsyncResult()

        def check():
            self.pipe.put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef object _create_remote_actor(self, ActorRef actor_ref, object actor_cls, args, kwargs):
        return self.remote_handler.create_actor(actor_ref.address, actor_ref.uid,
                                                actor_cls, *args, **kwargs)

    def create_actor(self, str address, object uid,
                     object actor_cls, *args, **kwargs):
        cdef object actor_id
        cdef ActorRef actor_ref

        actor_id = uid or new_actor_id()
        actor_ref = ActorRef(address, actor_id)
        return self._dispatch(self._create_local_actor, self._create_process_actor,
                              self._create_remote_actor, actor_ref,
                              (actor_ref, actor_cls, args, kwargs), dict())

    cpdef _destroy_local_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        cdef object future
        cdef object res
        cdef object t
        cdef object ex
        cdef object tb

        if wait:
            return self.pool.destroy_actor(actor_ref.uid)

        future = gevent.event.AsyncResult()
        try:
            res = self.pool.destroy_actor(actor_ref.uid)
            if callback is not None:
                callback(res)
            future.set_result(res)
        except:
            t, ex, tb = sys.exc_info()
            future.set_exception(ex, exc_info=(t, ex, tb))

        return future

    def _destroy_process_actor(self, int to_index, ActorRef actor_ref,
                               bint wait=True, object callback=None):
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        message_id, message = pack_destroy_actor_message(self.index, to_index, actor_ref)
        if wait:
            self.pipe.put(message)
            return self.submit(message_id)

        future = gevent.event.AsyncResult()

        def check():
            self.pipe.put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        return self._dispatch(self._destroy_local_actor, self._destroy_process_actor,
                              self.remote_handler.destroy_actor, actor_ref,
                              (actor_ref,), dict(wait=wait, callback=callback))

    cpdef _has_local_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        if wait:
            return self.pool.has_actor(actor_ref.uid)

        future = gevent.event.AsyncResult()
        try:
            ret = self.pool.has_actor(actor_ref.uid)
            if callback is not None:
                callback(ret)
            future.set_result(ret)
        except:
            t, ex, tb = sys.exc_info()
            future.set_exception(ex, exc_info=(t, ex, tb))

        return future

    def _has_process_actor(self, int to_index, ActorRef actor_ref,
                           bint wait=True, object callback=None):
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        message_id, message = pack_has_actor_message(self.index, to_index, actor_ref)

        if wait:
            self.pipe.put(message)
            return self.submit(message_id)

        future = gevent.event.AsyncResult()

        def check():
            self.pipe.put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef object has_actor(self, ActorRef actor_ref, bint wait=True, callback=None):
        return self._dispatch(self._has_local_actor, self._has_process_actor,
                              self.remote_handler.has_actor, actor_ref,
                              (actor_ref,), dict(wait=wait, callback=callback))

    cpdef _on_receive_send(self, bytes binary, object callback):
        cdef object message
        cdef MessageContext message_ctx
        cdef ActorExecutionContext actor_ctx
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_send_message(binary)
        try:
            message_ctx = MessageContext(message.message)
            actor_ctx = self.pool.get_actor_execution_ctx(message.actor_ref.uid)
            actor_ctx.receive(message_ctx)
            gevent.spawn(actor_ctx.fire_run)

            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index, message_ctx.result())
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return callback(result_binary)

    cpdef _on_receive_tell(self, bytes binary, object callback):
        cdef object message
        cdef ActorExecutionContext actor_ctx
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_tell_message(binary)
        try:
            actor_ctx = self.pool.get_actor_execution_ctx(message.actor_ref.uid)
            actor_ctx.receive(MessageContext(message.message))
            # do not wait for the completion of greenlet
            gevent.spawn(actor_ctx.fire_run)
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index, None)
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return callback(result_binary)

    cpdef _on_receive_create_actor(self, bytes binary, object callback):
        cdef object message
        cdef ActorExecutionContext actor_ctx
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_create_actor_message(binary)
        try:
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index,
                self.pool.create_actor(message.actor_cls, message.actor_ref.uid,
                                       *message.args, **message.kwargs))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return callback(result_binary)

    cpdef _on_receive_destroy_actor(self, bytes binary, object callback):
        cdef object message
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_destroy_actor_message(binary)

        try:
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index,
                self.pool.destroy_actor(message.actor_ref.uid))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return callback(result_binary)

    cpdef _on_receive_has_actor(self, bytes binary, object callback):
        cdef object message
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_has_actor_message(binary)

        try:
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index,
                self.pool.has_actor(message.actor_ref.uid))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return callback(result_binary)

    cpdef _on_receive_result(self, bytes binary, object callback):
        cdef object message

        message = unpack_result_message(binary)
        self.got(message.message_id, message.result)

    cpdef _on_receive_error(self, bytes binary, object callback):
        cdef object message

        message = unpack_error_message(binary)
        self.err(message.message_id, message.error_type, message.error, message.traceback)

    cpdef on_receive(self, bytes binary, object callback=None):
        cdef int message_type

        if callback is None:
            callback = self.pipe.put

        message_type = unpack_message_type_value(binary)
        return self._handlers[message_type](binary, callback)

    def run(self):
        cdef bytes message

        self.running.clear()
        if self.pipe is not None:
            while True:
                message = self.pipe.get()
                gevent.spawn(self.on_receive, message)


cdef class Dispatcher(AsyncHandler):
    """
    Only created when more than 1 process.

    What dispatcher do is redirect the message to the destination process
    according to the distributor, and then return back the result to the original process.
    """

    cdef ClusterInfo cluster_info
    cdef int index
    cdef list pipes
    cdef list pipe_checkers
    cdef object distributor  # has to be object, not Distributor
    cdef object remote_handler
    cdef object checker_group
    cdef dict handlers

    def __init__(self, ClusterInfo cluster_info, list pipes, Distributor distributor=None, parallel=None):
        super(Dispatcher, self).__init__()
        AsyncHandler.__init__(self)

        self.cluster_info = cluster_info
        self.index = -1
        self.pipes = pipes  # type: list
        self.pipe_checkers = []
        self.checker_group = None
        if distributor is None:
            self.distributor = Distributor(self.cluster_info.n_process)
        else:
            self.distributor = distributor
        self.remote_handler = ActorRemoteHelper(parallel=parallel)

        self.handlers = {
            MessageType.send_all: self._on_receive_action,
            MessageType.tell_all: self._on_receive_action,
            MessageType.create_actor: self._on_receive_action,
            MessageType.destroy_actor: self._on_receive_action,
            MessageType.has_actor: self._on_receive_action,
            MessageType.result: self._on_receive_result,
            MessageType.error: self._on_receive_error,
        }

    cdef inline bint _is_remote(self, ActorRef actor_ref):
        if actor_ref.address is None or actor_ref.address == self.cluster_info.address:
            return False
        return True

    def _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                      bint wait=True, object callback=None):
        cdef int to_index
        cdef object func
        cdef list msg
        cdef bytes message_id
        cdef list messages
        cdef object future

        if self._is_remote(actor_ref):
            if wait_response:
                return self.remote_handler.send(actor_ref, message, wait=wait, callback=callback)
            else:
                return self.remote_handler.tell(actor_ref, message, wait=wait, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)
        try:
            if wait_response:
                msg = pack_send_message(self.index, to_index, actor_ref, message)
            else:
                msg = pack_tell_message(self.index, to_index, actor_ref, message)
            message_id, messages = msg[0], msg[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle message: {0}'.format(message))

        if wait:
            self.pipes[to_index].put(*messages)
            return self.submit(message_id)

        # return future
        future = gevent.event.AsyncResult()

        def check():
            self.pipes[to_index].put(*messages)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef object send(self, ActorRef actor_ref, object message, bint wait=True, object callback=None):
        return self._send(actor_ref, message, wait_response=True, wait=wait, callback=callback)

    cpdef object tell(self, ActorRef actor_ref, object message, object delay=None,
                      bint wait=True, object callback=None):
        if delay is not None:
            gevent.spawn_later(delay, self._send, actor_ref, message, wait_response=False,
                               wait=wait, callback=callback)
            return

        return self._send(actor_ref, message, wait_response=False, wait=wait, callback=callback)

    def create_actor(self, str address, object uid, object actor_cls, *args, **kwargs):
        cdef object actor_id
        cdef ActorRef actor_ref
        cdef bint wait
        cdef object callback
        cdef bytes message_id
        cdef bytearray message
        cdef int to_index
        cdef object future

        actor_id = uid or new_actor_id()
        actor_ref = ActorRef(address, actor_id)

        if self._is_remote(actor_ref):
            return self.remote_handler.create_actor(address, uid, actor_cls, *args, **kwargs)

        wait = kwargs.pop('wait', True)
        callback = kwargs.pop('callback', None)

        to_index = self.distributor.distribute(actor_id)
        try:
            message_id, message = pack_create_actor_message(
                self.index, to_index, actor_ref, actor_cls, args, kwargs)
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        if wait:
            self.pipes[to_index].put(message)
            return self.submit(message_id)

        # return future
        future = gevent.event.AsyncResult()

        def check():
            self.pipes[to_index].put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    def destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        cdef int to_index
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        if self._is_remote(actor_ref):
            return self.remote_handler.destroy_actor(actor_ref, wait=wait, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)

        message_id, message = pack_destroy_actor_message(
            self.index, to_index, actor_ref)

        if wait:
            self.pipes[to_index].put(message)
            return self.submit(message_id)

        # return future
        future = gevent.event.AsyncResult()

        def check():
            self.pipes[to_index].put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    def has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        cdef int to_index
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        if self._is_remote(actor_ref):
            return self.remote_handler.has_actor(actor_ref, wait=wait, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)

        message_id, message = pack_has_actor_message(
            self.index, to_index, actor_ref)

        if wait:
            self.pipes[to_index].put(message)
            return self.submit(message_id)

        # return future
        future = gevent.event.AsyncResult()

        def check():
            self.pipes[to_index].put(message)
            self.future(message_id, future, callback=callback)

        gevent.spawn(check)
        return future

    cpdef _on_receive_action(self, bytes binary):
        cdef int from_index
        cdef int to_index
        cdef bytes message_id

        from_index, to_index = get_index(binary, self.distributor.distribute)
        message_id = unpack_message_id(binary)

        if from_index == REMOTE_FROM_INDEX:
            # sent from remote, redirect to process, and wait for result
            self.pipes[to_index].put(binary)
            return self.wait(message_id)
        else:
            # sent from other process, just redirect
            self.pipes[to_index].put(binary)

    cpdef _on_receive_result(self, bytes binary):
        cdef int from_index
        cdef int to_index
        cdef bytes message_id
        cdef object result

        from_index, to_index = get_index(binary, None)
        if to_index == self.index:
            # sent from the dispatcher
            result = unpack_result_message(binary, from_index=from_index, to_index=to_index)
            self.got(result.message_id, result.result)
        elif to_index == REMOTE_FROM_INDEX:
            # sent from remote
            message_id = unpack_message_id(binary)
            self.got(message_id, binary)
        else:
            # sent from process
            self.pipes[to_index].put(binary)

    cpdef _on_receive_error(self, binary):
        cdef int to_index
        cdef object err
        cdef bytes message_id

        _, to_index = get_index(binary, None)
        if to_index == self.index:
            # sent from the dispatcher
            err = unpack_error_message(binary)
            self.err(err.message_id, err.error_type, err.error, err.traceback)
        elif to_index == REMOTE_FROM_INDEX:
            # sent from remote
            message_id = unpack_message_id(binary)
            self.got(message_id, binary)
        else:
            # sent from process
            self.pipes[to_index].put(binary)

    cpdef on_receive(self, bytes binary):
        cdef int message_type

        message_type = unpack_message_type_value(binary)
        return self.handlers[message_type](binary)

    def replace_pipe(self, idx):
        new_gl = gevent.spawn(self._check_pipe, self.pipes[idx])
        self.checker_group.start(new_gl)
        self.checker_group.discard(self.pipe_checkers[idx])
        self.pipe_checkers[idx] = new_gl

    def _check_pipe(self, pipe):
        cdef bytes message
        while True:
            try:
                message = pipe.get()
            except EOFError:
                # broken pipe
                return
            try:
                gevent.spawn(self.on_receive, message)
            except BrokenPipeError:
                return

    def run(self):
        while True:
            self.checker_group = gevent.pool.Group()
            for pipe in self.pipes:
                gl = gevent.spawn(self._check_pipe, pipe)
                self.pipe_checkers.append(gl)
                self.checker_group.start(gl)
            self.checker_group.join()
            gevent.sleep(0.05)
            if all(p is None for p in self.pipes):
                break


cdef class ActorServerHandler:
    cdef object sender
    cdef bint multi_process
    cdef object server

    def __init__(self, sender, multi_process):
        self.sender = sender
        self.multi_process = multi_process
        self.server = None

    def set_server(self, server):
        self.server = weakref.ref(server)

    def on_receive(self, bytes binary):
        if not self.multi_process:
            return self.sender.on_receive(binary, callback=lambda x: x)
        else:
            return self.sender.on_receive(binary)

    def __call__(self, sock, address):
        cdef bytes binary
        cdef object p

        server = self.server()
        if server:
            server.add_sock(sock)

        while True:
            try:
                binary = read_remote_message(sock.recv)
                p = gevent.spawn(self.on_receive, binary)
                write_remote_message(sock.sendall, p.get())
            except (gevent.socket.error, struct.error):
                break


class ActorStreamServer(gevent.server.StreamServer):
    def __init__(self, listener, handler):
        super(ActorStreamServer, self).__init__(listener, handler)
        self._socks = []
        handler.set_server(self)

    def add_sock(self, sock):
        self._socks.append(sock)

    def close(self):
        super(ActorStreamServer, self).close()
        [sock.close() for sock in self._socks]


cpdef object start_actor_server(ClusterInfo cluster_info, object sender):
    cdef str address
    cdef int port
    cdef bint multi_process
    cdef object s

    address, port = cluster_info.location, cluster_info.port
    if address is None or port is None:
        raise ValueError('Both address and port should be provided')
    multi_process = cluster_info.n_process > 1

    s = ActorStreamServer((address, port), ActorServerHandler(sender, multi_process))
    s.start()
    return s


def start_local_pool(int index, ClusterInfo cluster_info,
                     object pipe=None, Distributor distributor=None,
                     object parallel=None, bint join=False):
    # new process will pickle the numpy RandomState, we seed the random one
    import numpy as np
    np.random.seed()

    # gevent.signal(signal.SIGINT, lambda *_: None)

    # all these work in a single process
    # we start a local pool to handle messages
    # and a communicator to do the redirection of messages
    local_pool = LocalActorPool(cluster_info.address, index)
    comm = Communicator(local_pool, cluster_info, pipe, distributor, parallel)
    local_pool.set_comm(comm)
    p = gevent.spawn(comm.run)
    if join:
        p.get()
    else:
        return comm


def close_pipe(p):
    if p is None:
        return
    for _ in range(3):
        try:
            p.close()
        except gipc.GIPCClosed:
            break
        except gipc.GIPCLocked:
            continue


cdef class ActorPool:
    """
    1) If only 1 process, start local pool
    2) More than 1, start dispatcher, and start local pool in several times
    """

    cdef public ClusterInfo cluster_info
    cdef public object _dispatcher
    cpdef public Distributor distributor
    cdef bint _started
    cdef object _stopped
    cdef bint _multi_process
    cdef public object _server
    cdef object _parallel
    cdef list _stop_funcs
    cdef list _processes
    cdef list _comm_pipes
    cdef list _pool_pipes

    def __init__(self, ClusterInfo cluster_info, Distributor distributor=None, parallel=None):
        self.cluster_info = cluster_info
        self.distributor = distributor

        self._started = False
        self._stopped = gevent.event.Event()
        self._multi_process = self.cluster_info.n_process > 1
        self._dispatcher = None
        self._server = None
        self._parallel = parallel
        self._stop_funcs = []
        self._processes = []
        self._comm_pipes = []
        self._pool_pipes = []

    cdef int _check_started(self) except -1:
        if not self._started:
            raise ActorPoolNotStarted('Actor pool need to run first')
        return 0

    def create_actor(self, object actor_cls, *args, **kwargs):
        cdef bint wait
        cdef object address
        cdef object uid
        cdef ActorRef actor_ref

        self._check_started()

        wait = kwargs.pop('wait', True)

        address = kwargs.pop('address', None)
        uid = kwargs.pop('uid', None)

        if wait:
            actor_ref = self._dispatcher.create_actor(address, uid,
                                                      actor_cls, *args, **kwargs)
            if address:
                actor_ref.address = address
            actor_ref.ctx = ActorContext(self._dispatcher)
            return actor_ref

        def callback(ref):
            if address is not None:
                ref.address = address
            ref.ctx = ActorContext(self._dispatcher)

        kwargs['wait'] = False
        kwargs['callback'] = callback
        return self._dispatcher.create_actor(address, uid,
                                             actor_cls, *args, **kwargs)

    cpdef object destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        self._check_started()

        return self._dispatcher.destroy_actor(actor_ref, wait=wait, callback=callback)

    cpdef object has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        self._check_started()

        return self._dispatcher.has_actor(actor_ref, wait=wait, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = ActorContext(self._dispatcher)
        return ref

    @property
    def processes(self):
        return self._processes

    cpdef tuple _start_process(self, idx):
        comm_pipe, pool_pipe = gipc.pipe(True, encoder=_inaction_encoder, decoder=_inaction_decoder)
        p = gipc.start_process(start_local_pool,
                               args=(idx, self.cluster_info, comm_pipe, self.distributor),
                               kwargs={'parallel': self._parallel, 'join': True}, daemon=True)
        return p, comm_pipe, pool_pipe

    cpdef restart_process(self, int idx):
        if self._processes[idx].is_alive():
            self._processes[idx].terminate()
        close_pipe(self._comm_pipes[idx])
        close_pipe(self._pool_pipes[idx])
        self._processes[idx], self._comm_pipes[idx], self._pool_pipes[idx] = self._start_process(idx)
        self._dispatcher.replace_pipe(idx)

    def run(self):
        if self._started:
            return

        if not self._multi_process:
            # only start local pool
            self._dispatcher = start_local_pool(0, self.cluster_info, distributor=self.distributor,
                                                parallel=self._parallel)
        else:
            self._processes, self._comm_pipes, self._pool_pipes = [list(tp) for tp in zip(
                *(self._start_process(idx) for idx in range(self.cluster_info.n_process))
            )]

            def stop_func():
                for process in self._processes:
                    process.terminate()
                for idx, p in enumerate(self._comm_pipes):
                    close_pipe(p)
                    self._comm_pipes[idx] = None
                for idx, p in enumerate(self._pool_pipes):
                    close_pipe(p)
                    self._pool_pipes[idx] = None

            self._stop_funcs.append(stop_func)

            # start dispatcher
            self._dispatcher = Dispatcher(self.cluster_info, self._pool_pipes,
                                          self.distributor)
            gevent.spawn(self._dispatcher.run)

        if not self.cluster_info.standalone:
            # start stream server to handle remote message
            self._server = start_actor_server(self.cluster_info, self._dispatcher)

            def close():
                self._server.stop()

            self._stop_funcs.append(close)

        self._started = True

    @staticmethod
    def sleep(seconds):
        gevent.sleep(seconds)

    def join(self, timeout=None):
        self._stopped.wait(timeout)

    cpdef stop(self):
        try:
            for stop_func in self._stop_funcs:
                stop_func()
        finally:
            self._stopped.set()
            self._started = False

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, *_):
        self.stop()


cdef class ActorClient:
    cdef object remote_handler

    def __init__(self, parallel=None):
        self.remote_handler = ActorRemoteHelper(parallel)

    def create_actor(self, object actor_cls, *args, **kwargs):
        cdef object address
        cdef object uid

        if 'address' not in kwargs or kwargs.get('address') is None:
            raise ValueError('address must be provided')
        address = kwargs.pop('address')
        uid = kwargs.pop('uid', new_actor_id())
        return self.remote_handler.create_actor(address, uid, actor_cls, *args, **kwargs)

    cpdef object has_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        return self.remote_handler.has_actor(actor_ref, wait=wait, callback=callback)

    cpdef destroy_actor(self, ActorRef actor_ref, bint wait=True, object callback=None):
        return self.remote_handler.destroy_actor(actor_ref, wait=wait, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = self.remote_handler.actor_ref(*args, **kwargs)
        if ref.address is None:
            raise ValueError('address must be provided')
        return ref

    @staticmethod
    def sleep(seconds):
        gevent.sleep(seconds)

    @staticmethod
    def popen(*args, **kwargs):
        return gevent.subprocess.Popen(*args, **kwargs)

    @staticmethod
    def threadpool(size):
        return ThreadPool(size)

    @staticmethod
    def asyncpool(size=None):
        return gevent.pool.Pool(size)
