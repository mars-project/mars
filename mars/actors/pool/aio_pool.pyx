#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from __future__ import absolute_import

import asyncio
import functools
import itertools
import os
import pickle
import random
import socket
import struct
import sys
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy

from ...lib import asyncio_pool, aiomultiprocess
from ..cluster cimport ClusterInfo
from ..core cimport ActorRef, Actor
from ..distributor cimport Distributor
from ..errors import ActorPoolNotStarted, ActorNotExist, ActorAlreadyExist
from .messages cimport pack_send_message, pack_tell_message, pack_create_actor_message, \
    pack_destroy_actor_message, pack_result_message, unpack_send_message, unpack_tell_message, \
    unpack_create_actor_message, unpack_destroy_actor_message, unpack_result_message, \
    pack_has_actor_message, unpack_has_actor_message, pack_error_message, unpack_error_message, \
    unpack_message_type_value, unpack_message_type, unpack_message_id, get_index, MessageType
from .utils cimport new_actor_id
from .utils import create_actor_ref

cdef int REMOTE_FROM_INDEX = -2
cdef int UNKNOWN_TO_INDEX = -1
cpdef int REMOTE_DEFAULT_PARALLEL = 50  # parallel connection at most
cpdef int REMOTE_MAX_CONNECTION = 200  # most connections

if sys.platform == 'darwin':
    aiomultiprocess.set_context('fork')


cdef class MessageContext:
    cdef public object message
    cdef public object async_result

    def __init__(self, message):
        self.message = message
        self.async_result = asyncio.Future()

    async def result(self):
        return await self.async_result


cdef class ActorExecutionContext:
    cdef public object actor
    cdef public object lock
    cdef public object inbox
    cdef public object comm

    def __init__(self, actor, comm):
        self.actor = weakref.ref(actor)
        self.lock = asyncio.locks.Lock()
        self.inbox = asyncio.queues.Queue()
        self.comm = comm

    async def fire_run(self):
        cdef MessageContext message_ctx

        async with self.lock:
            message_ctx = await self.inbox.get()
            actor = self.actor()

            try:
                res = await actor.on_receive(message_ctx.message)
                message_ctx.async_result.set_result(res)
                return res
            except:
                t, ex, tb = sys.exc_info()
                message_ctx.async_result.set_exception(ex.with_traceback(tb))
                raise

    async def receive(self, MessageContext message_ctx):
        await self.inbox.put(message_ctx)


class AioThreadPool:
    def __init__(self, num_threads):
        self._pool = ThreadPoolExecutor(num_threads)

    def submit(self, fn, *args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            self._pool, functools.partial(fn, *args, **kwargs))


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

    async def create_actor(self, actor_cls, *args, **kwargs):
        cdef object address
        cdef object uid
        cdef object ref

        address = kwargs.pop('address', None)
        uid = kwargs.pop('uid', None)

        ref = await self._comm.create_actor(address, uid, actor_cls, *args, **kwargs)
        ref.ctx = self
        return ref

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        return await self._comm.destroy_actor(actor_ref, callback=callback)

    async def has_actor(self, ActorRef actor_ref, object callback=None):
        return await self._comm.has_actor(actor_ref, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = self
        return ref

    async def send(self, ActorRef actor_ref, object message, object callback=None):
        return await self._comm.send(actor_ref, message, callback=callback)

    async def tell(self, ActorRef actor_ref, object message, object delay=None, object callback=None):
        return await self._comm.tell(actor_ref, message, delay=delay, callback=callback)

    @staticmethod
    def threadpool(size):
        return AioThreadPool(size)


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

    async def create_actor(self, object actor_cls, object uid, *args, **kw):
        cdef Actor actor

        actor = actor_cls(*args, **kw)
        actor.address = self.address
        actor.uid = uid
        actor_execution_ctx = ActorExecutionContext(actor, self.comm)
        actor.ctx = ActorContext(actor_execution_ctx.comm)
        if uid in self.actors:
            raise ActorAlreadyExist('Actor {0} already exist, cannot create'.format(actor.uid))
        self.actors[uid] = (actor, actor_execution_ctx)
        await actor.post_create()
        return ActorRef(self.address, uid)

    cpdef bint has_actor(self, object actor_uid) except -128:
        if actor_uid not in self.actors:
            return False
        return True

    async def destroy_actor(self, object actor_uid):
        cdef Actor actor

        try:
            actor, _ = self.actors[actor_uid]
        except KeyError:
            raise ActorNotExist('Actor {0} does not exist'.format(actor_uid))
        await actor.pre_destroy()
        del self.actors[actor_uid]
        return actor_uid


cdef class AsyncHandler:
    cdef dict async_results

    def __init__(self):
        self.async_results = dict()

    async def submit(self, bytes unique_id):
        cdef object ar

        ar = self.async_results[unique_id] = asyncio.Future()

        # wait for result
        try:
            return await ar
        finally:
            del self.async_results[unique_id]

    async def future(self, bytes unique_id, object future, object callback=None):
        cdef object ar, result

        ar = self.async_results[unique_id] = asyncio.Future()

        # wait for result
        try:
            result = await ar
            if callback:
                callback(result)
            future.set_result(result)
        except:  # noqa: E722
            future.set_exception(ar.exception())
        finally:
            del self.async_results[unique_id]

    async def wait(self, bytes unique_id):
        cdef object ar

        ar = self.async_results[unique_id] = asyncio.Future()

        # wait for result
        try:
            return await ar
        except:  # noqa: E722
            return ar.exception()
        finally:
            del self.async_results[unique_id]

    cpdef void got(self, bytes unique_id, object result):
        # receive event handling
        self.async_results[unique_id].set_result(result)

    cpdef void err(self, bytes unique_id, object t, object ex, object tb):
        self.async_results[unique_id].set_exception(ex.with_traceback(tb))


cdef int PIPE_BUF_SIZE = 65536


cdef class AsyncIOPair:
    cdef object socket, pipe_fds
    cdef object reader, writer
    cdef object init_lock, read_lock, write_lock
    cdef int chunk_size

    def __init__(self, socket=None, reader_writer=None, pipe_fds=None):
        self.socket = socket
        self.pipe_fds = pipe_fds

        if reader_writer is None:
            self.reader, self.writer = None, None
        else:
            self.reader, self.writer = reader_writer

        if pipe_fds is None:
            self.chunk_size = -1
        else:
            self.chunk_size = PIPE_BUF_SIZE

        self.init_lock = asyncio.locks.Lock()
        self.read_lock = asyncio.locks.Lock()
        self.write_lock = asyncio.locks.Lock()

    @property
    def socket_fileno(self):
        return self.writer.transport.get_extra_info('socket').fileno()

    def reset_reader_writer(self):
        self.reader, self.writer = None, None
        self.init_lock = asyncio.locks.Lock()
        self.read_lock = asyncio.locks.Lock()
        self.write_lock = asyncio.locks.Lock()

    async def _build_reader_writer(self):
        async with self.init_lock:
            if self.reader is not None:
                return
            if self.socket is not None:
                self.reader, self.writer = await asyncio.open_connection(sock=self.socket)
            elif self.pipe_fds is not None:
                loop = asyncio.get_event_loop()
                self.reader = asyncio.StreamReader(loop=loop)
                reader_protocol = asyncio.StreamReaderProtocol(self.reader)
                await loop.connect_read_pipe(lambda: reader_protocol, os.fdopen(self.pipe_fds[0], 'rb'))

                write_protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader())
                write_transport, _ = await loop.connect_write_pipe(
                    lambda: write_protocol, os.fdopen(self.pipe_fds[1], 'wb'))
                self.writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)

    async def read(self):
        cdef uint64_t size
        cdef bytearray buf
        cdef uint32_t received_size
        cdef uint64_t left
        cdef uint64_t size_to_read
        cdef bytes read_bytes
        if self.reader is None:
            await self._build_reader_writer()

        async with self.read_lock:
            try:
                read_bytes = await self.reader.read(8)
            except ConnectionResetError:
                raise BrokenPipeError('The remote server is closed')
            if len(read_bytes) == 0:
                raise BrokenPipeError('The remote server is closed')

            size = 0
            memcpy(<char *>&size, <const char *>read_bytes, 8)
            buf = bytearray()
            received_size = 0
            left = size
            if self.chunk_size > 0:
                size_to_read = min(left, self.chunk_size)
            else:
                size_to_read = left

            while True:
                try:
                    read_bytes = await self.reader.read(size_to_read)
                except ConnectionResetError:
                    raise BrokenPipeError('The remote server is closed')
                if read_bytes == 0:
                    raise BrokenPipeError('The remote server is closed')

                buf.extend(read_bytes)
                received_size += len(read_bytes)
                if received_size >= size:
                    break
                left = size - received_size

                if self.chunk_size > 0:
                    size_to_read = min(left, self.chunk_size)
                else:
                    size_to_read = left

        return bytes(buf)

    async def write(self, *binary):
        cdef bytes size_bytes
        cdef uint64_t size = 0
        if self.writer is None:
            await self._build_reader_writer()

        async with self.write_lock:
            for b in binary:
                size += len(b)
            size_bytes = (<char *>&size)[:sizeof(uint64_t)]
            self.writer.write(size_bytes)

            for b in binary:
                self.writer.write(b)

            await self.writer.drain()

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        if self.socket is not None:
            self.socket.close()
        elif self.pipe_fds is not None:
            os.close(self.pipe_fds[0])
            try:
                os.close(self.pipe_fds[1])
            except OSError:
                pass


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
    global_lock = asyncio.locks.Semaphore()
    addrs = 0

    @classmethod
    async def create(cls, *args, **kwargs):
        async with cls.global_lock:
            Connections.addrs += 1
            return cls(*args, **kwargs)

    def __init__(self, address):
        if isinstance(address, str):
            self.address = address.split(':', 1)
        else:
            self.address = address

        self.lock = asyncio.locks.Semaphore()
        self.conn_locks = OrderedDict()

    @property
    def conn(self):
        return [conn_lock[0] for conn_lock in self.conn_locks.values()]

    def _connect(self, conn, lock):
        return Connection(conn, lock)

    def got_broken_pipe(self, fd):
        del self.conn_locks[fd]

    async def connect(self):
        cdef int maxlen, fd
        cdef AsyncIOPair conn
        cdef object lock

        async with self.lock:
            for conn, lock in self.conn_locks.values():
                # try to reuse the connections before
                if lock.locked():
                    continue
                await lock.acquire()
                return self._connect(conn, lock)

            maxlen = max(REMOTE_MAX_CONNECTION // Connections.addrs, 1)

            if len(self.conn) < maxlen:
                # create a new connection
                lock = asyncio.locks.Semaphore()
                await lock.acquire()
                reader, writer = await asyncio.open_connection(*self.address)
                conn = AsyncIOPair(reader_writer=(reader, writer))
                self.conn_locks[conn.socket_fileno] = (conn, lock)
                return self._connect(conn, lock)

            async def close(c, lk):
                async with lk:
                    await c.close()

            ps = [asyncio.ensure_future(close(c, l)) for c, l in
                  itertools.islice(self.conn_locks.values(), maxlen, len(self.conn_locks))]

            i = random.randint(0, maxlen - 1)
            fd = next(itertools.islice(self.conn_locks.keys(), i, i + 1))
            conn, lock = self.conn_locks[fd]
            await lock.acquire()

            # wait for conn finished
            await asyncio.wait(ps, return_when=asyncio.ALL_COMPLETED)
            self.conn_locks = OrderedDict(itertools.islice(self.conn_locks.items(), maxlen))

            return self._connect(conn, lock)

    def __del__(self):
        for c, _ in self.conn_locks.values():
            try:
                asyncio.ensure_future(c.close())
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
        self._pool = asyncio_pool.AioPool(size=self._parallel)
        self._connections = dict()
        self._lock = asyncio.locks.Lock()

    async def _new_connection(self, str address):
        async with self._lock:
            if address not in self._connections:
                connections = await Connections.create(address)
                self._connections[address] = connections

            return await self._connections[address].connect()

    async def _send_remote(self, str address, object binary):
        cdef bytes res_binary
        cdef object message_type

        with await self._new_connection(address) as conn:
            try:
                await conn.write(*binary)
                res_binary = await conn.read()
                message_type = unpack_message_type(res_binary)
                if message_type == MessageType.error:
                    error_message = unpack_error_message(res_binary)
                    raise error_message.error.with_traceback(error_message.traceback) from None
                else:
                    assert message_type == MessageType.result
                    return unpack_result_message(res_binary).result
            except BrokenPipeError:
                self._connections[address].got_broken_pipe(conn.socket_fileno)
                raise

    async def create_actor(self, str address, object uid, object actor_cls, *args, **kwargs):
        cdef object callback
        cdef tuple binaries
        cdef ActorRef actor_ref
        cdef object future

        callback = kwargs.pop('callback', None)

        try:
            binaries = pack_create_actor_message(
                ActorRemoteHelper.index, UNKNOWN_TO_INDEX,
                ActorRef(address, uid), actor_cls, args, kwargs)[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        actor_ref = await self._pool.exec(self._send_remote(address, binaries))
        actor_ref.ctx = ActorContext(self)
        return actor_ref

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        cdef tuple binaries
        binaries = pack_destroy_actor_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref)[1:]
        return await self._pool.exec(self._send_remote(actor_ref.address, binaries))

    async def has_actor(self, ActorRef actor_ref, object callback=None):
        cdef tuple binaries
        binaries = pack_has_actor_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref)[1:]
        return await self._pool.exec(self._send_remote(actor_ref.address, binaries))

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = self
        return ref

    async def _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                    object callback=None):
        cdef object func
        cdef list binaries

        try:
            if wait_response:
                binaries = pack_send_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref, message)[1:]
            else:
                binaries = pack_tell_message(ActorRemoteHelper.index, UNKNOWN_TO_INDEX, actor_ref, message)[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle message {0}'.format(message))

        return await self._pool.exec(self._send_remote(actor_ref.address, binaries))

    async def send(self, ActorRef actor_ref, object message, object callback=None):
        return await self._send(actor_ref, message, wait_response=True, callback=callback)

    async def tell(self, ActorRef actor_ref, object message, object delay=None, object callback=None):
        if delay is not None:
            async def delay_tell():
                await asyncio.sleep(delay)
                return await self._send(actor_ref, message, wait_response=False, callback=callback)

            asyncio.ensure_future(delay_tell())
        else:
            return await self._send(actor_ref, message, wait_response=False, callback=callback)


cdef object _no_callback = object()


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
        super().__init__()
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

        self.running = asyncio.locks.Event()
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

    async def _dispatch(self, local_func, redirect_func, remote_func, ActorRef actor_ref, args, kwargs):
        cdef int send_to_index
        cdef object local_result
        if actor_ref.address is None or actor_ref.address == self.cluster_info.address \
                or actor_ref.address == self.cluster_info.advertise_address:
            if self.cluster_info.n_process == 1:
                send_to_index = self.index
            else:
                send_to_index = self.distributor.distribute(actor_ref.uid)
            if send_to_index == self.index:
                # send to local actor_pool
                local_result = local_func(*args, **kwargs)
                if asyncio.iscoroutine(local_result):
                    local_result = await local_result
                return local_result
            else:
                return await redirect_func(send_to_index, *args, **kwargs)
        else:
            if self.cluster_info.standalone:
                raise ValueError('Not allow to send message to remote in standalone mode')

            return await remote_func(*args, **kwargs)

    async def _send_local(self, ActorRef actor_ref, object message, bint wait_response=True,
                          object callback=None):
        cdef ActorExecutionContext actor_ctx
        cdef object p
        cdef object future
        cdef MessageContext message_ctx

        # send to self's actor pool
        actor_ctx = self.pool.get_actor_execution_ctx(actor_ref.uid)
        message_ctx = MessageContext(message)
        await actor_ctx.receive(message_ctx)
        asyncio.ensure_future(actor_ctx.fire_run())

        if wait_response:
            return await message_ctx.result()
        else:
            return

    async def _send_process(self, int to_index, ActorRef actor_ref, message, bint wait_response=True,
                            callback=None):
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

        await self.pipe.write(*message)
        return await self.submit(message_id)

    async def _send_remote(self, ActorRef actor_ref, object message, bint wait_response=True,
                           object callback=None):
        if wait_response:
            return await self.remote_handler.send(actor_ref, message, callback=callback)
        else:
            return await self.remote_handler.tell(actor_ref, message, callback=callback)

    async def _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                    object callback=None):
        return await self._dispatch(self._send_local, self._send_process, self._send_remote, actor_ref,
                                    (actor_ref, message), dict(wait_response=wait_response, callback=callback))

    async def send(self, ActorRef actor_ref, object message, object callback=None):
        return await self._send(actor_ref, message, wait_response=True, callback=callback)

    async def tell(self, ActorRef actor_ref, object message, object delay=None, object callback=None):
        if delay is not None:
            async def delay_tell():
                await asyncio.sleep(delay)
                return await self._send(actor_ref, message, wait_response=False, callback=callback)

            asyncio.ensure_future(delay_tell())
        else:
            return await self._send(actor_ref, message, wait_response=False, callback=callback)

    async def _create_local_actor(self, ActorRef actor_ref, actor_cls, args, kwargs):
        cdef bint wait
        cdef object callback

        callback = kwargs.pop('callback', None)
        return await self.pool.create_actor(actor_cls, actor_ref.uid, *args, **kwargs)

    async def _create_process_actor(self, int to_index, ActorRef actor_ref, object actor_cls,
                                    args, kwargs):
        cdef object callback
        cdef bytes message_id
        cdef bytearray message

        callback = kwargs.pop('callback', None)

        try:
            message_id, message = pack_create_actor_message(
                self.index, to_index, actor_ref, actor_cls, args, kwargs)
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        await self.pipe.write(message)
        return await self.submit(message_id)

    async def _create_remote_actor(self, ActorRef actor_ref, object actor_cls, args, kwargs):
        return await self.remote_handler.create_actor(
            actor_ref.address, actor_ref.uid, actor_cls, *args, **kwargs)

    async def create_actor(self, str address, object uid,
                           object actor_cls, *args, **kwargs):
        cdef object actor_id
        cdef ActorRef actor_ref

        actor_id = uid or new_actor_id()
        actor_ref = ActorRef(address, actor_id)
        return await self._dispatch(self._create_local_actor, self._create_process_actor,
                                    self._create_remote_actor, actor_ref,
                                    (actor_ref, actor_cls, args, kwargs), dict())

    async def _destroy_local_actor(self, ActorRef actor_ref, object callback=None):
        return await self.pool.destroy_actor(actor_ref.uid)

    async def _destroy_process_actor(self, int to_index, ActorRef actor_ref, object callback=None):
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        message_id, message = pack_destroy_actor_message(self.index, to_index, actor_ref)
        await self.pipe.write(message)
        return await self.submit(message_id)

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        return await self._dispatch(self._destroy_local_actor, self._destroy_process_actor,
                              self.remote_handler.destroy_actor, actor_ref,
                              (actor_ref,), dict(callback=callback))

    cpdef _has_local_actor(self, ActorRef actor_ref, object callback=None):
        return self.pool.has_actor(actor_ref.uid)

    async def _has_process_actor(self, int to_index, ActorRef actor_ref, object callback=None):
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        message_id, message = pack_has_actor_message(self.index, to_index, actor_ref)

        await self.pipe.write(message)
        return await self.submit(message_id)

    async def has_actor(self, ActorRef actor_ref, callback=None):
        return await self._dispatch(self._has_local_actor, self._has_process_actor,
                                    self.remote_handler.has_actor, actor_ref,
                                    (actor_ref,), dict(callback=callback))

    async def _on_receive_send(self, bytes binary, object callback):
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
            await actor_ctx.receive(message_ctx)
            asyncio.ensure_future(actor_ctx.fire_run())

            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index, await message_ctx.result())
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return result_binary if callback is _no_callback else await callback(result_binary)

    async def _on_receive_tell(self, bytes binary, object callback):
        cdef object message
        cdef ActorExecutionContext actor_ctx
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_tell_message(binary)
        try:
            actor_ctx = self.pool.get_actor_execution_ctx(message.actor_ref.uid)
            await actor_ctx.receive(MessageContext(message.message))
            # do not wait for the completion of greenlet
            asyncio.ensure_future(actor_ctx.fire_run())
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index, None)
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return result_binary if callback is _no_callback else await callback(result_binary)

    async def _on_receive_create_actor(self, bytes binary, object callback):
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
                await self.pool.create_actor(message.actor_cls, message.actor_ref.uid,
                                             *message.args, **message.kwargs))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return result_binary if callback is _no_callback else await callback(result_binary)

    async def _on_receive_destroy_actor(self, bytes binary, object callback):
        cdef object message
        cdef bytearray result_binary
        cdef object t
        cdef object ex
        cdef object tb

        message = unpack_destroy_actor_message(binary)

        try:
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index,
                await self.pool.destroy_actor(message.actor_ref.uid))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return result_binary if callback is _no_callback else await callback(result_binary)

    async def _on_receive_has_actor(self, bytes binary, object callback):
        cdef object message
        cdef bytearray result_binary
        cdef object t, ex, tb

        message = unpack_has_actor_message(binary)

        try:
            _, result_binary = pack_result_message(
                message.message_id, self.index, message.from_index,
                self.pool.has_actor(message.actor_ref.uid))
        except:
            t, ex, tb = sys.exc_info()
            _, result_binary = pack_error_message(
                message.message_id, self.index, message.from_index, t, ex, tb)

        return result_binary if callback is _no_callback else await callback(result_binary)

    async def _on_receive_result(self, bytes binary, object callback):
        cdef object message

        message = unpack_result_message(binary)
        self.got(message.message_id, message.result)

    async def _on_receive_error(self, bytes binary, object callback):
        cdef object message

        message = unpack_error_message(binary)
        self.err(message.message_id, message.error_type, message.error, message.traceback)

    async def on_receive(self, bytes binary, object callback=None):
        cdef int message_type

        if callback is None:
            async def callback(data):
                await self.pipe.write(data)

        message_type = unpack_message_type_value(binary)
        return await self._handlers[message_type](binary, callback)

    async def run(self):
        cdef bytes message

        self.running.clear()
        if self.pipe is not None:
            while True:
                message = await self.pipe.read()
                asyncio.ensure_future(self.on_receive(message))


cdef class Dispatcher(AsyncHandler):
    """
    Only created when more than 1 process.

    What dispatcher do is redirect the message to the destination process
    according to the distributor, and then return back the result to the original process.
    """

    cdef ClusterInfo cluster_info
    cdef int index
    cdef list pipes
    cdef object distributor  # has to be object, not Distributor
    cdef object remote_handler
    cdef dict handlers

    def __init__(self, ClusterInfo cluster_info, list pipes, Distributor distributor=None, parallel=None):
        super().__init__()
        AsyncHandler.__init__(self)

        self.cluster_info = cluster_info
        self.index = -1
        self.pipes = pipes  # type: list
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
        if actor_ref.address is None or actor_ref.address == self.cluster_info.address \
                or actor_ref.address == self.cluster_info.advertise_address:
            return False
        return True

    async def _send(self, ActorRef actor_ref, object message, bint wait_response=True,
                    object callback=None):
        cdef int to_index
        cdef list msg
        cdef bytes message_id
        cdef list messages

        if self._is_remote(actor_ref):
            if wait_response:
                return await self.remote_handler.send(actor_ref, message, callback=callback)
            else:
                return await self.remote_handler.tell(actor_ref, message, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)
        try:
            if wait_response:
                msg = pack_send_message(self.index, to_index, actor_ref, message)
            else:
                msg = pack_tell_message(self.index, to_index, actor_ref, message)
            message_id, messages = msg[0], msg[1:]
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle message: {0}'.format(message))

        await self.pipes[to_index].write(*messages)
        return await self.submit(message_id)

    async def send(self, ActorRef actor_ref, object message, object callback=None):
        return await self._send(actor_ref, message, wait_response=True, callback=callback)

    async def tell(self, ActorRef actor_ref, object message, object delay=None, object callback=None):
        if delay is not None:
            async def delay_tell():
                await asyncio.sleep(delay)
                return await self._send(actor_ref, message, wait_response=False, callback=callback)

            asyncio.ensure_future(delay_tell())
        else:
            return await self._send(actor_ref, message, wait_response=False, callback=callback)

    async def create_actor(self, str address, object uid, object actor_cls, *args, **kwargs):
        cdef object actor_id
        cdef ActorRef actor_ref
        cdef object callback
        cdef bytes message_id
        cdef bytearray message
        cdef int to_index

        actor_id = uid or new_actor_id()
        actor_ref = ActorRef(address, actor_id)

        if self._is_remote(actor_ref):
            return await self.remote_handler.create_actor(address, uid, actor_cls, *args, **kwargs)

        callback = kwargs.pop('callback', None)

        to_index = self.distributor.distribute(actor_id)
        try:
            message_id, message = pack_create_actor_message(
                self.index, to_index, actor_ref, actor_cls, args, kwargs)
        except (AttributeError, pickle.PickleError):
            raise pickle.PicklingError('Unable to pickle {0}(*{1}, **{2})'.format(actor_cls, args, kwargs))

        await self.pipes[to_index].write(message)
        return await self.submit(message_id)

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        cdef int to_index
        cdef bytes message_id
        cdef bytearray message

        if self._is_remote(actor_ref):
            return await self.remote_handler.destroy_actor(actor_ref, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)

        message_id, message = pack_destroy_actor_message(
            self.index, to_index, actor_ref)

        await self.pipes[to_index].write(message)
        return await self.submit(message_id)

    async def has_actor(self, ActorRef actor_ref, object callback=None):
        cdef int to_index
        cdef bytes message_id
        cdef bytearray message
        cdef object future

        if self._is_remote(actor_ref):
            return await self.remote_handler.has_actor(actor_ref, callback=callback)

        to_index = self.distributor.distribute(actor_ref.uid)

        message_id, message = pack_has_actor_message(
            self.index, to_index, actor_ref)

        await self.pipes[to_index].write(message)
        return await self.submit(message_id)

    async def _on_receive_action(self, bytes binary):
        cdef int from_index
        cdef int to_index
        cdef bytes message_id

        from_index, to_index = get_index(binary, self.distributor.distribute)
        message_id = unpack_message_id(binary)

        if from_index == REMOTE_FROM_INDEX:
            # sent from remote, redirect to process, and wait for result
            await self.pipes[to_index].write(binary)
            return await self.wait(message_id)
        else:
            # sent from other process, just redirect
            await self.pipes[to_index].write(binary)

    async def _on_receive_result(self, bytes binary):
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
            await self.pipes[to_index].write(binary)

    async def _on_receive_error(self, binary):
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
            await self.pipes[to_index].write(binary)

    async def on_receive(self, bytes binary):
        cdef int message_type

        message_type = unpack_message_type_value(binary)
        return await self.handlers[message_type](binary)

    async def _check_pipe(self, int idx):
        cdef bytes message
        while True:
            try:
                message = await self.pipes[idx].read()
            except EOFError:
                # broken pipe
                return

            asyncio.ensure_future(self.on_receive(message))

    async def run(self):
        cdef int idx
        cdef list pipe_checkers

        pipe_checkers = []
        for idx, pipe in enumerate(self.pipes):
            task = asyncio.ensure_future(self._check_pipe(idx))
            pipe_checkers.append(task)
        await asyncio.wait([c for c in pipe_checkers], return_when=asyncio.FIRST_EXCEPTION)


cdef class ActorServer(object):
    cdef object _host
    cdef object _port
    cdef object _dispatcher
    cdef object _server
    cdef dict _aio_pairs
    cdef bint _multi_process

    def __init__(self, host, port, dispatcher, multi_process=True):
        self._host = host
        self._port = port
        self._dispatcher = dispatcher
        self._multi_process = multi_process
        self._aio_pairs = dict()
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self.handle_client, self._host, self._port)

    async def stop(self):
        close_waiters = []
        if self._server is not None:
            self._server.close()
            close_waiters.append(self._server.wait_closed())
            close_waiters.extend([p.close() for p in self._aio_pairs.values()])
            await asyncio.wait(close_waiters, return_when=asyncio.ALL_COMPLETED)
            self._server = None

    async def on_receive(self, bytes binary):
        if not self._multi_process:
            return await self._dispatcher.on_receive(binary, callback=_no_callback)
        else:
            return await self._dispatcher.on_receive(binary)

    async def handle_client(self, reader, writer):
        cdef bytes binary
        cdef object p
        cdef object aio_pair
        cdef object fileno

        aio_pair = AsyncIOPair(reader_writer=(reader, writer))
        fileno = aio_pair.socket_fileno
        self._aio_pairs[fileno] = aio_pair
        try:
            while True:
                try:
                    binary = await aio_pair.read()
                    p = await self.on_receive(binary)
                    await aio_pair.write(p)
                except (socket.error, struct.error):
                    break
        finally:
            await aio_pair.close()
            self._aio_pairs.pop(fileno)


async def start_actor_server(ClusterInfo cluster_info, object sender):
    cdef str address
    cdef int port
    cdef bint multi_process
    cdef object s

    address, port = cluster_info.location, cluster_info.port
    if address is None or port is None:
        raise ValueError('Both address and port should be provided')
    multi_process = cluster_info.n_process > 1

    s = ActorServer(address, port, sender, multi_process)
    await s.start()
    return s


async def start_local_pool(int index, ClusterInfo cluster_info,
                           object pipe=None, Distributor distributor=None,
                           object parallel=None, bint join=False):
    # new process will pickle the numpy RandomState, we seed the random one
    import numpy as np
    np.random.seed()

    if pipe is not None:
        pipe.reset_reader_writer()

    # all these work in a single process
    # we start a local pool to handle messages
    # and a communicator to do the redirection of messages
    local_pool = LocalActorPool(cluster_info.advertise_address or cluster_info.address, index)
    comm = Communicator(local_pool, cluster_info, pipe, distributor, parallel)
    local_pool.set_comm(comm)
    if join:
        await comm.run()
    else:
        asyncio.ensure_future(comm.run())
        return comm


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
        self._stopped = asyncio.locks.Event()
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

    async def create_actor(self, object actor_cls, *args, **kwargs):
        cdef bint wait
        cdef object address
        cdef object uid
        cdef ActorRef actor_ref

        self._check_started()

        address = kwargs.pop('address', None)
        uid = kwargs.pop('uid', None)

        actor_ref = await self._dispatcher.create_actor(address, uid,
                                                        actor_cls, *args, **kwargs)
        if address:
            actor_ref.address = address
        actor_ref.ctx = ActorContext(self._dispatcher)
        return actor_ref

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        self._check_started()
        return await self._dispatcher.destroy_actor(actor_ref, callback=callback)

    async def has_actor(self, ActorRef actor_ref, object callback=None):
        self._check_started()
        return await self._dispatcher.has_actor(actor_ref, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = create_actor_ref(*args, **kwargs)
        ref.ctx = ActorContext(self._dispatcher)
        return ref

    @property
    def processes(self):
        return self._processes

    def _start_process(self, idx, create_pipe=True):
        if create_pipe:
            parent_in, child_out = os.pipe()
            child_in, parent_out = os.pipe()
            parent_pair = AsyncIOPair(pipe_fds=(parent_in, parent_out))
            child_pair = AsyncIOPair(pipe_fds=(child_in, child_out))
        else:
            parent_pair = self._pool_pipes[idx]
            child_pair = self._comm_pipes[idx]

        p = aiomultiprocess.Process(target=start_local_pool,
                                    args=(idx, self.cluster_info, child_pair, self.distributor),
                                    kwargs={'parallel': self._parallel, 'join': True}, daemon=True)
        p.start()
        return p, child_pair, parent_pair

    async def restart_process(self, int idx):
        if self._processes[idx].is_alive():
            self._processes[idx].terminate()
        self._processes[idx], self._comm_pipes[idx], self._pool_pipes[idx] = self._start_process(idx, False)

    async def run(self):
        if self._started:
            return

        if not self._multi_process:
            # only start local pool
            self._dispatcher = await start_local_pool(0, self.cluster_info, distributor=self.distributor,
                                                      parallel=self._parallel)
        else:
            self._processes, self._comm_pipes, self._pool_pipes = [list(tp) for tp in zip(
                *(self._start_process(idx) for idx in range(self.cluster_info.n_process))
            )]

            async def stop_func():
                for process in self._processes:
                    process.terminate()
                for idx, p in enumerate(self._comm_pipes):
                    if p is not None:
                        try:
                            await p.close()
                        except OSError:
                            pass
                        self._comm_pipes[idx] = None
                for idx, p in enumerate(self._pool_pipes):
                    if p is not None:
                        try:
                            await p.close()
                        except OSError:
                            pass
                        self._pool_pipes[idx] = None

            self._stop_funcs.append(stop_func)

            # start dispatcher
            self._dispatcher = Dispatcher(self.cluster_info, self._pool_pipes,
                                          self.distributor)
            asyncio.ensure_future(self._dispatcher.run())

        if not self.cluster_info.standalone:
            # start stream server to handle remote message
            self._server = await start_actor_server(self.cluster_info, self._dispatcher)

            async def close():
                await self._server.stop()

            self._stop_funcs.append(close)

        self._started = True

    async def join(self, timeout=None):
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self._stopped.is_set()

    async def stop(self):
        try:
            if self._stop_funcs:
                await asyncio.wait([stop_func() for stop_func in self._stop_funcs],
                                    return_when=asyncio.ALL_COMPLETED)
        finally:
            self._stopped.set()
            self._started = False

    async def __aenter__(self):
        await self.run()
        return self

    async def __aexit__(self, *_):
        await self.stop()


cdef class ActorClient:
    cdef object remote_handler

    def __init__(self, parallel=None):
        self.remote_handler = ActorRemoteHelper(parallel)

    async def create_actor(self, object actor_cls, *args, **kwargs):
        cdef object address
        cdef object uid

        if 'address' not in kwargs or kwargs.get('address') is None:
            raise ValueError('address must be provided')
        address = kwargs.pop('address')
        uid = kwargs.pop('uid', new_actor_id())
        return await self.remote_handler.create_actor(address, uid, actor_cls, *args, **kwargs)

    async def has_actor(self, ActorRef actor_ref, object callback=None):
        return await self.remote_handler.has_actor(actor_ref, callback=callback)

    async def destroy_actor(self, ActorRef actor_ref, object callback=None):
        return await self.remote_handler.destroy_actor(actor_ref, callback=callback)

    def actor_ref(self, *args, **kwargs):
        cdef ActorRef ref

        ref = self.remote_handler.actor_ref(*args, **kwargs)
        if ref.address is None:
            raise ValueError('address must be provided')
        return ref

    @staticmethod
    def popen(*args, **kwargs):
        new_args = args
        if args:
            new_args = tuple(args[0]) + args[1:]
        return asyncio.create_subprocess_exec(*new_args, **kwargs)

    @staticmethod
    def threadpool(size):
        return AioThreadPool(size)
