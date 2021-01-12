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

import itertools
import random

import gevent.lock
import gevent.socket

cpdef int REMOTE_MAX_CONNECTION = 200  # most connections


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


cdef object Connections_global_lock = gevent.lock.Semaphore()
cdef int Connections_addrs = 0


cdef class Connections(object):
    cdef object lock
    cdef object address
    cdef dict conn_locks

    def __init__(self, address):
        global Connections_addrs

        if isinstance(address, str):
            self.address = address.split(':', 1)
        else:
            self.address = address

        with Connections_global_lock:
            Connections_addrs += 1

        self.lock = gevent.lock.Semaphore()
        self.conn_locks = dict()

    @property
    def conn(self):
        return [conn_lock[0] for conn_lock in self.conn_locks.values()]

    def _connect(self, conn, lock):
        return Connection(conn, lock)

    def got_broken_pipe(self, fd):
        del self.conn_locks[fd]

    def connect(self):
        cdef int maxlen
        cdef object conn
        cdef object lock

        with self.lock:
            for conn, lock in self.conn_locks.values():
                # try to reuse the connections before
                locked = lock.acquire(blocking=False)
                if not locked:
                    continue
                return self._connect(conn, lock)

            maxlen = max(REMOTE_MAX_CONNECTION // Connections_addrs, 1)

            if len(self.conn_locks) < maxlen:
                # create a new connection
                lock = gevent.lock.Semaphore()
                lock.acquire()
                conn = gevent.socket.create_connection(self.address)
                self.conn_locks[conn.fileno()] = (conn, lock)
                return self._connect(conn, lock)

            def close(c, lk):
                with lk:
                    c.close()

            ps = [gevent.spawn(close, c, l) for c, l in
                  itertools.islice(self.conn_locks.values(), maxlen, len(self.conn_locks))]

            i = random.randint(0, maxlen - 1)
            fd = next(itertools.islice(self.conn_locks.keys(), i, i + 1))
            conn, lock = self.conn_locks[fd]
            lock.acquire()

            # wait for conn finished
            gevent.joinall(ps)
            self.conn_locks = dict(itertools.islice(self.conn_locks.items(), maxlen))

            return self._connect(conn, lock)

    def __del__(self):
        for c, _ in self.conn_locks.values():
            try:
                c.close()
            except:  # pragma: no cover
                pass


cdef class MarsRemoteHandler:
    pass
