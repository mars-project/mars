#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import pickle
import itertools
import hashlib
import uuid
import time
import unittest
import sys

import gevent

from mars.compat import six, BrokenPipeError
from mars.actors import create_actor_pool as new_actor_pool, Actor, ActorNotExist, Distributor, new_client
from mars.actors.pool.gevent_pool import Dispatcher, Connections
from mars.utils import to_binary


DEFAULT_PORT = 12345


def create_actor_pool(*args, **kwargs):
    import gevent.socket

    address = kwargs.pop('address', None)
    if not address:
        return new_actor_pool(*args, **kwargs)

    if isinstance(address, six.string_types):
        port = int(address.rsplit(':', 1)[1])
    else:
        port = DEFAULT_PORT
    it = itertools.count(port)

    auto_port = kwargs.pop('auto_port', True)
    while True:
        try:
            address = '127.0.0.1:{0}'.format(next(it))
            return new_actor_pool(address, *args, **kwargs)
        except gevent.socket.error:
            if auto_port:
                continue
            raise


class DummyActor(Actor):
    def __init__(self, value):
        super(DummyActor, self).__init__()

        if value < 0:
            raise ValueError('value < 0')
        self.value = value

    def on_receive(self, message):  # noqa: C901
        if message[0] == 'add':
            if not isinstance(message[1], six.integer_types):
                raise TypeError('add number must be int')
            self.value += message[1]
            return self.value
        elif message[0] == 'add_ret':
            return self.value + message[1]
        elif message[0] == 'create':
            kw = message[2] if len(message) > 2 else dict()
            return self.ctx.create_actor(*message[1], **kw)
        elif message[0] == 'create_async':
            kw = message[2] if len(message) > 2 else dict()
            kw['wait'] = False
            future = self.ctx.create_actor(*message[1], **kw)
            return future.result()
        elif message[0] == 'create_ignore':
            kw = message[2] if len(message) > 2 else dict()
            try:
                return self.ctx.create_actor(*message[1], **kw)
            except ValueError:
                pass
        elif message[0] == 'create_send':
            ref = self.ctx.create_actor(*message[1], **message[2])
            return ref.send(message[3])
        elif message[0] == 'delete':
            return self.ctx.destroy_actor(message[1])
        elif message[0] == 'delete_async':
            future = self.ctx.destroy_actor(message[1], wait=False)
            return future.result()
        elif message[0] == 'has':
            return self.ctx.has_actor(message[1])
        elif message[0] == 'has_async':
            future = self.ctx.has_actor(message[1], wait=False)
            return future.result()
        elif message[0] == 'send':
            actor_ref = self.ctx.actor_ref(message[1])
            return actor_ref.send(message[2:])
        elif message[0] == 'send_async':
            actor_ref = self.ctx.actor_ref(message[1])
            future = actor_ref.send(message[2:], wait=False)
            return future.result()
        elif message[0] == 'tell':
            actor_ref = self.ctx.actor_ref(message[1])
            return actor_ref.tell(message[2:])
        elif message[0] == 'tell_async':
            actor_ref = self.ctx.actor_ref(message[1])
            future = actor_ref.tell(message[2:], wait=False)
            return future.result()
        elif message[0] == 'tell_delay':
            actor_ref = self.ctx.actor_ref(message[1])
            return actor_ref.tell(message[2:-1], delay=message[-1])
        elif message[0] == 'send_unpickled':
            actor_ref = self.ctx.actor_ref(message[1])
            return actor_ref.send(lambda x: x)
        elif message[0] == 'create_unpickled':
            return self.ctx.create_actor(DummyActor, lambda x: x, uid='admin-5')
        elif message[0] == 'index':
            return self.ctx.index
        elif message[0] == 'ref':
            return self.ref()
        elif message[0] == 'destroy':
            self.ref().destroy()
        elif message[0] == 'destroy_async':
            future = self.ref().destroy(wait=False)
            return future.result()
        else:
            return self.value


class DummyDistributor(Distributor):
    def distribute(self, uid):
        if str(uid).startswith('admin-'):
            return 0
        else:
            return 1


class EmptyActor(Actor):
    def on_receive(self, message):
        # do nothing
        pass


class EventActor(Actor):
    def post_create(self):
        assert 'sth' == self.ref().send('sth')

    def pre_destroy(self):
        assert 'sth2' == self.ref().send('sth2')

    def on_receive(self, message):
        return message


class AdminDistributor(Distributor):
    def distribute(self, uid):
        if self.n_process == 1:
            return 0

        if str(uid).startswith('admin-'):
            return 0
        if isinstance(uid, uuid.UUID):
            return uid.int % (self.n_process - 1) + 1

        return int(hashlib.sha1(to_binary(uid)).hexdigest(), 16) % (self.n_process - 1) + 1


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def setUp(self):
        self.exceptions = gevent.hub.Hub.NOT_ERROR
        gevent.hub.Hub.NOT_ERROR = (Exception,)

    def tearDown(self):
        gevent.hub.Hub.NOT_ERROR = self.exceptions

    def testSimpleLocalActorPool(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            actor_ref = pool.create_actor(DummyActor, 100)
            self.assertEqual(actor_ref.send(('add', 1)), 101)
            actor_ref.tell(('add', 1))

            res = actor_ref.send(('get',))
            self.assertEqual(res, 102)

            ref2 = actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            self.assertEqual(pool.actor_ref(uid=actor_ref.uid).send(('add', 2)), 104)

    def testLocalPostCreatePreDestroy(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            actor_ref = pool.create_actor(EventActor)
            actor_ref.destroy()

    def testLocalCreateActor(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            actor_ref = pool.create_actor(DummyActor, 1)
            self.assertIsNotNone(actor_ref._ctx)
            # create actor inside on_receive
            r = actor_ref.send(('create', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(ref.send(('add', 10)), 15)
            # create actor inside on_receive and send message
            r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)
            # create actor asynchronously
            future = pool.create_actor(DummyActor, 1, wait=False)
            self.assertIsNotNone(future.result()._ctx)
            # create actor asynchronously inside on_receive
            r = actor_ref.send(('create_async', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(ref.send(('add', 10)), 15)

    def testLocalCreateActorError(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            with self.assertRaises(ValueError):
                pool.create_actor(DummyActor, -1)
            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(ValueError):
                ref1.send(('create', (DummyActor, -2)))
            with self.assertRaises(ValueError):
                future = pool.create_actor(DummyActor, -1, wait=False)
                future.result()
            with self.assertRaises(ValueError):
                ref1.send(('create_async', (DummyActor, -2)))

    def testLocalSend(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1)
            ref2 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)
            # send message asynchronously
            future = ref1.send(('create', (DummyActor, 2)), wait=False)
            ref3 = pool.actor_ref(future.result())
            future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
            self.assertEqual(future2.result(), 5)

    def testLocalSendError(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0))
            ref2 = pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                pool.actor_ref('fake_uid').send(('add', 1))

            # test async error
            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0), wait=False).result()
            ref2 = pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                ref1.send(('send_async', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                pool.actor_ref('fake_uid').send(('add', 1), wait=False).result()

    def testLocalTell(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1)
            ref2 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(ref2.send(('get',)), 5)
            # tell message asynchronously
            ref3 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            future = ref3.tell(('add', 3), wait=False)
            self.assertIsNone(future.result())
            self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
            self.assertEqual(ref3.send(('get',)), 8)

            ref1.send(('tell_delay', ref2, 'add', 4, .5))  # delay 0.5 secs
            self.assertEqual(ref2.send(('get',)), 5)
            pool.sleep(0.5)
            self.assertEqual(ref2.send(('get',)), 9)

    def testLocalDestroyHasActor(self):
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1)
            self.assertTrue(pool.has_actor(ref1))

            pool.destroy_actor(ref1)
            self.assertFalse(pool.has_actor(ref1))

            ref1 = pool.create_actor(DummyActor, 1)
            ref2 = ref1.send(('create', (DummyActor, 2)))

            self.assertTrue(pool.has_actor(ref2))

            ref1.send(('delete', ref2))
            self.assertFalse(ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'))

            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(ActorNotExist):
                ref1.send(('delete', pool.actor_ref('fake_uid')))

            # test self destroy
            ref1 = pool.create_actor(DummyActor, 2)
            ref1.send(('destroy',))
            self.assertFalse(pool.has_actor(ref1))

            # test asynchronously destroy
            ref3 = pool.create_actor(DummyActor, 1)
            future = pool.destroy_actor(ref3, wait=False)
            future.result()
            self.assertFalse(pool.has_actor(ref3))

            ref3 = pool.create_actor(DummyActor, 1)
            ref4 = ref3.send(('create', (DummyActor, 2)))
            ref3.send(('delete_async', ref4))
            self.assertFalse(ref3.send(('has_async', ref4)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'), wait=False).result()

            with self.assertRaises(ActorNotExist):
                ref3.send(('delete_async', pool.actor_ref('fake_uid')))

            # test self destroy
            ref4 = pool.create_actor(DummyActor, 2)
            ref4.send(('destroy_async',))
            self.assertFalse(pool.has_actor(ref4))

    def testSimpleMultiprocessActorPool(self):
        with create_actor_pool(n_process=2, backend='gevent') as pool:
            self.assertIsInstance(pool._dispatcher, Dispatcher)

            actor_ref = pool.create_actor(DummyActor, 101)
            self.assertEqual(actor_ref.send(('add', 1)), 102)
            actor_ref.tell(('add', 1))

            res = actor_ref.send(('get',))
            self.assertEqual(res, 103)

    def testProcessPostCreatePreDestroy(self):
        with create_actor_pool(n_process=3, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            actor_ref = pool.create_actor(EventActor)
            actor_ref.destroy()

    def testProcessCreateActor(self):
        with create_actor_pool(n_process=3, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            actor_ref = pool.create_actor(DummyActor, 1, uid='admin-1')
            self.assertIsNotNone(actor_ref._ctx)
            self.assertEqual(actor_ref.send(('index',)), 0)
            # create actor inside on_receive
            r = actor_ref.send(('create', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(ref.send(('index',)), 1)
            self.assertEqual(ref.send(('add', 10)), 15)

            ref2 = actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            # create actor inside on_receive and send message
            r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

            # create actor asynchronously
            future = pool.create_actor(DummyActor, 1, wait=False)
            self.assertIsNotNone(future.result()._ctx)
            # create actor asynchronously inside on_receive
            r = actor_ref.send(('create_async', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(ref.send(('add', 10)), 15)

    def testProcessCreateActorError(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            with self.assertRaises(ValueError):
                pool.create_actor(DummyActor, -1)
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(ValueError):
                ref1.send(('create', (DummyActor, -2)))
            ref1.send(('create_ignore', (DummyActor, -3)))

            with self.assertRaises(ValueError):
                future = pool.create_actor(DummyActor, -1, wait=False)
                future.result()
            with self.assertRaises(ValueError):
                ref1.send(('create_async', (DummyActor, -2)))

    def testProcessRestart(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = pool.create_actor(DummyActor, 2, uid='user-2')
            self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)
            pool.processes[1].terminate()
            pool.restart_process(1)
            ref2 = pool.create_actor(DummyActor, 2, uid='user-2')
            self.assertEqual(ref1.send(('send', ref2, 'add', 5)), 7)

    def testProcessSend(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)
            # send message asynchronously
            future = ref1.send(('create', (DummyActor, 2)), wait=False)
            ref3 = pool.actor_ref(future.result())
            future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
            self.assertEqual(future2.result(), 5)

    def testProcessSendError(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0))
            ref2 = pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                pool.actor_ref('fake_uid').send(('add', 1))

            # test async error
            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0), wait=False).result()
            ref2 = pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                ref1.send(('send_async', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                pool.actor_ref('fake_uid').send(('add', 1), wait=False).result()

    def testProcessTell(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(ref2.send(('get',)), 5)
            # tell message asynchronously
            ref3 = pool.actor_ref(ref1.send(('create', (DummyActor, 2))))
            future = ref3.tell(('add', 3), wait=False)
            self.assertIsNone(future.result())
            self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
            self.assertEqual(ref3.send(('get',)), 8)

            ref1.send(('tell_delay', ref2, 'add', 4, 0.3))  # delay 0.5 secs
            self.assertEqual(ref2.send(('get',)), 5)
            pool.sleep(0.5)
            self.assertEqual(ref2.send(('get',)), 9)

    def testProcessDestroyHas(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            self.assertTrue(pool.has_actor(ref1))

            pool.destroy_actor(ref1)
            self.assertFalse(pool.has_actor(ref1))

            ref1 = pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = ref1.send(('create', (DummyActor, 2)))

            self.assertTrue(pool.has_actor(ref2))

            ref1.send(('delete', ref2))
            self.assertFalse(ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'))

            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(ActorNotExist):
                ref1.send(('delete', pool.actor_ref('fake_uid')))

            # test self destroy
            ref1 = pool.create_actor(DummyActor, 2)
            ref1.send(('destroy',))
            self.assertFalse(pool.has_actor(ref1))

            # test asynchronously destroy
            ref3 = pool.create_actor(DummyActor, 1)
            future = pool.destroy_actor(ref3, wait=False)
            future.result()
            self.assertFalse(pool.has_actor(ref3))

            ref3 = pool.create_actor(DummyActor, 1)
            ref4 = ref3.send(('create', (DummyActor, 2)))
            ref3.send(('delete_async', ref4))
            self.assertFalse(ref3.send(('has_async', ref4)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'), wait=False).result()

            with self.assertRaises(ActorNotExist):
                ref3.send(('delete_async', pool.actor_ref('fake_uid')))

            # test self destroy
            ref4 = pool.create_actor(DummyActor, 2)
            ref4.send(('destroy_async',))
            self.assertFalse(pool.has_actor(ref4))

    def testProcessUnpickled(self):
        with create_actor_pool(n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 1)
            with self.assertRaises(pickle.PickleError):
                ref1.send(lambda x: x)

            ref2 = pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(pickle.PickleError):
                ref1.send(('send_unpickled', ref2))

            with self.assertRaises(pickle.PickleError):
                ref1.send(('send', ref2, 'send_unpickled', ref1))

            with self.assertRaises(pickle.PickleError):
                pool.create_actor(DummyActor, lambda x: x)

            with self.assertRaises(pickle.PickleError):
                ref1.send(('create_unpickled',))

    def testRemoteConnections(self):
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool:
            addr = pool.cluster_info.address

            connections = Connections(addr)

            with connections.connect() as conn:
                default_conn = conn
                # conn's lock has not been released, a new connection will be established
                with connections.connect() as conn2:
                    self.assertIsNot(conn, conn2)

            with connections.connect() as conn3:
                self.assertIs(default_conn, conn3)

            del connections
            Connections.addrs = 0

            connections1 = Connections(addr)
            conns1 = [connections1.connect() for _ in range(100)]

            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                connections2 = Connections(addr2)
                conns2 = [connections2.connect() for _ in range(100)]

                self.assertEqual(len(connections2.conn), 100)
                [conn.release() for conn in conns2]

                conn = connections2.connect()  # do not create new connection, reuse old one
                self.assertEqual(len(connections2.conn), 100)
                self.assertFalse(conn.lock.acquire(blocking=False))
                conn.release()

                with create_actor_pool(address='127.0.0.1:12347', n_process=2, backend='gevent') as pool3:
                    addr3 = pool3.cluster_info.address

                    conns3 = Connections(addr3)

                    ps = list()
                    ps.append(gevent.spawn(connections1.connect))
                    ps.append(gevent.spawn(lambda: [c.release() for c in conns1]))
                    gevent.joinall(ps)

                    self.assertEqual(len(connections1.conn), 66)

                    del conns3

    def testRemotePostCreatePreDestroy(self):
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            actor_ref = client.create_actor(EventActor, address=addr)
            actor_ref.destroy()

    def testRemoteCreateLocalPoolActor(self):
        # client -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            actor_ref = client.create_actor(DummyActor, 1, address=addr)
            self.assertIsNotNone(actor_ref)
            self.assertEqual(actor_ref.address, addr)

            ref2 = actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            with self.assertRaises(ValueError):
                client.create_actor(DummyActor, -1, address=addr)
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ValueError):
                ref1.send(('create', (DummyActor, -2), dict(address=addr)))
            ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr)))

            # create actor inside on_receive and send message
            r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

            # create actor asynchronously
            future = client.create_actor(DummyActor, 1, wait=False, address=addr)
            self.assertIsNotNone(future.result()._ctx)
            # create actor asynchronously inside on_receive
            r = actor_ref.send(('create_async', (DummyActor, 5), dict(address=addr)))
            ref = client.actor_ref(r)
            self.assertEqual(ref.send(('add', 10)), 15)

            with self.assertRaises(ValueError):
                future = client.create_actor(DummyActor, -1, wait=False, address=addr)
                future.result()
            with self.assertRaises(ValueError):
                ref1.send(('create_async', (DummyActor, -2), dict(address=addr)))

    def testRemoteCreateProcessPoolActor(self):
        # client -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            actor_ref = client.create_actor(DummyActor, 1, address=addr)
            self.assertIsNotNone(actor_ref)
            self.assertEqual(actor_ref.address, addr)

            with self.assertRaises(ValueError):
                client.create_actor(DummyActor, -1, address=addr)
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ValueError):
                ref1.send(('create', (DummyActor, -2), dict(address=addr)))
            ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr)))

            # create actor inside on_receive and send message
            r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

            # create actor asynchronously
            future = client.create_actor(DummyActor, 1, wait=False, address=addr)
            self.assertIsNotNone(future.result()._ctx)
            # create actor asynchronously inside on_receive
            r = actor_ref.send(('create_async', (DummyActor, 5)))
            ref = client.actor_ref(r)
            self.assertEqual(ref.send(('add', 10)), 15)

    def testRemoteCreateLocalPoolToLocalPoolActor(self):
        # client -> local pool -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                actor_ref = client.create_actor(DummyActor, 1, address=addr1)
                actor_ref2 = actor_ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

                # create actor asynchronously
                future = client.create_actor(DummyActor, 1, wait=False, address=addr1)
                self.assertIsNotNone(future.result()._ctx)
                # create actor asynchronously inside on_receive
                r = actor_ref.send(('create_async', (DummyActor, 5), dict(address=addr2)))
                ref = client.actor_ref(r)
                self.assertEqual(ref.send(('add', 10)), 15)

                with self.assertRaises(ValueError):
                    future = client.create_actor(DummyActor, -1, wait=False, address=addr1)
                    future.result()
                with self.assertRaises(ValueError):
                    ref1.send(('create_async', (DummyActor, -2), dict(address=addr2)))

    def testRemoteCreateProcessPoolToProcessPoolActor(self):
        # client -> process pool -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                actor_ref = client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

                # create actor asynchronously
                future = client.create_actor(DummyActor, 1, wait=False, address=addr1)
                self.assertIsNotNone(future.result()._ctx)
                # create actor asynchronously inside on_receive
                r = actor_ref.send(('create_async', (DummyActor, 5), dict(address=addr2)))
                ref = client.actor_ref(r)
                self.assertEqual(ref.send(('add', 10)), 15)

                with self.assertRaises(ValueError):
                    future = client.create_actor(DummyActor, -1, wait=False, address=addr1)
                    future.result()
                with self.assertRaises(ValueError):
                    ref1.send(('create_async', (DummyActor, -2), dict(address=addr2)))

    def testRemoteCreateLocalPoolToProcessPoolActor(self):
        # client -> local pool -> process pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                actor_ref = client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

                # create actor asynchronously
                future = client.create_actor(DummyActor, 1, wait=False, address=addr1)
                self.assertIsNotNone(future.result()._ctx)
                # create actor asynchronously inside on_receive
                r = actor_ref.send(('create_async', (DummyActor, 5), dict(address=addr2)))
                ref = client.actor_ref(r)
                self.assertEqual(ref.send(('add', 10)), 15)

                with self.assertRaises(ValueError):
                    future = client.create_actor(DummyActor, -1, wait=False, address=addr1)
                    future.result()
                with self.assertRaises(ValueError):
                    ref1.send(('create_async', (DummyActor, -2), dict(address=addr2)))

    def testRemoteCreateProcessPoolToLocalPoolActor(self):
        # client -> process pool -> local pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                actor_ref = client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

                # create actor asynchronously
                future = client.create_actor(DummyActor, 1, wait=False, address=addr1)
                self.assertIsNotNone(future.result()._ctx)
                # create actor asynchronously inside on_receive
                r = actor_ref.send(('create_async', (DummyActor, 5), dict(address=addr2)))
                ref = client.actor_ref(r)
                self.assertEqual(ref.send(('add', 10)), 15)

                with self.assertRaises(ValueError):
                    future = client.create_actor(DummyActor, -1, wait=False, address=addr1)
                    future.result()
                with self.assertRaises(ValueError):
                    ref1.send(('create_async', (DummyActor, -2), dict(address=addr2)))

    def testRemoteSendLocalPool(self):
        # client -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0))
            ref2 = client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                client.actor_ref('fake_uid', address=addr).send(('add', 1))

            # send message asynchronously
            future = ref1.send(('create', (DummyActor, 2)), wait=False)
            ref3 = client.actor_ref(future.result())
            future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
            self.assertEqual(future2.result(), 5)

            # test async error
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0), wait=False).result()
            ref2 = client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('send_async', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                client.actor_ref('fake_uid', address=addr).send(('add', 1), wait=False).result()

    def testRemoteSendProcessPool(self):
        # client -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0))
            ref2 = client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                client.actor_ref('fake_uid', address=addr).send(('add', 1))

            # send message asynchronously
            future = ref1.send(('create', (DummyActor, 2)), wait=False)
            ref3 = client.actor_ref(future.result())
            future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
            self.assertEqual(future2.result(), 5)

            # test async error
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('add', 1.0), wait=False).result()
            ref2 = client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                ref1.send(('send_async', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                client.actor_ref('fake_uid', address=addr).send(('add', 1), wait=False).result()

    def testRemoteSendLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0))
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1))

                # send message asynchronously
                future = ref1.send(('create', (DummyActor, 2), dict(address=addr2)), wait=False)
                ref3 = client.actor_ref(future.result())
                future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
                self.assertEqual(future2.result(), 5)

                # test async error
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0), wait=False).result()
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send_async', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1), wait=False).result()

    def testRemoteSendProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0))
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1))

                # send message asynchronously
                future = ref1.send(('create', (DummyActor, 2), dict(address=addr2)), wait=False)
                ref3 = client.actor_ref(future.result())
                future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
                self.assertEqual(future2.result(), 5)

                # test async error
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0), wait=False).result()
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send_async', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1), wait=False).result()

    def testRemoteSendLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0))
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1))

                # send message asynchronously
                future = ref1.send(('create', (DummyActor, 2), dict(address=addr2)), wait=False)
                ref3 = client.actor_ref(future.result())
                future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
                self.assertEqual(future2.result(), 5)

                # test async error
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0), wait=False).result()
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send_async', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1), wait=False).result()

    def testRemoteSendProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0))
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1))

                # send message asynchronously
                future = ref1.send(('create', (DummyActor, 2), dict(address=addr2)), wait=False)
                ref3 = client.actor_ref(future.result())
                future2 = ref1.send(('send_async', ref3, 'add', 3), wait=False)
                self.assertEqual(future2.result(), 5)

                # test async error
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    ref1.send(('add', 1.0), wait=False).result()
                ref2 = client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    ref1.send(('send_async', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    client.actor_ref('fake_uid', address=addr1).send(('add', 1), wait=False).result()

    def testRemoteTellLocalPool(self):
        # client -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(ref2.send(('get',)), 5)

            ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
            self.assertEqual(ref2.send(('get',)), 5)
            pool.sleep(.5)
            self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteTellProcessPool(self):
        # client -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')
            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(ref2.send(('get',)), 5)

            # tell message asynchronously
            ref3 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            future = ref3.tell(('add', 3), wait=False)
            self.assertIsNone(future.result())
            self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
            self.assertEqual(ref3.send(('get',)), 8)

            ref1.send(('tell_delay', ref2, 'add', 4, 0.3))  # delay 0.3 secs
            self.assertEqual(ref2.send(('get',)), 5)
            client.sleep(.5)
            self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteTellLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(ref2.send(('get',)), 5)

                # tell message asynchronously
                ref3 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                future = ref3.tell(('add', 3), wait=False)
                self.assertIsNone(future.result())
                self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
                self.assertEqual(ref3.send(('get',)), 8)

                ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(ref2.send(('get',)), 5)
                client.sleep(.5)
                self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteTellProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(ref2.send(('get',)), 5)

                # tell message asynchronously
                ref3 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                future = ref3.tell(('add', 3), wait=False)
                self.assertIsNone(future.result())
                self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
                self.assertEqual(ref3.send(('get',)), 8)

                ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(ref2.send(('get',)), 5)
                client.sleep(.5)
                self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteTellLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(ref2.send(('get',)), 5)

                # tell message asynchronously
                ref3 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                future = ref3.tell(('add', 3), wait=False)
                self.assertIsNone(future.result())
                self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
                self.assertEqual(ref3.send(('get',)), 8)

                ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(ref2.send(('get',)), 5)
                client.sleep(.5)
                self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteTellProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')
                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(ref2.send(('get',)), 5)

                # tell message asynchronously
                ref3 = client.actor_ref(ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                future = ref3.tell(('add', 3), wait=False)
                self.assertIsNone(future.result())
                self.assertIsNone(ref1.send(('tell_async', ref3, 'add', 3)))
                self.assertEqual(ref3.send(('get',)), 8)

                ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(ref2.send(('get',)), 5)
                client.sleep(.5)
                self.assertEqual(ref2.send(('get',)), 9)

    def testRemoteDestroyHasLocalPool(self):
        # client -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            self.assertTrue(client.has_actor(ref1))

            client.destroy_actor(ref1)
            self.assertFalse(client.has_actor(ref1))

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr)))

            self.assertTrue(client.has_actor(ref2))

            ref1.send(('delete', ref2))
            self.assertFalse(ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                client.destroy_actor(client.actor_ref('fake_uid', address=addr))

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ActorNotExist):
                ref1.send(('delete', client.actor_ref('fake_uid', address=addr)))

            # test self destroy
            ref1 = client.create_actor(DummyActor, 2, address=addr)
            ref1.send(('destroy',))
            self.assertFalse(client.has_actor(ref1))

            # test asynchronously destroy
            ref3 = client.create_actor(DummyActor, 1, address=addr)
            future = client.destroy_actor(ref3, wait=False)
            future.result()
            self.assertFalse(pool.has_actor(ref3))

            ref3 = client.create_actor(DummyActor, 1, address=addr)
            ref4 = ref3.send(('create', (DummyActor, 2)))
            ref3.send(('delete_async', ref4))
            self.assertFalse(ref3.send(('has_async', ref4)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'), wait=False).result()

            with self.assertRaises(ActorNotExist):
                ref3.send(('delete_async', pool.actor_ref('fake_uid')))

            # test self destroy
            ref4 = pool.create_actor(DummyActor, 2)
            ref4.send(('destroy_async',))
            self.assertFalse(pool.has_actor(ref4))

    def testRemoteDestroyHasProcessPool(self):
        # client -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            self.assertTrue(client.has_actor(ref1))

            client.destroy_actor(ref1)
            self.assertFalse(client.has_actor(ref1))

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr)))

            self.assertTrue(client.has_actor(ref2))

            ref1.send(('delete', ref2))
            self.assertFalse(ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                client.destroy_actor(client.actor_ref('fake_uid', address=addr))

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ActorNotExist):
                ref1.send(('delete', client.actor_ref('fake_uid', address=addr)))

            # test asynchronously destroy
            ref3 = client.create_actor(DummyActor, 1, address=addr)
            future = client.destroy_actor(ref3, wait=False)
            future.result()
            self.assertFalse(pool.has_actor(ref3))

            ref3 = client.create_actor(DummyActor, 1, address=addr)
            ref4 = ref3.send(('create', (DummyActor, 2)))
            ref3.send(('delete_async', ref4))
            self.assertFalse(ref3.send(('has_async', ref4)))

            with self.assertRaises(ActorNotExist):
                pool.destroy_actor(pool.actor_ref('fake_uid'), wait=False).result()

            with self.assertRaises(ActorNotExist):
                ref3.send(('delete_async', pool.actor_ref('fake_uid')))

            # test self destroy
            ref4 = pool.create_actor(DummyActor, 2)
            ref4.send(('destroy_async',))
            self.assertFalse(pool.has_actor(ref4))

    def testRemoteDestroyHasLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(client.has_actor(ref1))

                client.destroy_actor(ref1)
                self.assertFalse(client.has_actor(ref1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                self.assertTrue(client.has_actor(ref2))

                ref1.send(('delete', ref2))
                self.assertFalse(ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test asynchronously destroy
                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                future = client.destroy_actor(ref3, wait=False)
                future.result()
                self.assertFalse(client.has_actor(ref3))

                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                ref4 = ref3.send(('create', (DummyActor, 2), dict(address=addr2)))
                ref3.send(('delete_async', ref4))
                self.assertFalse(ref3.send(('has_async', ref4)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1), wait=False).result()

                with self.assertRaises(ActorNotExist):
                    ref3.send(('delete_async', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref4 = client.create_actor(DummyActor, 2, address=addr2)
                ref4.send(('destroy_async',))
                self.assertFalse(client.has_actor(ref4))

    def testRemoteDestroyHasProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(client.has_actor(ref1))

                client.destroy_actor(ref1)
                self.assertFalse(client.has_actor(ref1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                self.assertTrue(client.has_actor(ref2))

                ref1.send(('delete', ref2))
                self.assertFalse(ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test asynchronously destroy
                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                future = client.destroy_actor(ref3, wait=False)
                future.result()
                self.assertFalse(client.has_actor(ref3))

                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                ref4 = ref3.send(('create', (DummyActor, 2), dict(address=addr2)))
                ref3.send(('delete_async', ref4))
                self.assertFalse(ref3.send(('has_async', ref4)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1), wait=False).result()

                with self.assertRaises(ActorNotExist):
                    ref3.send(('delete_async', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref4 = client.create_actor(DummyActor, 2, address=addr2)
                ref4.send(('destroy_async',))
                self.assertFalse(client.has_actor(ref4))

    def testRemoteDestroyHasLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        with create_actor_pool(address=True, n_process=1, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=2, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(client.has_actor(ref1))

                client.destroy_actor(ref1)
                self.assertFalse(client.has_actor(ref1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                self.assertTrue(client.has_actor(ref2))

                ref1.send(('delete', ref2))
                self.assertFalse(ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test asynchronously destroy
                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                future = client.destroy_actor(ref3, wait=False)
                future.result()
                self.assertFalse(client.has_actor(ref3))

                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                ref4 = ref3.send(('create', (DummyActor, 2), dict(address=addr2)))
                ref3.send(('delete_async', ref4))
                self.assertFalse(ref3.send(('has_async', ref4)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1), wait=False).result()

                with self.assertRaises(ActorNotExist):
                    ref3.send(('delete_async', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref4 = client.create_actor(DummyActor, 2, address=addr2)
                ref4.send(('destroy_async',))
                self.assertFalse(client.has_actor(ref4))

    def testRemoteDestroyHasProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        with create_actor_pool(address=True, n_process=2, backend='gevent') as pool1:
            addr1 = pool1.cluster_info.address
            with create_actor_pool(address='127.0.0.1:12346', n_process=1, backend='gevent') as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client(backend='gevent')

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(client.has_actor(ref1))

                client.destroy_actor(ref1)
                self.assertFalse(client.has_actor(ref1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                ref2 = ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                self.assertTrue(client.has_actor(ref2))

                ref1.send(('delete', ref2))
                self.assertFalse(ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test asynchronously destroy
                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                future = client.destroy_actor(ref3, wait=False)
                future.result()
                self.assertFalse(client.has_actor(ref3))

                ref3 = client.create_actor(DummyActor, 1, address=addr1)
                ref4 = ref3.send(('create', (DummyActor, 2), dict(address=addr2)))
                ref3.send(('delete_async', ref4))
                self.assertFalse(ref3.send(('has_async', ref4)))

                with self.assertRaises(ActorNotExist):
                    client.destroy_actor(client.actor_ref('fake_uid', address=addr1), wait=False).result()

                with self.assertRaises(ActorNotExist):
                    ref3.send(('delete_async', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref4 = client.create_actor(DummyActor, 2, address=addr2)
                ref4.send(('destroy_async',))
                self.assertFalse(client.has_actor(ref4))

    def testRemoteProcessPoolUnpickled(self):
        with create_actor_pool(address=True, n_process=2, distributor=DummyDistributor(2),
                               backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')

            ref1 = client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(pickle.PickleError):
                ref1.send(lambda x: x)

            ref2 = client.create_actor(DummyActor, 1, address=addr, uid='admin-1')
            with self.assertRaises(pickle.PickleError):
                ref1.send(('send_unpickled', ref2))

            with self.assertRaises(pickle.PickleError):
                ref1.send(('send', ref2, 'send_unpickled', ref1))

            with self.assertRaises(pickle.PickleError):
                client.create_actor(DummyActor, lambda x: x, address=addr)

            with self.assertRaises(pickle.PickleError):
                ref1.send(('create_unpickled',))

    def testRemoteEmpty(self):
        with create_actor_pool(address=True, n_process=2, distributor=AdminDistributor(2),
                               backend='gevent') as pool:
            addr = pool.cluster_info.address

            client = new_client(backend='gevent')

            ref = client.create_actor(EmptyActor, address=addr)
            self.assertIsNone(ref.send(None))

    def testPoolJoin(self):
        with create_actor_pool(address=True, n_process=2, distributor=AdminDistributor(2),
                               backend='gevent') as pool:
            start = time.time()
            pool.join(0.2)
            self.assertGreaterEqual(time.time() - start, 0.2)

            gevent.spawn_later(0.2, lambda: pool.stop())

            start = time.time()
            pool.join()
            self.assertGreaterEqual(time.time() - start, 0.2)

    def testConcurrentSend(self):
        with create_actor_pool(address=True, n_process=4, distributor=AdminDistributor(4),
                               backend='gevent') as pool:
            ref1 = pool.create_actor(DummyActor, 0)

            def ref_send(ref, rg):
                p = []
                for i in range(*rg):
                    p.append(gevent.spawn(ref.send, ('send', ref1, 'add_ret', i)))
                self.assertEqual([f.get() for f in p], list(range(*rg)))

            n_ref = 20

            refs = [pool.create_actor(DummyActor, 0) for _ in range(n_ref)]

            ps = []
            for i in range(n_ref):
                r = (i * 100, (i + 1) * 100)
                refx = refs[i]
                ps.append(gevent.spawn(ref_send, refx, r))

            [p.get() for p in ps]

    def testRemoteBrokenPipe(self):
        pool1 = create_actor_pool(address=True, n_process=1, backend='gevent')
        addr = pool1.cluster_info.address

        try:
            client = new_client(parallel=1, backend='gevent')
            # make client create a connection
            client.create_actor(DummyActor, 10, address=addr)

            # stop
            pool1.stop()

            # the connection is broken
            with self.assertRaises(BrokenPipeError):
                client.create_actor(DummyActor, 10, address=addr)

            pool1 = create_actor_pool(address=addr, n_process=1, backend='gevent', auto_port=False)

            # create a new pool, so the actor can be created
            client.create_actor(DummyActor, 10, address=addr)
        finally:
            pool1.stop()
