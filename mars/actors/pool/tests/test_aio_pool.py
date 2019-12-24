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
import pickle
import itertools
import uuid
import socket
import time
import unittest
import sys

from mars.actors import create_actor_pool as new_actor_pool, Actor, FunctionActor, \
    ActorPoolNotStarted, ActorAlreadyExist, ActorNotExist, Distributor, new_client, \
    register_actor_implementation, unregister_actor_implementation
from mars.actors.pool.aio_pool import Dispatcher, Connections
from mars.lib.mmh3 import hash as mmh_hash
from mars.tests.core import aio_case
from mars.utils import to_binary

import tracemalloc
tracemalloc.start()


DEFAULT_PORT = 12345


async def create_actor_pool(*args, **kwargs):
    address = kwargs.pop('address', None)
    if not address:
        return await new_actor_pool(*args, **kwargs)

    if isinstance(address, str):
        port = int(address.rsplit(':', 1)[1])
    else:
        port = DEFAULT_PORT
    it = itertools.count(port)

    auto_port = kwargs.pop('auto_port', True)
    while True:
        try:
            address = '127.0.0.1:{0}'.format(next(it))
            return await new_actor_pool(address, *args, **kwargs)
        except socket.error:
            if auto_port:
                continue
            raise


class DummyActor(Actor):
    def __init__(self, value):
        super().__init__()

        if value < 0:
            raise ValueError('value < 0')
        self.value = value

    async def on_receive(self, message):  # noqa: C901
        if message[0] == 'add':
            if not isinstance(message[1], int):
                raise TypeError('add number must be int, not %s' % type(message[1]))
            self.value += message[1]
            return self.value
        elif message[0] == 'add_ret':
            return self.value + message[1]
        elif message[0] == 'create':
            kw = message[2] if len(message) > 2 else dict()
            return await self.ctx.create_actor(*message[1], **kw)
        elif message[0] == 'create_ignore':
            kw = message[2] if len(message) > 2 else dict()
            try:
                return await self.ctx.create_actor(*message[1], **kw)
            except ValueError:
                pass
        elif message[0] == 'create_send':
            ref = await self.ctx.create_actor(*message[1], **message[2])
            return await ref.send(message[3])
        elif message[0] == 'delete':
            return await self.ctx.destroy_actor(message[1])
        elif message[0] == 'has':
            return await self.ctx.has_actor(message[1])
        elif message[0] == 'send':
            actor_ref = self.ctx.actor_ref(message[1])
            return await actor_ref.send(message[2:])
        elif message[0] == 'tell':
            actor_ref = self.ctx.actor_ref(message[1])
            return await actor_ref.tell(message[2:])
        elif message[0] == 'tell_delay':
            actor_ref = self.ctx.actor_ref(message[1])
            return await actor_ref.tell(message[2:-1], delay=message[-1])
        elif message[0] == 'send_unpickled':
            actor_ref = self.ctx.actor_ref(message[1])
            return await actor_ref.send(lambda x: x)
        elif message[0] == 'create_unpickled':
            return await self.ctx.create_actor(DummyActor, lambda x: x, uid='admin-5')
        elif message[0] == 'index':
            return self.ctx.index
        elif message[0] == 'ref':
            return self.ref()
        elif message[0] == 'destroy':
            await self.ref().destroy()
        else:
            return self.value


class DummyFunctionActor(FunctionActor):
    def __init__(self, value):
        super().__init__()
        self._val = value

    async def func(self, value):
        return value + self._val


class SurrogateFunctionActor(DummyFunctionActor):
    def __init__(self, value):
        super().__init__(value)
        self._val = value * 2

    async def func(self, value):
        return value * self._val


class DummyDistributor(Distributor):
    def distribute(self, uid):
        if str(uid).startswith('admin-'):
            return 0
        else:
            return 1


class EmptyActor(Actor):
    async def on_receive(self, message):
        # do nothing
        pass


class EventActor(Actor):
    async def post_create(self):
        assert 'sth' == await self.ref().send('sth')

    async def pre_destroy(self):
        assert 'sth2' == await self.ref().send('sth2')

    async def on_receive(self, message):
        return message


class AdminDistributor(Distributor):
    def distribute(self, uid):
        if self.n_process == 1:
            return 0

        if str(uid).startswith('admin-'):
            return 0
        if isinstance(uid, uuid.UUID):
            return uid.int % (self.n_process - 1) + 1

        return mmh_hash(to_binary(uid)) % (self.n_process - 1) + 1


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
@aio_case
class Test(unittest.TestCase):
    async def testSimpleLocalActorPool(self):
        async with await create_actor_pool(n_process=1) as pool:
            actor_ref = await pool.create_actor(DummyActor, 100)
            self.assertEqual(await actor_ref.send(('add', 1)), 101)
            await actor_ref.tell(('add', 1))

            res = await actor_ref.send(('get',))
            self.assertEqual(res, 102)

            ref2 = await actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            self.assertEqual(await pool.actor_ref(uid=actor_ref.uid).send(('add', 2)), 104)

    async def testLocalPostCreatePreDestroy(self):
        async with await create_actor_pool(n_process=1) as pool:
            actor_ref = await pool.create_actor(EventActor)
            await actor_ref.destroy()

    async def testFunctionActor(self):
        async with await create_actor_pool(n_process=1) as pool:
            actor_ref = await pool.create_actor(DummyFunctionActor, 1)
            self.assertEqual(await actor_ref.func(2), 3)
            actor_ref.destroy()

            try:
                register_actor_implementation(DummyFunctionActor, SurrogateFunctionActor)
                actor_ref = await pool.create_actor(DummyFunctionActor, 3)
                self.assertEqual(await actor_ref.func(2), 12)
                await actor_ref.destroy()
            finally:
                unregister_actor_implementation(DummyFunctionActor)

            actor_ref = await pool.create_actor(DummyFunctionActor, 2)
            self.assertEqual(await actor_ref.func(2), 4)
            await actor_ref.destroy()

    async def testLocalCreateActor(self):
        async with await create_actor_pool(n_process=1) as pool:
            actor_ref = await pool.create_actor(DummyActor, 1)
            self.assertIsNotNone(actor_ref._ctx)
            # create actor inside on_receive
            r = await actor_ref.send(('create', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(await ref.send(('add', 10)), 15)
            # create actor inside on_receive and send message
            r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

    async def testLocalCreateActorError(self):
        async with await create_actor_pool(n_process=1) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='dummy1')
            with self.assertRaises(ActorAlreadyExist):
                await pool.create_actor(DummyActor, 1, uid='dummy1')
            await pool.destroy_actor(ref1)

            with self.assertRaises(ValueError):
                await pool.create_actor(DummyActor, -1)
            ref1 = await pool.create_actor(DummyActor, 1)
            with self.assertRaises(ValueError):
                await ref1.send(('create', (DummyActor, -2)))

    async def testLocalSend(self):
        async with await create_actor_pool(n_process=1) as pool:
            ref1 = await pool.create_actor(DummyActor, 1)
            ref2 = pool.actor_ref(await ref1.send(('create', (DummyActor, 2))))
            self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

    async def testLocalSendError(self):
        async with await create_actor_pool(n_process=1) as pool:
            ref1 = await pool.create_actor(DummyActor, 1)
            with self.assertRaises(TypeError):
                await ref1.send(('add', 1.0))
            ref2 = await pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                await ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                await pool.actor_ref('fake_uid').send(('add', 1))

    async def testLocalTell(self):
        async with await create_actor_pool(n_process=1) as pool:
            ref1 = await pool.create_actor(DummyActor, 1)
            ref2 = pool.actor_ref(await ref1.send(('create', (DummyActor, 2))))
            self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(await ref2.send(('get',)), 5)

            await ref1.send(('tell_delay', ref2, 'add', 4, .5))  # delay 0.5 secs
            self.assertEqual(await ref2.send(('get',)), 5)
            await asyncio.sleep(0.5)
            self.assertEqual(await ref2.send(('get',)), 9)

            # error needed when illegal uids are passed
            with self.assertRaises(TypeError):
                await ref1.send(('tell', pool.actor_ref(set()), 'add', 3))

    async def testLocalDestroyHasActor(self):
        async with await create_actor_pool(n_process=1) as pool:
            ref1 = await pool.create_actor(DummyActor, 1)
            self.assertTrue(await pool.has_actor(ref1))

            await pool.destroy_actor(ref1)
            self.assertFalse(await pool.has_actor(ref1))

            # error needed when illegal uids are passed
            with self.assertRaises(TypeError):
                await pool.has_actor(pool.actor_ref(set()))

            ref1 = await pool.create_actor(DummyActor, 1)
            ref2 = await ref1.send(('create', (DummyActor, 2)))

            await ref1.send(('delete', ref2))
            self.assertFalse(await ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                await pool.destroy_actor(pool.actor_ref('fake_uid'))

            ref1 = await pool.create_actor(DummyActor, 1)
            with self.assertRaises(ActorNotExist):
                await ref1.send(('delete', pool.actor_ref('fake_uid')))

            # test self destroy
            ref1 = await pool.create_actor(DummyActor, 2)
            await ref1.send(('destroy',))
            self.assertFalse(await pool.has_actor(ref1))

        with self.assertRaises(ActorPoolNotStarted):
            await pool.has_actor(ref1)

    async def testSimpleMultiprocessActorPool(self):
        async with await create_actor_pool(n_process=2) as pool:
            self.assertIsInstance(pool._dispatcher, Dispatcher)

            actor_ref = await pool.create_actor(DummyActor, 101)
            self.assertEqual(await actor_ref.send(('add', 1)), 102)
            await actor_ref.tell(('add', 1))

            res = await actor_ref.send(('get',))
            self.assertEqual(res, 103)

    async def testProcessPostCreatePreDestroy(self):
        async with await create_actor_pool(
                n_process=3, distributor=DummyDistributor(2)) as pool:
            actor_ref = await pool.create_actor(EventActor)
            await actor_ref.destroy()

    async def testProcessCreateActor(self):
        async with await create_actor_pool(
                n_process=3, distributor=DummyDistributor(2)) as pool:
            actor_ref = await pool.create_actor(DummyActor, 1, uid='admin-1')
            self.assertIsNotNone(actor_ref._ctx)
            self.assertEqual(await actor_ref.send(('index',)), 0)
            # create actor inside on_receive
            r = await actor_ref.send(('create', (DummyActor, 5)))
            ref = pool.actor_ref(r)
            self.assertEqual(await ref.send(('index',)), 1)
            self.assertEqual(await ref.send(('add', 10)), 15)

            ref2 = await actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            # create actor inside on_receive and send message
            r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

    async def testProcessCreateActorError(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            with self.assertRaises(ValueError):
                await pool.create_actor(DummyActor, -1)
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(ValueError):
                await ref1.send(('create', (DummyActor, -2)))
            await ref1.send(('create_ignore', (DummyActor, -3)))

    async def testProcessRestart(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = await pool.create_actor(DummyActor, 2, uid='user-2')
            self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)
            pool.processes[1].terminate()
            await pool.restart_process(1)
            ref2 = await pool.create_actor(DummyActor, 2, uid='user-2')
            self.assertEqual(await ref1.send(('send', ref2, 'add', 5)), 7)

    async def testProcessSend(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = pool.actor_ref(await ref1.send(('create', (DummyActor, 2))))
            self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

    async def testProcessSendError(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(TypeError):
                await ref1.send(('add', 1.0))
            ref2 = await pool.create_actor(DummyActor, 2)
            with self.assertRaises(TypeError):
                await ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                await pool.actor_ref('fake_uid').send(('add', 1))

    async def testProcessTell(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            ref2 = pool.actor_ref(await ref1.send(('create', (DummyActor, 2))))
            self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(await ref2.send(('get',)), 5)

            await ref1.send(('tell_delay', ref2, 'add', 4, 0.3))  # delay 0.5 secs
            self.assertEqual(await ref2.send(('get',)), 5)
            await asyncio.sleep(0.5)
            self.assertEqual(await ref2.send(('get',)), 9)

            # error needed when illegal uids are passed
            with self.assertRaises(TypeError):
                await ref1.send(('tell', pool.actor_ref(set()), 'add', 3))

    async def testProcessDestroyHas(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            self.assertTrue(await pool.has_actor(ref1))
            await pool.destroy_actor(ref1)
            self.assertFalse(await pool.has_actor(ref1))

            # error needed when illegal uids are passed
            with self.assertRaises(TypeError):
                await pool.has_actor(pool.actor_ref(set()))

            with self.assertRaises(ActorNotExist):
                await pool.destroy_actor(pool.actor_ref('fake_uid'))

            ref1 = await pool.create_actor(DummyActor, 1)
            with self.assertRaises(ActorNotExist):
                await ref1.send(('delete', pool.actor_ref('fake_uid')))

            # test self destroy
            ref1 = await pool.create_actor(DummyActor, 2)
            await ref1.send(('destroy',))
            self.assertFalse(await pool.has_actor(ref1))

    async def testProcessUnpickled(self):
        async with await create_actor_pool(
                n_process=2, distributor=DummyDistributor(2)) as pool:
            ref1 = await pool.create_actor(DummyActor, 1)
            with self.assertRaises(pickle.PickleError):
                await ref1.send(lambda x: x)

            ref2 = await pool.create_actor(DummyActor, 1, uid='admin-1')
            with self.assertRaises(pickle.PickleError):
                await ref1.send(('send_unpickled', ref2))

            with self.assertRaises(pickle.PickleError):
                await ref1.send(('send', ref2, 'send_unpickled', ref1))

            with self.assertRaises(pickle.PickleError):
                await pool.create_actor(DummyActor, lambda x: x)

            with self.assertRaises(pickle.PickleError):
                await ref1.send(('create_unpickled',))

    async def testRemoteConnections(self):
        async with await create_actor_pool(address=True, n_process=2) as pool:
            addr = pool.cluster_info.address

            connections = await Connections.create(addr)

            with await connections.connect() as conn:
                default_conn = conn
                # conn's lock has not been released, a new connection will be established
                with await connections.connect() as conn2:
                    self.assertIsNot(conn, conn2)

            with await connections.connect() as conn3:
                self.assertIs(default_conn, conn3)

            del connections
            Connections.addrs = 0

            connections1 = await Connections.create(addr)
            conns1 = [await connections1.connect() for _ in range(100)]

            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                connections2 = await Connections.create(addr2)
                conns2 = [await connections2.connect() for _ in range(100)]

                self.assertEqual(len(connections2.conn), 100)
                [conn.release() for conn in conns2]

                conn = await connections2.connect()  # do not create new connection, reuse old one
                self.assertEqual(len(connections2.conn), 100)
                self.assertTrue(conn.lock.locked())
                conn.release()

                async with await create_actor_pool(
                        address='127.0.0.1:12347', n_process=2) as pool3:
                    addr3 = pool3.cluster_info.address

                    _conns3 = await Connections.create(addr3)

                    async def _unlocker():
                        await asyncio.sleep(0.2)
                        [c.release() for c in conns1]

                    await asyncio.wait([connections1.connect(), _unlocker()],
                                       return_when=asyncio.ALL_COMPLETED)

                    self.assertEqual(len(connections1.conn), 66)

                    del _conns3

    async def testRemotePostCreatePreDestroy(self):
        async with await create_actor_pool(address=True, n_process=1) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            actor_ref = await client.create_actor(EventActor, address=addr)
            await actor_ref.destroy()

    async def testRemoteCreateLocalPoolActor(self):
        # client -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            actor_ref = await client.create_actor(DummyActor, 1, address=addr)
            self.assertIsNotNone(actor_ref)
            self.assertEqual(actor_ref.address, addr)

            ref2 = await actor_ref.send(('ref',))
            self.assertEqual(actor_ref.address, ref2.address)
            self.assertEqual(actor_ref.uid, ref2.uid)

            with self.assertRaises(ValueError):
                await client.create_actor(DummyActor, -1, address=addr)
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ValueError):
                await ref1.send(('create', (DummyActor, -2), dict(address=addr)))
            await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr)))

            # create actor inside on_receive and send message
            r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

    async def testRemoteCreateProcessPoolActor(self):
        # client -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            actor_ref = await client.create_actor(DummyActor, 1, address=addr)
            self.assertIsNotNone(actor_ref)
            self.assertEqual(actor_ref.address, addr)

            with self.assertRaises(ValueError):
                await client.create_actor(DummyActor, -1, address=addr)
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ValueError):
                await ref1.send(('create', (DummyActor, -2), dict(address=addr)))
            await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr)))

            # create actor inside on_receive and send message
            r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
            self.assertEqual(r, 6)

    async def testRemoteCreateLocalPoolToLocalPoolActor(self):
        # client -> local pool -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                actor_ref = await client.create_actor(DummyActor, 1, address=addr1)
                actor_ref2 = await actor_ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    await ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

    async def testRemoteCreateProcessPoolToProcessPoolActor(self):
        # client -> process pool -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                actor_ref = await client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = await ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    await ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

    async def testRemoteCreateLocalPoolToProcessPoolActor(self):
        # client -> local pool -> process pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                actor_ref = await client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = await ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    await ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

    async def testRemoteCreateProcessPoolToLocalPoolActor(self):
        # client -> process pool -> local pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                actor_ref = await client.create_actor(DummyActor, 1, address=addr1)
                ref = client.actor_ref(actor_ref)
                actor_ref2 = await ref.send(('create', (DummyActor, 1), {'address': addr2}))
                self.assertIsNotNone(actor_ref2)
                self.assertEqual(actor_ref2.address, addr2)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ValueError):
                    await ref1.send(('create', (DummyActor, -2), dict(address=addr2)))
                await ref1.send(('create_ignore', (DummyActor, -3), dict(address=addr2)))

                # create actor inside on_receive and send message
                r = await actor_ref.send(('create_send', (DummyActor, 5), {}, ('add', 1)))
                self.assertEqual(r, 6)

    async def testRemoteSendLocalPool(self):
        # client -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                await ref1.send(('add', 1.0))
            ref2 = await client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                await ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                await client.actor_ref('fake_uid', address=addr).send(('add', 1))

    async def testRemoteSendProcessPool(self):
        # client -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(TypeError):
                await ref1.send(('add', 1.0))
            ref2 = await client.create_actor(DummyActor, 2, address=addr)
            with self.assertRaises(TypeError):
                await ref1.send(('send', ref2, 'add', 1.0))
            with self.assertRaises(ActorNotExist):
                await client.actor_ref('fake_uid', address=addr).send(('add', 1))

    async def testRemoteSendLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    await ref1.send(('add', 1.0))
                ref2 = await client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    await ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    await client.actor_ref('fake_uid', address=addr1).send(('add', 1))

    async def testRemoteSendProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    await ref1.send(('add', 1.0))
                ref2 = await client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    await ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    await client.actor_ref('fake_uid', address=addr1).send(('add', 1))

    async def testRemoteSendLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    await ref1.send(('add', 1.0))
                ref2 = await client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    await ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    await client.actor_ref('fake_uid', address=addr1).send(('add', 1))

    async def testRemoteSendProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertEqual(await ref1.send(('send', ref2, 'add', 3)), 5)

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(TypeError):
                    await ref1.send(('add', 1.0))
                ref2 = await client.create_actor(DummyActor, 2, address=addr2)
                with self.assertRaises(TypeError):
                    await ref1.send(('send', ref2, 'add', 1.0))
                with self.assertRaises(ActorNotExist):
                    await client.actor_ref('fake_uid', address=addr1).send(('add', 1))

    async def testRemoteTellLocalPool(self):
        # client -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(await ref2.send(('get',)), 5)

            await ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
            self.assertEqual(await ref2.send(('get',)), 5)
            await asyncio.sleep(.5)
            self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteTellProcessPool(self):
        # client -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool:
            addr = pool.cluster_info.address

            client = new_client()
            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr))))
            self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
            self.assertEqual(await ref2.send(('get',)), 5)

            await ref1.send(('tell_delay', ref2, 'add', 4, 0.3))  # delay 0.3 secs
            self.assertEqual(await ref2.send(('get',)), 5)
            await asyncio.sleep(.5)
            self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteTellLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(await ref2.send(('get',)), 5)

                await ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(await ref2.send(('get',)), 5)
                await asyncio.sleep(.5)
                self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteTellProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                ref1 = await pool2.create_actor(DummyActor, 1, address=addr1)
                ref2 = pool2.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(await ref2.send(('get',)), 5)

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(await ref2.send(('get',)), 5)

                await ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(await ref2.send(('get',)), 5)
                await asyncio.sleep(.5)
                self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteTellLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(await ref2.send(('get',)), 5)

                await ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(await ref2.send(('get',)), 5)
                await asyncio.sleep(.5)
                self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteTellProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()
                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = client.actor_ref(await ref1.send(('create', (DummyActor, 2), dict(address=addr2))))
                self.assertIsNone(await ref1.send(('tell', ref2, 'add', 3)))
                self.assertEqual(await ref2.send(('get',)), 5)

                await ref1.send(('tell_delay', ref2, 'add', 4, .3))  # delay 0.3 secs
                self.assertEqual(await ref2.send(('get',)), 5)
                await asyncio.sleep(.5)
                self.assertEqual(await ref2.send(('get',)), 9)

    async def testRemoteDestroyHasLocalPool(self):
        # client -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool:
            addr = pool.cluster_info.address

            client = new_client()

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            self.assertTrue(await client.has_actor(ref1))
            await client.destroy_actor(ref1)
            self.assertFalse(await client.has_actor(ref1))

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr)))

            await ref1.send(('delete', ref2))
            self.assertFalse(await ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                await client.destroy_actor(client.actor_ref('fake_uid', address=addr))

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ActorNotExist):
                await ref1.send(('delete', client.actor_ref('fake_uid', address=addr)))

            # test self destroy
            ref1 = await client.create_actor(DummyActor, 2, address=addr)
            await ref1.send(('destroy',))
            self.assertFalse(await client.has_actor(ref1))

    async def testRemoteDestroyHasProcessPool(self):
        # client -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool:
            addr = pool.cluster_info.address

            client = new_client()

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            self.assertTrue(await client.has_actor(ref1))

            await client.destroy_actor(ref1)
            self.assertFalse(await client.has_actor(ref1))

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr)))

            await ref1.send(('delete', ref2))
            self.assertFalse(await ref1.send(('has', ref2)))

            with self.assertRaises(ActorNotExist):
                await client.destroy_actor(client.actor_ref('fake_uid', address=addr))

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(ActorNotExist):
                await ref1.send(('delete', client.actor_ref('fake_uid', address=addr)))

            # test self destroy
            ref1 = await client.create_actor(DummyActor, 2, address=addr)
            await ref1.send(('destroy',))
            self.assertFalse(await client.has_actor(ref1))

    async def testRemoteDestroyHasLocalPoolToLocalPool(self):
        # client -> local pool -> local pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(await client.has_actor(ref1))

                await client.destroy_actor(ref1)
                self.assertFalse(await client.has_actor(ref1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                await ref1.send(('delete', ref2))
                self.assertFalse(await ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    await client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    await ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref1 = await client.create_actor(DummyActor, 2, address=addr2)
                await ref1.send(('destroy',))
                self.assertFalse(await client.has_actor(ref1))

    async def testRemoteDestroyHasProcessPoolToProcessPool(self):
        # client -> process pool -> process pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                ref1 = await pool2.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(await pool2.has_actor(ref1))
                await pool2.destroy_actor(ref1)
                self.assertFalse(await pool2.has_actor(ref1))

                client = new_client()

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(await client.has_actor(ref1))

                await client.destroy_actor(ref1)
                self.assertFalse(await client.has_actor(ref1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                await ref1.send(('delete', ref2))
                self.assertFalse(await ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    await client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    await ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref1 = await client.create_actor(DummyActor, 2, address=addr2)
                await ref1.send(('destroy',))
                self.assertFalse(await client.has_actor(ref1))

    async def testRemoteDestroyHasLocalPoolToProcessPool(self):
        # client -> local pool -> process pool
        async with await create_actor_pool(address=True, n_process=1) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=2) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(await client.has_actor(ref1))
                await client.destroy_actor(ref1)
                self.assertFalse(await client.has_actor(ref1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                await ref1.send(('delete', ref2))
                self.assertFalse(await ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    await client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    await ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref1 = await client.create_actor(DummyActor, 2, address=addr2)
                await ref1.send(('destroy',))
                self.assertFalse(await client.has_actor(ref1))

    async def testRemoteDestroyHasProcessPoolToLocalPool(self):
        # client -> process pool -> local pool
        async with await create_actor_pool(address=True, n_process=2) as pool1:
            addr1 = pool1.cluster_info.address
            async with await create_actor_pool(
                    address='127.0.0.1:12346', n_process=1) as pool2:
                addr2 = pool2.cluster_info.address

                client = new_client()

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                self.assertTrue(await client.has_actor(ref1))

                await client.destroy_actor(ref1)
                self.assertFalse(await client.has_actor(ref1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                ref2 = await ref1.send(('create', (DummyActor, 2), dict(address=addr2)))

                await ref1.send(('delete', ref2))
                self.assertFalse(await ref1.send(('has', ref2)))

                with self.assertRaises(ActorNotExist):
                    await client.destroy_actor(client.actor_ref('fake_uid', address=addr1))

                ref1 = await client.create_actor(DummyActor, 1, address=addr1)
                with self.assertRaises(ActorNotExist):
                    await ref1.send(('delete', client.actor_ref('fake_uid', address=addr2)))

                # test self destroy
                ref1 = await client.create_actor(DummyActor, 2, address=addr2)
                await ref1.send(('destroy',))
                self.assertFalse(await client.has_actor(ref1))

    async def testRemoteProcessPoolUnpickled(self):
        async with await create_actor_pool(
                address=True, n_process=2, distributor=DummyDistributor(2)) as pool:
            addr = pool.cluster_info.address

            client = new_client()

            ref1 = await client.create_actor(DummyActor, 1, address=addr)
            with self.assertRaises(pickle.PickleError):
                await ref1.send(lambda x: x)

            ref2 = await client.create_actor(DummyActor, 1, address=addr, uid='admin-1')
            with self.assertRaises(pickle.PickleError):
                await ref1.send(('send_unpickled', ref2))

            with self.assertRaises(pickle.PickleError):
                await ref1.send(('send', ref2, 'send_unpickled', ref1))

            with self.assertRaises(pickle.PickleError):
                await client.create_actor(DummyActor, lambda x: x, address=addr)

            with self.assertRaises(pickle.PickleError):
                await ref1.send(('create_unpickled',))

    async def testRemoteEmpty(self):
        async with await create_actor_pool(
                address=True, n_process=2, distributor=AdminDistributor(2)) as pool:
            addr = pool.cluster_info.address

            client = new_client()

            ref = await client.create_actor(EmptyActor, address=addr)
            self.assertIsNone(await ref.send(None))

    async def testPoolJoin(self):
        async with await create_actor_pool(
                address=True, n_process=2, distributor=AdminDistributor(2)) as pool:
            start = time.time()
            await pool.join(0.2)

            self.assertGreaterEqual(time.time() - start, 0.2)

            async def _stop_fun():
                await asyncio.sleep(0.2)
                await pool.stop()

            start = time.time()
            asyncio.ensure_future(_stop_fun())
            await pool.join()
            self.assertGreaterEqual(time.time() - start, 0.2)

    async def testConcurrentSend(self):
        async with await create_actor_pool(
                address=True, n_process=4, distributor=AdminDistributor(4)) as pool:
            ref1 = await pool.create_actor(DummyActor, 0)

            async def ref_send(ref, rg):
                p = []
                for i in range(*rg):
                    p.append(asyncio.ensure_future(ref.send(('send', ref1, 'add_ret', i))))
                self.assertEqual([await f for f in p], list(range(*rg)))

            n_ref = 20

            ps = [asyncio.ensure_future(pool.create_actor(DummyActor, 0))
                  for _ in range(n_ref)]
            refs = [await p for p in ps]

            ps = []
            for i in range(n_ref):
                r = (i * 100, (i + 1) * 100)
                refx = refs[i]
                ps.append(ref_send(refx, r))

            await asyncio.wait(ps, return_when=asyncio.ALL_COMPLETED)

    async def testRemoteBrokenPipe(self):
        pool1 = await create_actor_pool(address=True, n_process=1)
        addr = pool1.cluster_info.address

        try:
            client = new_client(parallel=1)
            # make client create a connection
            await client.create_actor(DummyActor, 10, address=addr)

            # stop
            await pool1.stop()

            # the connection is broken
            with self.assertRaises(BrokenPipeError):
                await client.create_actor(DummyActor, 10, address=addr)

            pool1 = await create_actor_pool(address=addr, n_process=1, auto_port=False)

            # create a new pool, so the actor can be created
            await client.create_actor(DummyActor, 10, address=addr)
        finally:
            await pool1.stop()
