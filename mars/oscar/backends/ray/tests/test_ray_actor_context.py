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
import unittest
import ray

from mars.tests.core import aio_case
import mars.oscar as mo

RAY_TEST_ADDRESS = 'ray://test'


class DummyActor(mo.Actor):
    def __init__(self, value):
        super().__init__()

        if value < 0:
            raise ValueError('value < 0')
        self.value = value

    def add(self, value):
        if not isinstance(value, int):
            raise TypeError('add number must be int')
        self.value += value
        return self.value

    def add_ret(self, value):
        return self.value + value

    async def create(self, actor_cls, *args, **kw):
        return await mo.create_actor(actor_cls, *args, **kw)

    async def create_ignore(self, actor_cls, *args, **kw):
        try:
            return await mo.create_actor(actor_cls, *args, **kw)
        except ValueError:
            pass

    async def create_send(self, actor_cls, *args, **kw):
        method = kw.pop('method')
        method_args = kw.pop('method_args')
        ref = await mo.create_actor(actor_cls, *args, **kw)
        return await getattr(ref, method)(*method_args)

    async def delete(self, value):
        return await mo.destroy_actor(value)

    async def has(self, value):
        return await mo.has_actor(value)

    async def send(self, uid, method, *args):
        actor_ref = mo.actor_ref(uid)
        return await getattr(actor_ref, method)(*args)

    async def tell(self, uid, method, *args):
        actor_ref = mo.actor_ref(uid)
        return getattr(actor_ref, method).tell(*args)

    async def tell_delay(self, uid, method, *args, delay=None):
        actor_ref = mo.actor_ref(uid)
        getattr(actor_ref, method).tell_delay(*args, delay=delay)

    async def send_unpickled(self, value):
        actor_ref = mo.actor_ref(value)
        return await actor_ref.send(lambda x: x)

    async def create_unpickled(self):
        return await mo.create_actor(DummyActor, lambda x: x, uid='admin-5')

    async def destroy(self):
        await self.ref().destroy()

    def get_value(self):
        return self.value

    def get_ref(self):
        return self.ref()


class EventActor(mo.Actor):
    async def __post_create__(self):
        assert 'sth' == await self.ref().echo('sth')

    async def pre_destroy(self):
        assert 'sth2' == await self.ref().echo('sth2')

    def echo(self, message):
        return message


@aio_case
class Test(unittest.TestCase):
    def setUp(self) -> None:
        ray.init()

    def tearDown(self) -> None:
        ray.shutdown()

    async def testSimpleRayActorPool(self):
        actor_ref = await mo.create_actor(DummyActor, 100, address=RAY_TEST_ADDRESS)
        self.assertEqual(await actor_ref.add(1), 101)
        await actor_ref.add(1)

        res = await actor_ref.get_value()
        self.assertEqual(res, 102)

        ref2 = await actor_ref.get_ref()
        self.assertEqual(actor_ref.address, ref2.address)
        self.assertEqual(actor_ref.uid, ref2.uid)

        self.assertEqual(await mo.actor_ref(
                uid=actor_ref.uid, address=actor_ref.address).add(2), 104)

    async def testRayPostCreatePreDestroy(self):
        actor_ref = await mo.create_actor(EventActor, address=RAY_TEST_ADDRESS)
        await actor_ref.destroy()

    async def testRayCreateActor(self):
        actor_ref = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
        # create actor inside on_receive
        r = await actor_ref.create(DummyActor, 5, address=RAY_TEST_ADDRESS)
        ref = mo.actor_ref(r)
        self.assertEqual(await ref.add(10), 15)
        # create actor inside on_receive and send message
        r = await actor_ref.create_send(DummyActor, 5, method='add', method_args=(1,), address=RAY_TEST_ADDRESS)
        self.assertEqual(r, 6)

    async def testRayCreateActorError(self):
        ref1 = await mo.create_actor(DummyActor, 1, uid='dummy1', address=RAY_TEST_ADDRESS)
        with self.assertRaises(mo.ActorAlreadyExist):
            await mo.create_actor(DummyActor, 1, uid='dummy1', address=RAY_TEST_ADDRESS)
        await mo.destroy_actor(ref1)

        # It's hard to get the exception raised from Actor.__init__.
        # with self.assertRaises(ValueError):
        #     await mo.create_actor(DummyActor, -1, address=RAY_TEST_ADDRESS)
        ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
        with self.assertRaises(ValueError):
            await ref1.create(DummyActor, -2)

    async def testRaySend(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        ref2 = mo.actor_ref(await ref1.create(DummyActor, 2))
        self.assertEqual(await ref1.send(ref2, 'add', 3), 5)

    async def testRaySendError(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        with self.assertRaises(TypeError):
            await ref1.add(1.0)
        ref2 = await mo.create_actor(DummyActor, 2)
        with self.assertRaises(TypeError):
            await ref1.send(ref2, 'add', 1.0)
        with self.assertRaises(mo.ActorNotExist):
            await mo.actor_ref('fake_uid').add(1)

    async def testRayTell(self):
        ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
        ref2 = mo.actor_ref(await ref1.create(DummyActor, 2, address=RAY_TEST_ADDRESS))
        self.assertIsNone(await ref1.tell(ref2, 'add', 3))
        self.assertEqual(await ref2.get_value(), 5)

        await ref1.tell_delay(ref2, 'add', 4, delay=.5)  # delay 0.5 secs
        self.assertEqual(await ref2.get_value(), 5)
        await asyncio.sleep(0.5)
        self.assertEqual(await ref2.get_value(), 9)

        # error needed when illegal uids are passed
        with self.assertRaises(TypeError):
            await ref1.tell(mo.actor_ref(set()), 'add', 3)

    async def testRayDestroyHasActor(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        self.assertTrue(await mo.has_actor(ref1))

        await mo.destroy_actor(ref1)
        self.assertFalse(await mo.has_actor(ref1))

        # error needed when illegal uids are passed
        with self.assertRaises(TypeError):
            await mo.has_actor(await mo.actor_ref(set()))

        ref1 = await mo.create_actor(DummyActor, 1)
        await mo.destroy_actor(ref1)
        self.assertFalse(await mo.has_actor(ref1))

        ref1 = await mo.create_actor(DummyActor, 1)
        ref2 = await ref1.create(DummyActor, 2)

        self.assertTrue(await mo.has_actor(ref2))

        await ref1.delete(ref2)
        self.assertFalse(await ref1.has(ref2))

        with self.assertRaises(mo.ActorNotExist):
            await mo.destroy_actor(mo.actor_ref('fake_uid'))

        ref1 = await mo.create_actor(DummyActor, 1)
        with self.assertRaises(mo.ActorNotExist):
            await ref1.delete(mo.actor_ref('fake_uid'))

        # test self destroy
        ref1 = await mo.create_actor(DummyActor, 2)
        await ref1.destroy()
        self.assertFalse(await mo.has_actor(ref1))
