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
import time
import unittest
from collections import deque

import pandas as pd

import mars.oscar as mo
from mars.tests.core import aio_case


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


class ResourceLockActor(mo.Actor):
    def __init__(self, count=1):
        self._count = count
        self._requests = deque()

    async def apply(self, val=None):
        if self._count:
            self._count -= 1
            return val + 1 if val is not None else None

        event = asyncio.Event()

        async def waiter():
            await event.wait()
            self._count -= 1
            return val + 1 if val is not None else None

        self._requests.append(event)
        return waiter()

    def release(self):
        self._count += 1
        if self._requests:
            event = self._requests.popleft()
            event.set()


class PromiseTestActor(mo.Actor):
    def __init__(self, res_lock_ref):
        self.res_lock_ref = res_lock_ref
        self.call_log = []

    async def _apply_step(self, idx, delay):
        res = None
        try:
            self.call_log.append(('A', idx, time.time()))
            res = yield self.res_lock_ref.apply(idx)
            assert res == idx + 1

            self.call_log.append(('B', idx, time.time()))
            yield asyncio.sleep(delay)
            self.call_log.append(('C', idx, time.time()))
        finally:
            await self.res_lock_ref.release()
            yield res

    async def test_promise_call(self, idx, delay=0.1):
        return self._apply_step(idx, delay)

    async def test_yield_tuple(self, delay=0.1):
        yield tuple(
            self._apply_step(idx, delay) for idx in range(4)
        ) + (asyncio.sleep(delay), 'PlainString')

    async def test_exceptions(self):
        async def async_raiser():
            yield asyncio.sleep(0.1)
            raise SystemError

        try:
            yield async_raiser(),
        except SystemError:
            raise ValueError
        raise KeyError

    def get_call_log(self):
        log = self.call_log
        self.call_log = []
        return log


@aio_case
class Test(unittest.TestCase):
    async def testSimpleLocalActorPool(self):
        actor_ref = await mo.create_actor(DummyActor, 100)
        self.assertEqual(await actor_ref.add(1), 101)
        await actor_ref.add(1)

        res = await actor_ref.get_value()
        self.assertEqual(res, 102)

        ref2 = await actor_ref.get_ref()
        self.assertEqual(actor_ref.address, ref2.address)
        self.assertEqual(actor_ref.uid, ref2.uid)

        self.assertEqual(await mo.actor_ref(
            uid=actor_ref.uid).add(2), 104)

    async def testLocalPostCreatePreDestroy(self):
        actor_ref = await mo.create_actor(EventActor)
        await actor_ref.destroy()

    async def testLocalCreateActor(self):
        actor_ref = await mo.create_actor(DummyActor, 1)
        # create actor inside on_receive
        r = await actor_ref.create(DummyActor, 5)
        ref = mo.actor_ref(r)
        self.assertEqual(await ref.add(10), 15)
        # create actor inside on_receive and send message
        r = await actor_ref.create_send(DummyActor, 5, method='add', method_args=(1,))
        self.assertEqual(r, 6)

    async def testLocalCreateActorError(self):
        ref1 = await mo.create_actor(DummyActor, 1, uid='dummy1')
        with self.assertRaises(mo.ActorAlreadyExist):
            await mo.create_actor(DummyActor, 1, uid='dummy1')
        await mo.destroy_actor(ref1)

        with self.assertRaises(ValueError):
            await mo.create_actor(DummyActor, -1)
        ref1 = await mo.create_actor(DummyActor, 1)
        with self.assertRaises(ValueError):
            await ref1.create(DummyActor, -2)

    async def testLocalSend(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        ref2 = mo.actor_ref(await ref1.create(DummyActor, 2))
        self.assertEqual(await ref1.send(ref2, 'add', 3), 5)

    async def testLocalSendError(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        with self.assertRaises(TypeError):
            await ref1.add(1.0)
        ref2 = await mo.create_actor(DummyActor, 2)
        with self.assertRaises(TypeError):
            await ref1.send(ref2, 'add', 1.0)
        with self.assertRaises(mo.ActorNotExist):
            await mo.actor_ref('fake_uid').add(1)

    async def testLocalTell(self):
        ref1 = await mo.create_actor(DummyActor, 1)
        ref2 = mo.actor_ref(await ref1.create(DummyActor, 2))
        self.assertIsNone(await ref1.tell(ref2, 'add', 3))
        self.assertEqual(await ref2.get_value(), 5)

        await ref1.tell_delay(ref2, 'add', 4, delay=.5)  # delay 0.5 secs
        self.assertEqual(await ref2.get_value(), 5)
        await asyncio.sleep(0.5)
        self.assertEqual(await ref2.get_value(), 5)

        # error needed when illegal uids are passed
        with self.assertRaises(TypeError):
            await ref1.tell(mo.actor_ref(set()), 'add', 3)

    async def testLocalDestroyHasActor(self):
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

    async def testLocalResourceLock(self):
        ref = await mo.create_actor(ResourceLockActor)
        event_list = []

        async def test_task(idx):
            await ref.apply()
            event_list.append(('A', idx, time.time()))
            await asyncio.sleep(0.1)
            event_list.append(('B', idx, time.time()))
            await ref.release()

        tasks = [asyncio.create_task(test_task(idx)) for idx in range(4)]
        await asyncio.wait(tasks)

        for idx in range(0, len(event_list), 2):
            event_pair = event_list[idx:idx + 2]
            self.assertEqual((event_pair[0][0], event_pair[1][0]), ('A', 'B'))
            self.assertEqual(event_pair[0][1], event_pair[1][1])

    async def testPromiseChain(self):
        lock_ref = await mo.create_actor(ResourceLockActor, 2)
        promise_test_ref = await mo.create_actor(PromiseTestActor, lock_ref)

        start_time = time.time()
        tasks = [asyncio.create_task(promise_test_ref.test_promise_call(idx, delay=0.1))
                 for idx in range(4)]
        dones, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        [t.result() for t in dones]

        logs = pd.DataFrame(await promise_test_ref.get_call_log(), columns=['group', 'idx', 'time'])
        logs.time -= start_time
        self.assertLess(logs.query('group == "A"').time.max(), 0.05)
        max_apply_time = logs.query('group == "A" | group == "B"').groupby('idx') \
            .apply(lambda s: s.time.max() - s.time.min()).max()
        self.assertGreater(max_apply_time, 0.1)
        max_delay_time = logs.query('group == "B" | group == "C"').groupby('idx') \
            .apply(lambda s: s.time.max() - s.time.min()).max()
        self.assertGreater(max_delay_time, 0.1)

        start_time = time.time()
        ret = await promise_test_ref.test_yield_tuple()
        self.assertSetEqual(set(ret), {1, 2, 3, 4, None, 'PlainString'})

        logs = pd.DataFrame(await promise_test_ref.get_call_log(), columns=['group', 'idx', 'time'])
        logs.time -= start_time
        self.assertLess(logs.query('group == "A"').time.max(), 0.05)
        max_apply_time = logs.query('group == "A" | group == "B"').groupby('idx') \
            .apply(lambda s: s.time.max() - s.time.min()).max()
        self.assertGreater(max_apply_time, 0.1)
        max_delay_time = logs.query('group == "B" | group == "C"').groupby('idx') \
            .apply(lambda s: s.time.max() - s.time.min()).max()
        self.assertGreater(max_delay_time, 0.1)

        with self.assertRaises(ValueError):
            await promise_test_ref.test_exceptions()
