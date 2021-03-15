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

import pytest

import mars.oscar as mo
from mars.tests.core import require_ray
from mars.utils import lazy_import

ray = lazy_import('ray', globals=globals())

RAY_TEST_ADDRESS = 'ray://'


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
        return await mo.create_actor(actor_cls, *args, address=RAY_TEST_ADDRESS, **kw)

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
        actor_ref = await mo.actor_ref(uid)
        return await getattr(actor_ref, method)(*args)

    async def tell(self, uid, method, *args):
        actor_ref = await mo.actor_ref(uid)
        return getattr(actor_ref, method).tell(*args)

    async def tell_delay(self, uid, method, *args, delay=None):
        actor_ref = await mo.actor_ref(uid)
        getattr(actor_ref, method).tell_delay(*args, delay=delay)

    async def send_unpickled(self, value):
        actor_ref = await mo.actor_ref(value)
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


@pytest.fixture
def ray_pool():
    ray.init()
    yield
    ray.shutdown()


@require_ray
@pytest.mark.asyncio
async def test_simple_ray_actor_pool(ray_pool):
    actor_ref = await mo.create_actor(DummyActor, 100, address=RAY_TEST_ADDRESS)
    assert await actor_ref.add(1) == 101
    await actor_ref.add(1)

    res = await actor_ref.get_value()
    assert res == 102

    ref2 = await actor_ref.get_ref()
    assert actor_ref.address == ref2.address
    assert actor_ref.uid == ref2.uid

    ref = await mo.actor_ref(uid=actor_ref.uid, address=actor_ref.address)
    assert await ref.add(2) == 104


@require_ray
@pytest.mark.asyncio
async def test_ray_post_create_pre_destroy(ray_pool):
    actor_ref = await mo.create_actor(EventActor, address=RAY_TEST_ADDRESS)
    await actor_ref.destroy()


@require_ray
@pytest.mark.asyncio
async def test_ray_create_actor(ray_pool):
    actor_ref = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    # create actor inside on_receive
    ref = await actor_ref.create(DummyActor, 5)
    assert await ref.add(10) == 15
    # create actor inside on_receive and send message
    r = await actor_ref.create_send(DummyActor, 5, method='add', method_args=(1,), address=RAY_TEST_ADDRESS)
    assert r == 6


@require_ray
@pytest.mark.asyncio
async def test_ray_create_actor_error(ray_pool):
    ref1 = await mo.create_actor(DummyActor, 1, uid='dummy1', address=RAY_TEST_ADDRESS)
    with pytest.raises(mo.ActorAlreadyExist):
        await mo.create_actor(DummyActor, 1, uid='dummy1', address=RAY_TEST_ADDRESS)
    await mo.destroy_actor(ref1)

    # It's hard to get the exception raised from Actor.__init__.
    # with pytest.raises(ValueError):
    #     await mo.create_actor(DummyActor, -1, address=RAY_TEST_ADDRESS)
    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    with pytest.raises(ValueError):
        await ref1.create(DummyActor, -2)


@require_ray
@pytest.mark.asyncio
async def test_ray_send(ray_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    ref2 = await ref1.create(DummyActor, 2)
    assert await ref1.send(ref2, 'add', 3) == 5


@require_ray
@pytest.mark.asyncio
async def test_ray_send_error(ray_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    with pytest.raises(TypeError):
        await ref1.add(1.0)
    ref2 = await mo.create_actor(DummyActor, 2, address=RAY_TEST_ADDRESS)
    with pytest.raises(TypeError):
        await ref1.send(ref2, 'add', 1.0)
    with pytest.raises(mo.ActorNotExist):
        await (await mo.actor_ref('fake_uid', address=RAY_TEST_ADDRESS)).add(1)


@require_ray
@pytest.mark.asyncio
async def test_ray_tell(ray_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    ref2 = await ref1.create(DummyActor, 2)
    assert await ref1.tell(ref2, 'add', 3) is None
    assert await ref2.get_value() == 5

    await ref1.tell_delay(ref2, 'add', 4, delay=.5)  # delay 0.5 secs
    assert await ref2.get_value() == 5
    await asyncio.sleep(0.5)
    assert await ref2.get_value() == 9

    # error needed when illegal uids are passed
    with pytest.raises(ValueError):
        await ref1.tell(await mo.actor_ref(set()), 'add', 3)


@require_ray
@pytest.mark.asyncio
async def test_ray_destroy_has_actor(ray_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    assert await mo.has_actor(ref1)

    await mo.destroy_actor(ref1)
    await asyncio.sleep(.5)
    assert not await mo.has_actor(ref1)

    # error needed when illegal uids are passed
    with pytest.raises(ValueError):
        await mo.has_actor(await mo.actor_ref(set()))

    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    await mo.destroy_actor(ref1)
    await asyncio.sleep(.5)
    assert not await mo.has_actor(ref1)

    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    ref2 = await ref1.create(DummyActor, 2)

    assert await mo.has_actor(ref2)

    await ref1.delete(ref2)
    assert not await ref1.has(ref2)

    with pytest.raises(mo.ActorNotExist):
        await mo.destroy_actor(
            await mo.actor_ref('fake_uid', address=RAY_TEST_ADDRESS))

    ref1 = await mo.create_actor(DummyActor, 1, address=RAY_TEST_ADDRESS)
    with pytest.raises(mo.ActorNotExist):
        await ref1.delete(await mo.actor_ref('fake_uid', address=RAY_TEST_ADDRESS))

    # test self destroy
    ref1 = await mo.create_actor(DummyActor, 2, address=RAY_TEST_ADDRESS)
    await ref1.destroy()
    await asyncio.sleep(.5)
    assert not await mo.has_actor(ref1)
