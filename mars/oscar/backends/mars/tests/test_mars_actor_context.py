# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import logging
import os
import sys
import time
import traceback
from collections import deque

import pandas as pd
import pytest

from ..... import oscar as mo
from .....oscar.core import ActorRef, LocalActorRef
from ....backends.allocate_strategy import RandomSubPool
from ....debug import set_debug_options, get_debug_options, DebugOptions
from ...router import Router

logger = logging.getLogger(__name__)


class DummyActor(mo.Actor):
    def __init__(self, value):
        super().__init__()

        if value < 0:
            raise ValueError("value < 0")
        self.value = value

    @mo.extensible
    async def add(self, value):
        if not isinstance(value, int):
            raise TypeError("add number must be int")
        self.value += value
        return self.value

    @add.batch
    async def add(self, args_list, _kwargs_list):
        self.value += sum(v[0] for v in args_list)
        return self.value

    @mo.extensible
    async def add_ret(self, value):
        return self.value + value

    @add_ret.batch
    async def add_ret(self, args_list, _kwargs_list):
        sum_val = sum(v[0] for v in args_list)
        return [self.value + sum_val for _ in args_list]

    async def create(self, actor_cls, *args, **kw):
        kw["address"] = self.address
        return await mo.create_actor(actor_cls, *args, **kw)

    async def create_ignore(self, actor_cls, *args, **kw):
        try:
            return await mo.create_actor(actor_cls, *args, **kw)
        except ValueError:
            pass

    async def create_send(self, actor_cls, *args, **kw):
        method = kw.pop("method")
        method_args = kw.pop("method_args")
        ref = await mo.create_actor(actor_cls, *args, **kw)
        return await getattr(ref, method)(*method_args)

    async def delete(self, value):
        return await mo.destroy_actor(value)

    async def has(self, value):
        return await mo.has_actor(value)

    async def send(self, uid, method, *args):
        actor_ref = await mo.actor_ref(uid, address=self.address)
        tp = (
            LocalActorRef
            if actor_ref.address == self.address and get_debug_options() is None
            else ActorRef
        )
        assert (
            type(actor_ref) is tp
        ), f"Expect type of actor ref is {tp}, but got {actor_ref} instead."
        return await getattr(actor_ref, method)(*args)

    async def tell(self, uid, method, *args):
        actor_ref = await mo.actor_ref(uid, address=self.address)
        await getattr(actor_ref, method).tell(*args)

    async def tell_delay(self, uid, method, *args, delay=None):
        actor_ref = await mo.actor_ref(uid)
        getattr(actor_ref, method).tell_delay(*args, delay=delay)

    async def send_unpickled(self, value):
        actor_ref = await mo.actor_ref(value)
        return await actor_ref.send(lambda x: x)

    async def create_unpickled(self):
        return await mo.create_actor(DummyActor, lambda x: x, uid="admin-5")

    async def destroy(self):
        await self.ref().destroy()

    def get_value(self):
        return self.value

    def get_ref(self):
        ref = self.ref()
        tp = LocalActorRef if get_debug_options() is None else ActorRef
        assert (
            type(ref) is tp
        ), f"Expect type of actor ref is {tp}, but got {ref} instead."
        return ref


class RecordActor(mo.Actor):
    def __init__(self):
        self._records = []

    def add_record(self, rec):
        self._records.append(rec)

    def get_records(self):
        return self._records


class CreateDestroyActor(mo.Actor):
    def __init__(self):
        self._record_ref = None

    async def __post_create__(self):
        self._record_ref = await mo.actor_ref(
            RecordActor.default_uid(), address=self.address
        )
        await self._record_ref.add_record(f"create {self.uid}")
        assert "sth" == await self.ref().echo("sth")

    async def __pre_destroy__(self):
        await self._record_ref.add_record(f"destroy {self.uid}")
        assert "sth2" == await self.ref().echo("sth2")

    def echo(self, message):
        return message


class ResourceLockActor(mo.StatelessActor):
    def __init__(self, count=1):
        self._sem = asyncio.Semaphore(count)
        self._requests = deque()

    async def apply(self, val=None):
        await self._sem.acquire()
        return val + 1 if val is not None else None

    def release(self):
        self._sem.release()


class PromiseTestActor(mo.Actor):
    def __init__(self, res_lock_ref):
        self.res_lock_ref = res_lock_ref
        self.call_log = []

    async def _apply_step(self, idx, delay):
        res = None
        try:
            self.call_log.append(("A", idx, time.time()))
            res = yield self.res_lock_ref.apply(idx)
            assert res == idx + 1

            self.call_log.append(("B", idx, time.time()))
            yield asyncio.sleep(delay)
            self.call_log.append(("C", idx, time.time()))
        finally:
            yield self.res_lock_ref.release()
            raise mo.Return(res)

    async def test_promise_call(self, idx, delay=0.1):
        return self._apply_step(idx, delay)

    async def test_yield_tuple(self, delay=0.1):
        tp = yield tuple(self._apply_step(idx, delay) for idx in range(4)) + (
            asyncio.sleep(delay),
            "PlainString",
        )
        raise mo.Return(tp)

    async def async_raiser_func(self):
        yield asyncio.sleep(0.1)
        raise ValueError

    async def test_yield_exceptions(self):
        task = asyncio.create_task(self.ref().async_raiser_func())
        return task

    async def test_exceptions(self):
        async def async_raiser():
            yield asyncio.sleep(0.1)
            raise SystemError

        try:
            yield async_raiser(),
        except SystemError:
            raise ValueError
        raise KeyError

    async def test_cancel(self, delay):
        async def intermediate_error():
            raise ValueError

        async def task_fun():
            try:
                yield intermediate_error()
            except ValueError:
                pass
            try:
                yield asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.call_log.append((time.time(), "CANCELLED"))
                raise

        self.call_log.append((time.time(), "START"))
        return task_fun()

    def get_call_log(self):
        log = self.call_log
        self.call_log = []
        return log


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=[False, True])
async def actor_pool(request):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await mo.create_actor_pool(
        "127.0.0.1", n_process=2, subprocess_start_method=start_method
    )

    try:
        if request.param:
            set_debug_options(DebugOptions())
        else:
            set_debug_options(None)

        await pool.start()
        yield pool
        await pool.stop()
    finally:
        set_debug_options(None)


@pytest.mark.asyncio
async def test_simple_local_actor_pool(actor_pool):
    actor_ref = await mo.create_actor(
        DummyActor, 100, address=actor_pool.external_address
    )
    assert await actor_ref.add(1) == 101
    await actor_ref.add(1)

    res = await actor_ref.get_value()
    assert res == 102

    ref2 = await actor_ref.get_ref()
    assert actor_ref.address == ref2.address
    assert actor_ref.uid == ref2.uid

    ref = await mo.actor_ref(uid=actor_ref.uid, address=actor_pool.external_address)
    assert await ref.add(2) == 104


@pytest.mark.asyncio
async def test_mars_post_create_pre_destroy(actor_pool):
    rec_ref = await mo.create_actor(
        RecordActor, uid=RecordActor.default_uid(), address=actor_pool.external_address
    )
    actor_ref = await mo.create_actor(
        CreateDestroyActor, address=actor_pool.external_address
    )
    await actor_ref.destroy()

    records = await rec_ref.get_records()
    assert len(records) == 2
    assert records[0].startswith("create")
    assert records[1].startswith("destroy")


@pytest.mark.asyncio
async def test_mars_create_actor(actor_pool):
    actor_ref = await mo.create_actor(
        DummyActor, 1, address=actor_pool.external_address
    )
    # create actor inside on_receive
    r = await actor_ref.create(DummyActor, 5, address=actor_pool.external_address)
    ref = await mo.actor_ref(r, address=actor_pool.external_address)
    assert await ref.add(10) == 15
    # create actor inside on_receive and send message
    r = await actor_ref.create_send(
        DummyActor,
        5,
        method="add",
        method_args=(1,),
        address=actor_pool.external_address,
    )
    assert r == 6


@pytest.mark.asyncio
async def test_mars_create_actor_error(actor_pool):
    ref1 = await mo.create_actor(
        DummyActor, 1, uid="dummy1", address=actor_pool.external_address
    )
    with pytest.raises(mo.ActorAlreadyExist):
        await mo.create_actor(
            DummyActor, 1, uid="dummy1", address=actor_pool.external_address
        )
    await mo.destroy_actor(ref1)

    with pytest.raises(ValueError):
        await mo.create_actor(DummyActor, -1, address=actor_pool.external_address)
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    with pytest.raises(ValueError):
        await ref1.create(DummyActor, -2, address=actor_pool.external_address)


@pytest.mark.asyncio
async def test_mars_send(actor_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    ref2 = await mo.actor_ref(
        await ref1.create(DummyActor, 2, address=actor_pool.external_address)
    )
    assert await ref1.send(ref2, "add", 3) == 5

    ref3 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    ref4 = await mo.create_actor(
        DummyActor,
        2,
        address=actor_pool.external_address,
        allocate_strategy=RandomSubPool(),
    )
    assert await ref4.send(ref3, "add", 3) == 4


@pytest.mark.asyncio
async def test_mars_send_error(actor_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    with pytest.raises(TypeError):
        await ref1.add(1.0)
    ref2 = await mo.create_actor(DummyActor, 2, address=actor_pool.external_address)
    with pytest.raises(TypeError):
        await ref1.send(ref2, "add", 1.0)
    with pytest.raises(mo.ActorNotExist):
        await (await mo.actor_ref("fake_uid", address=actor_pool.external_address)).add(
            1
        )


@pytest.mark.asyncio
async def test_mars_tell(actor_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    ref2 = await mo.actor_ref(await ref1.create(DummyActor, 2))
    await ref1.tell(ref2, "add", 3)
    assert await ref2.get_value() == 5

    await ref1.tell_delay(ref2, "add", 4, delay=0.5)  # delay 0.5 secs
    assert await ref2.get_value() == 5
    await asyncio.sleep(0.45)
    assert await ref2.get_value() == 5
    await asyncio.sleep(0.2)
    assert await ref2.get_value() == 9

    # error needed when illegal uids are passed
    with pytest.raises(ValueError):
        await ref1.tell(await mo.actor_ref(set()), "add", 3)


@pytest.mark.asyncio
async def test_mars_batch_method(actor_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    batch_result = await ref1.add_ret.batch(
        ref1.add_ret.delay(1), ref1.add_ret.delay(2), ref1.add_ret.delay(3)
    )
    assert len(batch_result) == 3
    assert all(r == 7 for r in batch_result)

    await ref1.add.batch(
        ref1.add.delay(1), ref1.add.delay(2), ref1.add.delay(3), send=False
    )
    assert await ref1.get_value() == 7

    with pytest.raises(ValueError):
        await ref1.add_ret.batch(ref1.add_ret.delay(1), ref1.add.delay(2))


@pytest.mark.asyncio
async def test_gather_exception(actor_pool):
    try:
        Router.get_instance_or_empty()._cache.clear()
        ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
        router = Router.get_instance_or_empty()
        client = next(iter(router._cache.values()))

        future = asyncio.Future()
        client_channel = client.channel

        class FakeChannel(type(client_channel)):
            def __init__(self):
                pass

            def __getattr__(self, item):
                return getattr(client_channel, item)

            async def recv(self):
                return await future

        client.channel = FakeChannel()

        class MyException(Exception):
            pass

        await ref1.add(1)
        tasks = [ref1.add(i) for i in range(200)]
        future.set_exception(MyException("Test recv exception!!"))
        with pytest.raises(MyException) as ex:
            await asyncio.gather(*tasks)
        s = traceback.format_tb(ex.tb)
        assert 10 > "\n".join(s).count("send") > 0
    finally:
        Router.get_instance_or_empty()._cache.clear()


@pytest.mark.asyncio
async def test_mars_destroy_has_actor(actor_pool):
    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    ref2 = await mo.actor_ref(ref1)
    ref2_add_method = ref2.add
    assert isinstance(ref1, ActorRef)
    assert await mo.has_actor(ref2)
    await mo.destroy_actor(ref2)
    assert not await mo.has_actor(ref1)
    assert not await mo.has_actor(ref2)

    if isinstance(ref2, LocalActorRef):
        assert "weakref" in str(ref2)
        assert "dead" in str(ref2)

    # error needed when illegal uids are passed
    with pytest.raises(ValueError):
        await mo.has_actor(await mo.actor_ref(set()))

    with pytest.raises(mo.ActorNotExist):
        await ref2.add(1)

    with pytest.raises(mo.ActorNotExist):
        await ref2_add_method(1)

    ref1 = await mo.create_actor(
        DummyActor, 1, uid=ref1.uid, address=actor_pool.external_address
    )

    # the ref2 should be works after actor is recreated.
    assert await ref2.add(1) == 2
    # the ref2 method should be works after actor is recreated.
    assert await ref2_add_method(1) == 3

    assert isinstance(ref2, ActorRef)
    assert await mo.has_actor(ref1)
    await mo.destroy_actor(ref1)
    assert not await mo.has_actor(ref1)
    assert not await mo.has_actor(ref2)

    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    ref2 = await ref1.create(DummyActor, 2, address=actor_pool.external_address)

    assert await mo.has_actor(ref2)

    await ref1.delete(ref2)
    assert not await ref1.has(ref2)

    with pytest.raises(mo.ActorNotExist):
        await mo.destroy_actor(
            await mo.actor_ref("fake_uid", address=actor_pool.external_address)
        )

    ref1 = await mo.create_actor(DummyActor, 1, address=actor_pool.external_address)
    with pytest.raises(mo.ActorNotExist):
        await ref1.delete(
            await mo.actor_ref("fake_uid", address=actor_pool.external_address)
        )

    # test self destroy
    ref1 = await mo.create_actor(DummyActor, 2, address=actor_pool.external_address)
    await ref1.destroy()
    assert not await mo.has_actor(ref1)


@pytest.mark.asyncio
async def test_mars_resource_lock(actor_pool):
    ref = await mo.create_actor(ResourceLockActor, address=actor_pool.external_address)
    event_list = []

    async def test_task(idx):
        await ref.apply()
        event_list.append(("A", idx, time.time()))
        await asyncio.sleep(0.1)
        event_list.append(("B", idx, time.time()))
        await ref.release()

    tasks = [asyncio.create_task(test_task(idx)) for idx in range(4)]
    await asyncio.wait(tasks)

    for idx in range(0, len(event_list), 2):
        event_pair = event_list[idx : idx + 2]
        assert (event_pair[0][0], event_pair[1][0]) == ("A", "B")
        assert event_pair[0][1] == event_pair[1][1]


@pytest.mark.asyncio
async def test_promise_chain(actor_pool):
    lock_ref = await mo.create_actor(
        ResourceLockActor, 2, address=actor_pool.external_address
    )
    promise_test_ref = await mo.create_actor(
        PromiseTestActor, lock_ref, address=actor_pool.external_address
    )

    delay_val = 1.0

    start_time = time.time()
    tasks = [
        asyncio.create_task(promise_test_ref.test_promise_call(idx, delay=delay_val))
        for idx in range(4)
    ]
    await asyncio.gather(*tasks)

    logs = pd.DataFrame(
        await promise_test_ref.get_call_log(), columns=["group", "idx", "time"]
    )
    logs.time -= start_time
    assert logs.query('group == "A"').time.max() < delay_val / 2
    max_apply_time = (
        logs.query('group == "A" | group == "B"')
        .groupby("idx")
        .apply(lambda s: s.time.max() - s.time.min())
        .max()
    )
    assert max_apply_time > delay_val / 2
    max_delay_time = (
        logs.query('group == "B" | group == "C"')
        .groupby("idx")
        .apply(lambda s: s.time.max() - s.time.min())
        .max()
    )
    assert max_delay_time > delay_val / 2

    start_time = time.time()
    ret = await promise_test_ref.test_yield_tuple(delay=delay_val)
    assert set(ret) == {1, 2, 3, 4, None, "PlainString"}

    logs = pd.DataFrame(
        await promise_test_ref.get_call_log(), columns=["group", "idx", "time"]
    )
    logs.time -= start_time
    assert logs.query('group == "A"').time.max() < delay_val / 2
    max_apply_time = (
        logs.query('group == "A" | group == "B"')
        .groupby("idx")
        .apply(lambda s: s.time.max() - s.time.min())
        .max()
    )
    assert max_apply_time > delay_val / 2
    max_delay_time = (
        logs.query('group == "B" | group == "C"')
        .groupby("idx")
        .apply(lambda s: s.time.max() - s.time.min())
        .max()
    )
    assert max_delay_time > delay_val / 2

    with pytest.raises(ValueError):
        await promise_test_ref.test_exceptions()
    with pytest.raises(ValueError):
        await promise_test_ref.test_yield_exceptions()

    with pytest.raises(asyncio.CancelledError):
        task = asyncio.create_task(promise_test_ref.test_cancel(5))
        await asyncio.sleep(0.1)
        task.cancel()
        await task
    call_log = await promise_test_ref.get_call_log()
    assert len(call_log) == 2
    assert call_log[1][0] - call_log[0][0] < 1


class ActorCannotDestroy(mo.Actor):
    async def __pre_destroy__(self):
        raise ValueError("Cannot destroy")


@pytest.mark.asyncio
@pytest.mark.parametrize("in_sub_pool", [True, False])
async def test_error_in_pre_destroy(actor_pool, in_sub_pool):
    pool = actor_pool

    strategy = None if not in_sub_pool else RandomSubPool()
    a = await mo.create_actor(
        ActorCannotDestroy, address=pool.external_address, strategy=strategy
    )
    with pytest.raises(ValueError, match="Cannot destroy"):
        await mo.destroy_actor(a)
