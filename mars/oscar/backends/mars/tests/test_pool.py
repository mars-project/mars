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
import re
import sys
import time

import pytest

from .....tests.core import mock
from .....utils import get_next_port
from .... import create_actor_ref, Actor, kill_actor
from ....context import get_context
from ....errors import NoIdleSlot, ActorNotExist, ServerClosed
from ...allocate_strategy import (
    AddressSpecified,
    IdleLabel,
    MainPool,
    RandomSubPool,
    ProcessIndex,
)
from ...config import ActorPoolConfig
from ...message import (
    new_message_id,
    CreateActorMessage,
    DestroyActorMessage,
    HasActorMessage,
    ActorRefMessage,
    SendMessage,
    TellMessage,
    ControlMessage,
    CancelMessage,
    ErrorMessage,
    ControlMessageType,
    MessageType,
)
from ...pool import create_actor_pool
from ...router import Router
from ..pool import MainActorPool, SubActorPool


class _CannotBeUnpickled:
    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        raise RuntimeError("cannot unpickle")


class TestActor(Actor):
    __test__ = False

    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val
        return self.value

    async def add_other(self, ref, val):
        self.value += await ref.add(val)
        return self.value

    async def sleep(self, second):
        try:
            await asyncio.sleep(second)
            return self.value
        except asyncio.CancelledError:
            return self.value + 1

    def return_cannot_unpickle(self):
        return _CannotBeUnpickled()


def _add_pool_conf(
    config: ActorPoolConfig,
    process_index: int,
    label: str,
    internal_address: str,
    external_address: str,
    env: dict = None,
):
    if sys.platform.startswith("win"):
        config.add_pool_conf(
            process_index, label, external_address, external_address, env=env
        )
    else:
        config.add_pool_conf(
            process_index, label, internal_address, external_address, env=env
        )


def _raise_if_error(message):
    if message.message_type == MessageType.error:
        raise message.error.with_traceback(message.traceback)


@pytest.fixture(autouse=True)
def clear_routers():
    yield
    Router.set_instance(None)


@pytest.mark.asyncio
@mock.patch("mars.oscar.backends.mars.pool.SubActorPool.notify_main_pool_to_create")
@mock.patch("mars.oscar.backends.mars.pool.SubActorPool.notify_main_pool_to_destroy")
async def test_sub_actor_pool(notify_main_pool_to_create, notify_main_pool_to_destroy):
    notify_main_pool_to_create.return_value = None
    notify_main_pool_to_destroy.return_value = None
    config = ActorPoolConfig()

    ext_address0 = f"127.0.0.1:{get_next_port()}"
    ext_address1 = f"127.0.0.1:{get_next_port()}"
    _add_pool_conf(config, 0, "main", "unixsocket:///0", ext_address0)
    _add_pool_conf(config, 1, "sub", "unixsocket:///1", ext_address1)

    pool = await SubActorPool.create({"actor_pool_config": config, "process_index": 1})
    await pool.start()

    try:
        create_actor_message = CreateActorMessage(
            new_message_id(),
            TestActor,
            b"test",
            tuple(),
            dict(),
            AddressSpecified(pool.external_address),
        )
        message = await pool.create_actor(create_actor_message)
        assert message.message_type == MessageType.result
        actor_ref = message.result
        assert actor_ref.address == pool.external_address
        assert actor_ref.uid == b"test"

        has_actor_message = HasActorMessage(new_message_id(), actor_ref)
        assert (await pool.has_actor(has_actor_message)).result is True

        actor_ref_message = ActorRefMessage(new_message_id(), actor_ref)
        assert (await pool.actor_ref(actor_ref_message)).result == actor_ref

        tell_message = TellMessage(
            new_message_id(), actor_ref, ("add", 0, (1,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None

        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, (3,), dict())
        )
        message = await pool.send(send_message)
        assert message.result == 4

        # test error message
        # type mismatch
        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, ("3",), dict())
        )
        result = await pool.send(send_message)
        assert result.message_type == MessageType.error
        assert isinstance(result.error, TypeError)

        send_message = SendMessage(
            new_message_id(),
            create_actor_ref(actor_ref.address, "non_exist"),
            ("add", 0, (3,), dict()),
        )
        result = await pool.send(send_message)
        assert isinstance(result.error, ActorNotExist)

        # test send message and cancel it
        send_message = SendMessage(
            new_message_id(), actor_ref, ("sleep", 0, (20,), dict())
        )
        result_task = asyncio.create_task(pool.send(send_message))
        await asyncio.sleep(0)
        start = time.time()
        cancel_message = CancelMessage(
            new_message_id(), actor_ref.address, send_message.message_id
        )
        cancel_task = asyncio.create_task(pool.cancel(cancel_message))
        result = await asyncio.wait_for(cancel_task, 3)
        assert result.message_type == MessageType.result
        assert result.result is True
        result = await result_task
        # test time
        assert time.time() - start < 3
        assert result.message_type == MessageType.result
        assert result.result == 5

        # test processing message on background
        async with await pool.router.get_client(pool.external_address) as client:
            send_message = SendMessage(
                new_message_id(), actor_ref, ("add", 0, (5,), dict())
            )
            await client.send(send_message)
            result = await client.recv()
            _raise_if_error(result)
            assert result.result == 9

            send_message = SendMessage(
                new_message_id(), actor_ref, ("add", 0, ("5",), dict())
            )
            await client.send(send_message)
            result = await client.recv()
            assert isinstance(result.error, TypeError)

        destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref)
        message = await pool.destroy_actor(destroy_actor_message)
        assert message.result == actor_ref.uid

        # send destroy failed
        message = await pool.destroy_actor(destroy_actor_message)
        assert isinstance(message.error, ActorNotExist)

        message = await pool.has_actor(has_actor_message)
        assert not message.result

        # test sync config
        _add_pool_conf(
            config, 1, "sub", "unixsocket:///1", f"127.0.0.1:{get_next_port()}"
        )
        sync_config_message = ControlMessage(
            new_message_id(), "", ControlMessageType.sync_config, config
        )
        message = await pool.handle_control_command(sync_config_message)
        assert message.result is True

        # test get config
        get_config_message = ControlMessage(
            new_message_id(), "", ControlMessageType.get_config, None
        )
        message = await pool.handle_control_command(get_config_message)
        config2 = message.result
        assert config.as_dict() == config2.as_dict()

        assert pool.router._mapping == Router.get_instance()._mapping
        assert (
            pool.router._curr_external_addresses
            == Router.get_instance()._curr_external_addresses
        )

        stop_message = ControlMessage(
            new_message_id(), "", ControlMessageType.stop, None
        )
        message = await pool.handle_control_command(stop_message)
        assert message.result is True

        await pool.join(0.05)
        assert pool.stopped
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_fail_when_create_subpool():
    config = ActorPoolConfig()
    my_label = "computation"
    main_address = f"127.0.0.1:{get_next_port()}"
    port = get_next_port()
    _add_pool_conf(config, 0, "main", "unixsocket:///0", main_address)

    # use the same port for sub pools, will raise `OSError` with "address already in use"
    _add_pool_conf(
        config, 1, my_label, "unixsocket:///1", f"127.0.0.1:{port}", env={"my_env": "1"}
    )
    _add_pool_conf(config, 2, my_label, "unixsocket:///2", f"127.0.0.1:{port}")

    with pytest.raises(OSError):
        await MainActorPool.create({"actor_pool_config": config})


@pytest.mark.asyncio
async def test_main_actor_pool():
    config = ActorPoolConfig()
    my_label = "computation"
    main_address = f"127.0.0.1:{get_next_port()}"
    _add_pool_conf(config, 0, "main", "unixsocket:///0", main_address)
    _add_pool_conf(
        config,
        1,
        my_label,
        "unixsocket:///1",
        f"127.0.0.1:{get_next_port()}",
        env={"my_env": "1"},
    )
    _add_pool_conf(
        config, 2, my_label, "unixsocket:///2", f"127.0.0.1:{get_next_port()}"
    )

    strategy = IdleLabel(my_label, "my_test")

    async with await MainActorPool.create({"actor_pool_config": config}) as pool:
        create_actor_message = CreateActorMessage(
            new_message_id(), TestActor, b"test", tuple(), dict(), MainPool()
        )
        message = await pool.create_actor(create_actor_message)
        actor_ref = message.result
        assert actor_ref.address == main_address

        create_actor_message1 = CreateActorMessage(
            new_message_id(), TestActor, b"test1", tuple(), dict(), strategy
        )
        message1 = await pool.create_actor(create_actor_message1)
        actor_ref1 = message1.result
        assert actor_ref1.address in config.get_external_addresses(my_label)

        create_actor_message2 = CreateActorMessage(
            new_message_id(), TestActor, b"test2", tuple(), dict(), strategy
        )
        message2 = await pool.create_actor(create_actor_message2)
        actor_ref2 = message2.result
        assert actor_ref2.address in config.get_external_addresses(my_label)
        assert actor_ref2.address != actor_ref1.address

        create_actor_message3 = CreateActorMessage(
            new_message_id(), TestActor, b"test3", tuple(), dict(), strategy
        )
        message3 = await pool.create_actor(create_actor_message3)
        # no slot to allocate the same label
        assert isinstance(message3.error, NoIdleSlot)

        has_actor_message = HasActorMessage(
            new_message_id(), create_actor_ref(main_address, b"test2")
        )
        assert (await pool.has_actor(has_actor_message)).result is True

        actor_ref_message = ActorRefMessage(
            new_message_id(), create_actor_ref(main_address, b"test2")
        )
        assert (await pool.actor_ref(actor_ref_message)).result == actor_ref2

        # tell
        tell_message = TellMessage(
            new_message_id(), actor_ref1, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None

        # send
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("add", 0, (4,), dict())
        )
        assert (await pool.send(send_message)).result == 6

        # test error message
        # type mismatch
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("add", 0, ("3",), dict())
        )
        result = await pool.send(send_message)
        assert isinstance(result.error, TypeError)

        # send and tell to main process
        tell_message = TellMessage(
            new_message_id(), actor_ref, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None
        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, (4,), dict())
        )
        assert (await pool.send(send_message)).result == 6

        # send and cancel
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("sleep", 0, (20,), dict())
        )
        result_task = asyncio.create_task(pool.send(send_message))
        start = time.time()
        cancel_message = CancelMessage(
            new_message_id(), actor_ref1.address, send_message.message_id
        )
        cancel_task = asyncio.create_task(pool.cancel(cancel_message))
        result = await asyncio.wait_for(cancel_task, 3)
        assert result.message_type == MessageType.result
        assert result.result is True
        result = await result_task
        assert time.time() - start < 3
        assert result.message_type == MessageType.result
        assert result.result == 7

        # destroy
        destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref1)
        message = await pool.destroy_actor(destroy_actor_message)
        assert message.result == actor_ref1.uid

        tell_message = TellMessage(
            new_message_id(), actor_ref1, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert isinstance(message, ErrorMessage)

        # destroy via connecting to sub pool directly
        async with await pool.router.get_client(
            config.get_external_addresses()[-1]
        ) as client:
            destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref2)
            await client.send(destroy_actor_message)
            result = await client.recv()
            _raise_if_error(result)
            assert result.result == actor_ref2.uid

        # test sync config
        config.add_pool_conf(
            3, "sub", "unixsocket:///3", f"127.0.0.1:{get_next_port()}"
        )
        sync_config_message = ControlMessage(
            new_message_id(),
            pool.external_address,
            ControlMessageType.sync_config,
            config,
        )
        message = await pool.handle_control_command(sync_config_message)
        assert message.result is True

        # test get config
        get_config_message = ControlMessage(
            new_message_id(),
            config.get_external_addresses()[1],
            ControlMessageType.get_config,
            None,
        )
        message = await pool.handle_control_command(get_config_message)
        config2 = message.result
        assert config.as_dict() == config2.as_dict()

    assert pool.stopped


@pytest.mark.asyncio
async def test_create_actor_pool():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
    )

    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        assert await actor_ref.add(1) == 4
        assert (await ctx.has_actor(actor_ref)) is True
        assert (await ctx.actor_ref(actor_ref)) == actor_ref
        # test cancel
        task = asyncio.create_task(actor_ref.sleep(20))
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False
        for f in actor_ref.add, ctx.actor_ref, ctx.destroy_actor:
            with pytest.raises(ActorNotExist):
                await f(actor_ref)

        # actor on sub pool
        actor_ref1 = await ctx.create_actor(
            TestActor, uid="test-main", address=pool.external_address
        )
        actor_ref2 = await ctx.create_actor(
            TestActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=RandomSubPool(),
        )
        assert (
            await ctx.actor_ref(uid="test-2", address=actor_ref2.address)
        ) == actor_ref2
        main_ref = await ctx.actor_ref(uid="test-main", address=actor_ref2.address)
        assert main_ref.address == pool.external_address
        main_ref = await ctx.actor_ref(actor_ref1)
        assert main_ref.address == pool.external_address
        assert actor_ref2.address != actor_ref.address
        assert await actor_ref2.add(3) == 3
        assert await actor_ref2.add(1) == 4
        with pytest.raises(RuntimeError):
            await actor_ref2.return_cannot_unpickle()
        assert (await ctx.has_actor(actor_ref2)) is True
        assert (await ctx.actor_ref(actor_ref2)) == actor_ref2
        # test cancel
        task = asyncio.create_task(actor_ref2.sleep(20))
        start = time.time()
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        assert time.time() - start < 3
        await ctx.destroy_actor(actor_ref2)
        assert (await ctx.has_actor(actor_ref2)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
async def test_errors():
    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, labels=["a"]
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            f"127.0.0.1:{get_next_port()}",
            pool_cls=MainActorPool,
            n_process=1,
            ports=[get_next_port(), get_next_port()],
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, ports=[get_next_port()]
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, auto_recover="illegal"
        )


@pytest.mark.asyncio
async def test_server_closed():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        auto_recover=False,
    )

    ctx = get_context()

    async with pool:
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )

        # check if error raised normally when subprocess killed
        task = asyncio.create_task(actor_ref.sleep(10))
        await asyncio.sleep(0)

        # kill subprocess 1
        process = list(pool._sub_processes.values())[0]
        process.kill()
        process.join()

        with pytest.raises(ServerClosed):
            # process already been killed,
            # ServerClosed will be raised
            await task

        assert not process.is_alive()

    with pytest.raises(RuntimeError):
        await pool.start()

    # test server unreachable
    with pytest.raises(ConnectionError):
        await ctx.has_actor(actor_ref)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform.startswith("win"), reason="skip under Windows")
@pytest.mark.parametrize("auto_recover", [False, True, "actor", "process"])
async def test_auto_recover(auto_recover):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    recovered = asyncio.Event()

    def on_process_recover(*_):
        recovered.set()

    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        auto_recover=auto_recover,
        on_process_recover=on_process_recover,
    )

    async with pool:
        ctx = get_context()

        # wait for recover of main pool always returned immediately
        await ctx.wait_actor_pool_recovered(
            pool.external_address, pool.external_address
        )

        # create actor on main
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=MainPool()
        )

        with pytest.raises(ValueError):
            # cannot kill actors on main pool
            await kill_actor(actor_ref)

        # create actor
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )
        # kill_actor will cause kill corresponding process
        await ctx.kill_actor(actor_ref)

        if auto_recover:
            # process must have been killed
            await ctx.wait_actor_pool_recovered(
                actor_ref.address, pool.external_address
            )
            assert recovered.is_set()

            expect_has_actor = True if auto_recover in ["actor", True] else False
            assert await ctx.has_actor(actor_ref) is expect_has_actor
        else:
            with pytest.raises((ServerClosed, ConnectionError)):
                await ctx.has_actor(actor_ref)


@pytest.mark.parametrize(
    "exception_config",
    [
        (Exception("recover exception"), False),
        (asyncio.CancelledError("cancel monitor"), True),
    ],
)
@pytest.mark.asyncio
async def test_monitor_sub_pool_exception(exception_config):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    recovered = asyncio.Event()
    exception, done = exception_config

    def on_process_recover(*_):
        recovered.set()
        raise exception

    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        on_process_recover=on_process_recover,
    )

    async with pool:
        ctx = get_context()
        task = await pool.start_monitor()

        # create actor
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )
        # kill_actor will cause kill corresponding process
        await ctx.kill_actor(actor_ref)

        await recovered.wait()
        assert task.done() is done


@pytest.mark.asyncio
async def test_two_pools():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )

    ctx = get_context()

    pool1 = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
    )
    pool2 = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
    )

    def is_interprocess_address(addr):
        if sys.platform.startswith("win"):
            return re.match(r"127\.0\.0\.1:\d+", addr)
        else:
            return addr.startswith("unixsocket://")

    try:
        actor_ref1 = await ctx.create_actor(
            TestActor, address=pool1.external_address, allocate_strategy=MainPool()
        )
        assert actor_ref1.address == pool1.external_address
        assert await actor_ref1.add(1) == 1
        assert (
            Router.get_instance()
            .get_internal_address(actor_ref1.address)
            .startswith("dummy://")
        )

        actor_ref2 = await ctx.create_actor(
            TestActor, address=pool1.external_address, allocate_strategy=RandomSubPool()
        )
        assert actor_ref2.address in pool1._config.get_external_addresses()[1:]
        assert await actor_ref2.add(3) == 3
        assert is_interprocess_address(
            Router.get_instance().get_internal_address(actor_ref2.address)
        )

        actor_ref3 = await ctx.create_actor(
            TestActor, address=pool2.external_address, allocate_strategy=MainPool()
        )
        assert actor_ref3.address == pool2.external_address
        assert await actor_ref3.add(5) == 5
        assert (
            Router.get_instance()
            .get_internal_address(actor_ref3.address)
            .startswith("dummy://")
        )

        actor_ref4 = await ctx.create_actor(
            TestActor, address=pool2.external_address, allocate_strategy=RandomSubPool()
        )
        assert actor_ref4.address in pool2._config.get_external_addresses()[1:]
        assert await actor_ref4.add(7) == 7
        assert is_interprocess_address(
            Router.get_instance().get_internal_address(actor_ref4.address)
        )

        assert await actor_ref2.add_other(actor_ref4, 3) == 13
    finally:
        await pool1.stop()
        await pool2.stop()


@pytest.mark.asyncio
async def test_parallel_allocate_idle_label():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        labels=[None, "my_label", "my_label"],
    )

    class _Actor(Actor):
        def get_pid(self):
            return os.getpid()

    async with pool:
        ctx = get_context()
        strategy = IdleLabel("my_label", "tests")
        tasks = [
            ctx.create_actor(
                _Actor, allocate_strategy=strategy, address=pool.external_address
            ),
            ctx.create_actor(
                _Actor, allocate_strategy=strategy, address=pool.external_address
            ),
        ]
        refs = await asyncio.gather(*tasks)
        # outputs identical process ids, while the result should be different
        assert len({await ref.get_pid() for ref in refs}) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logging_conf",
    [
        {
            "file": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test-logging.conf"
            )
        },
        {"level": logging.DEBUG},
        {"level": logging.DEBUG, "format": "%(asctime)s %(message)s"},
    ],
)
async def test_logging_config(logging_conf):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=1,
        subprocess_start_method=start_method,
        labels=[None, "my_label"],
        logging_conf=logging_conf,
    )

    class _Actor(Actor):
        def get_logger_level(self):
            logger = logging.getLogger(__name__)
            return logger.getEffectiveLevel()

    async with pool:
        ctx = get_context()
        strategy = IdleLabel("my_label", "tests")
        ref = await ctx.create_actor(
            _Actor, allocate_strategy=strategy, address=pool.external_address
        )
        assert await ref.get_logger_level() == logging.DEBUG


@pytest.mark.asyncio
async def test_ref_sub_pool_actor():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=1,
        subprocess_start_method=start_method,
    )

    async with pool:
        ctx = get_context()
        ref1 = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=RandomSubPool()
        )
        sub_address = ref1.address
        ref2 = await ctx.create_actor(TestActor, address=sub_address)
        ref2_main = await ctx.actor_ref(ref2.uid, address=pool.external_address)
        assert ref2_main.address == sub_address

        await ctx.destroy_actor(create_actor_ref(pool.external_address, ref2.uid))
        assert not await ctx.has_actor(
            create_actor_ref(pool.external_address, ref2.uid)
        )
        assert not await ctx.has_actor(create_actor_ref(sub_address, ref2.uid))
