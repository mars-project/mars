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

import threading
import time

import pytest
import numpy as np

from .... import oscar as mo
from .... import remote as mr
from ....core import TileableGraph, TileableGraphBuilder
from ....resource import Resource
from ....utils import get_next_port
from ... import start_services, stop_services, NodeRole
from ...task.api import TaskAPI
from .. import SessionAPI, WebSessionAPI


@pytest.mark.parametrize("test_web", [False, True])
@pytest.mark.asyncio
async def test_session_service(test_web):
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        config = {
            "services": ["cluster", "session", "meta"],
            "cluster": {
                "backend": "fixed",
                "lookup_address": pool.external_address,
            },
            "meta": {"store": "dict"},
        }
        if test_web:
            config["services"] += ["web"]
            config["web"] = {"port": get_next_port()}

        await start_services(NodeRole.SUPERVISOR, config, address=pool.external_address)

        if not test_web:
            session_api = await SessionAPI.create(pool.external_address)
        else:
            session_api = WebSessionAPI(f'http://127.0.0.1:{config["web"]["port"]}')
        session_id = "test_session"
        session_address = await session_api.create_session(session_id)
        assert session_address == pool.external_address
        assert await session_api.has_session(session_id) is True
        assert (await session_api.get_sessions())[0].session_id == session_id
        if not test_web:
            assert await session_api.get_session_address(session_id) == session_address
        await session_api.delete_session(session_id)
        assert await session_api.has_session(session_id) is False
        assert await session_api.get_sessions() == []
        await session_api.delete_all_sessions()
        assert await session_api.has_session(session_id) is False

        await stop_services(NodeRole.SUPERVISOR, config, address=pool.external_address)


@pytest.mark.asyncio
async def test_get_last_idle_time():
    sv_pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    worker_pool = await mo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
        labels=["main"] + ["numa-0"] * 2,
        subprocess_start_method="spawn",
    )
    async with sv_pool, worker_pool:
        config = {
            "services": [
                "cluster",
                "session",
                "meta",
                "lifecycle",
                "scheduling",
                "subtask",
                "task",
                "mutable",
            ],
            "cluster": {
                "backend": "fixed",
                "lookup_address": sv_pool.external_address,
                "resource": {"numa-0": Resource(num_cpus=2)},
            },
            "meta": {"store": "dict"},
        }
        await start_services(
            NodeRole.SUPERVISOR, config, address=sv_pool.external_address
        )
        await start_services(
            NodeRole.WORKER, config, address=worker_pool.external_address
        )

        start_time = time.time()
        session_api = await SessionAPI.create(sv_pool.external_address)
        assert await session_api.get_last_idle_time() < start_time

        session_id = "test_session"
        await session_api.create_session(session_id)
        # check last idle time is not None
        last_idle_time = await session_api.get_last_idle_time(session_id)
        assert last_idle_time is not None
        assert await session_api.get_last_idle_time(session_id) == last_idle_time
        # submit a task
        task_api = await TaskAPI.create(session_id, sv_pool.external_address)

        def f1():
            return np.arange(5)

        def f2():
            return np.arange(5, 10)

        def f3(f1r, f2r):
            return np.concatenate([f1r, f2r]).sum()

        r1 = mr.spawn(f1)
        r2 = mr.spawn(f2)
        r3 = mr.spawn(f3, args=(r1, r2))

        graph = TileableGraph([r3.data])
        next(TileableGraphBuilder(graph).build())
        task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)
        await task_api.wait_task(task_id)
        task_result = await task_api.get_task_result(task_id)

        # the error is Actor b'StorageHandlerActor' does not exist
        assert task_result.error is not None

        # the last idle time is changed
        new_last_idle_time = await session_api.get_last_idle_time()
        assert new_last_idle_time is not None
        assert new_last_idle_time != last_idle_time
        assert await session_api.get_last_idle_time() == new_last_idle_time
        assert new_last_idle_time > last_idle_time

        # blocking task.
        def f4():
            import time

            time.sleep(10)

        r4 = mr.spawn(f4)
        graph = TileableGraph([r4.data])
        next(TileableGraphBuilder(graph).build())
        await task_api.submit_tileable_graph(graph, fuse_enabled=False)
        assert await session_api.get_last_idle_time() is None

        await stop_services(
            NodeRole.WORKER, config, address=worker_pool.external_address
        )
        await stop_services(
            NodeRole.SUPERVISOR, config, address=sv_pool.external_address
        )


@pytest.mark.asyncio
async def test_dmap():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        config = {
            "services": [
                "cluster",
                "session",
                "meta",
                "lifecycle",
                "scheduling",
                "subtask",
                "task",
                "mutable",
            ],
            "cluster": {
                "backend": "fixed",
                "lookup_address": pool.external_address,
            },
            "meta": {"store": "dict"},
        }
        await start_services(NodeRole.SUPERVISOR, config, address=pool.external_address)

        session_api = await SessionAPI.create(pool.external_address)

        session_id = "test_session"
        await session_api.create_session(session_id)
        lock = await session_api.create_remote_object(
            session_id, "my_lock", threading.Lock
        )
        await lock.acquire()
        lock = await session_api.get_remote_object(session_id, "my_lock")
        await lock.release()
        with pytest.raises(AttributeError):
            await lock.abc()
        await session_api.destroy_remote_object(session_id, "my_lock")

        await stop_services(NodeRole.SUPERVISOR, config, address=pool.external_address)
