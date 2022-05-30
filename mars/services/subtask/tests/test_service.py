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
import time

import numpy as np
import pytest

from .... import oscar as mo
from .... import tensor as mt
from .... import remote as mr
from ....core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from ....resource import Resource
from ....utils import Timer
from ... import start_services, stop_services, NodeRole
from ...meta import MetaAPI
from ...session import SessionAPI
from ...storage import MockStorageAPI
from ...task import new_task_id
from ...task.supervisor.manager import TaskManagerActor
from .. import SubtaskAPI, Subtask, SubtaskResult


class FakeTaskManager(TaskManagerActor):
    def set_subtask_result(self, subtask_result: SubtaskResult):
        return


def _gen_subtask(t, session_id):
    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())

    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())
    subtask = Subtask(new_task_id(), session_id, new_task_id(), chunk_graph)

    return subtask


@pytest.fixture
async def actor_pools():
    async def start_pool(is_worker: bool):
        if is_worker:
            kw = dict(
                n_process=2,
                labels=["main"] + ["numa-0"] * 2,
                subprocess_start_method="spawn",
            )
        else:
            kw = dict(n_process=0, subprocess_start_method="spawn")
        pool = await mo.create_actor_pool("127.0.0.1", **kw)
        await pool.start()
        return pool

    try:
        sv_pool, worker_pool = await asyncio.gather(start_pool(False), start_pool(True))
        yield sv_pool, worker_pool
    finally:
        await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.asyncio
async def test_subtask_service(actor_pools):
    sv_pool, worker_pool = actor_pools

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
        "scheduling": {},
        "subtask": {},
    }
    await start_services(NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(NodeRole.WORKER, config, address=worker_pool.external_address)

    session_id = "test_session"
    session_api = await SessionAPI.create(sv_pool.external_address)
    await session_api.create_session(session_id)
    ref = await mo.actor_ref(
        FakeTaskManager.gen_uid(session_id), address=sv_pool.external_address
    )
    await mo.destroy_actor(ref)
    await mo.create_actor(
        FakeTaskManager,
        session_id,
        uid=FakeTaskManager.gen_uid(session_id),
        address=sv_pool.external_address,
    )

    subtask_api = await SubtaskAPI.create(worker_pool.external_address)
    # create mock meta and storage APIs
    meta_api = await MetaAPI.create(session_id, sv_pool.external_address)
    storage_api = await MockStorageAPI.create(session_id, worker_pool.external_address)

    a = mt.ones((10, 10), chunk_size=10)
    b = a + 1

    subtask = _gen_subtask(b, session_id)
    assert "TensorAdd" in repr(subtask)
    await subtask_api.run_subtask_in_slot("numa-0", 0, subtask)

    # check storage
    expected = np.ones((10, 10)) + 1
    result_key = subtask.chunk_graph.results[0].key
    result = await storage_api.get(result_key)
    np.testing.assert_array_equal(expected, result)

    # check meta
    chunk_meta = await meta_api.get_chunk_meta(result_key)
    assert chunk_meta is not None
    assert chunk_meta["bands"][0] == (worker_pool.external_address, "numa-0")

    def sleep(timeout: int):
        time.sleep(timeout)
        return timeout

    b = mr.spawn(sleep, 1)

    subtask2 = _gen_subtask(b, session_id)
    asyncio.create_task(subtask_api.run_subtask_in_slot("numa-0", 0, subtask2))
    await asyncio.sleep(0.2)
    with Timer() as timer:
        # normal cancel by cancel asyncio Task
        await asyncio.wait_for(
            subtask_api.cancel_subtask_in_slot("numa-0", 0), timeout=2
        )
    # need 1 sec to reach timeout, then killing actor and wait for auto recovering
    # the time would not be over 5 sec
    assert timer.duration < 2

    await MockStorageAPI.cleanup(worker_pool.external_address)
    await stop_services(NodeRole.WORKER, config, address=worker_pool.external_address)
    await stop_services(NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
