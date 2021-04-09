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
import os
import sys
import time

import numpy as np
import pytest

import mars.oscar as mo
import mars.remote as mr
import mars.tensor as mt
from mars.core.graph import TileableGraph, TileableGraphBuilder
from mars.oscar.backends.allocate_strategy import MainPool
from mars.services.cluster import MockClusterAPI
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.storage.api import MockStorageApi
from mars.services.task.core import TaskStatus, TaskResult
from mars.services.task.supervisor.task_manager import TaskManagerActor
from mars.services.task.worker.subtask import BandSubtaskManagerActor
from mars.utils import Timer


@pytest.fixture
async def actor_pool():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', n_process=2,
                                      labels=[None] + ['numa-0'] * 2,
                                      subprocess_start_method=start_method)

    async with pool:
        session_id = 'test_session'
        # create mock APIs
        await MockClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        storage_api = await MockStorageApi.create(session_id, pool.external_address)

        # create task manager
        manager = await mo.create_actor(TaskManagerActor, session_id,
                                        uid=TaskManagerActor.gen_uid(session_id),
                                        address=pool.external_address,
                                        allocate_strategy=MainPool())

        # create band subtask manager
        await mo.create_actor(
            BandSubtaskManagerActor, pool.external_address, 2,
            uid=BandSubtaskManagerActor.gen_uid('numa-0'),
            address=pool.external_address)

        yield pool, session_id, meta_api, storage_api, manager

        await MockStorageApi.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_run_task(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    graph = TileableGraph([b.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert task_result.error is None

    result_tileables = (await manager.get_task_result_tileables(task_id))[0]
    for i, chunk in enumerate(result_tileables.chunks):
        result = await storage_api.get(chunk.key)
        if i == 0:
            expect = raw[:5, :5] + 1
        elif i == 1:
            expect = raw[:5, 5:] + 1
        elif i == 2:
            expect = raw[5:, :5] + 1
        else:
            expect = raw[5:, 5:] + 1
        np.testing.assert_array_equal(result, expect)


@pytest.mark.asyncio
async def test_cancel_task(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    def func():
        time.sleep(20)

    rs = [mr.spawn(func) for _ in range(10)]

    graph = TileableGraph([r.data for r in rs])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await asyncio.sleep(.5)

    with Timer() as timer:
        await manager.cancel_task(task_id)
        result = await manager.get_task_result(task_id)
        assert result.status == TaskStatus.terminated

    assert timer.duration < 10
