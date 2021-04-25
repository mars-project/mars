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
import mars.tensor as mt
import mars.remote as mr
from mars.core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from mars.services.cluster import MockClusterAPI
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.storage import MockStorageAPI
from mars.services.task import Subtask, SubtaskStatus, SubtaskResult, new_task_id
from mars.services.task.supervisor.task_manager import TaskConfigurationActor, TaskManagerActor
from mars.services.task.worker.subtask import BandSubtaskManagerActor, SubtaskRunnerActor
from mars.utils import Timer


class FakeTaskManager(TaskManagerActor):
    def set_subtask_result(self, subtask_result: SubtaskResult):
        return


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
        await MockSessionAPI.create(
            pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        storage_api = await MockStorageAPI.create(session_id, pool.external_address)

        # create configuration
        await mo.create_actor(TaskConfigurationActor, dict(),
                              uid=TaskConfigurationActor.default_uid(),
                              address=pool.external_address)
        await mo.create_actor(
            FakeTaskManager, session_id,
            uid=FakeTaskManager.gen_uid(session_id),
            address=pool.external_address)
        manager = await mo.create_actor(
            BandSubtaskManagerActor, pool.external_address, 2,
            uid=BandSubtaskManagerActor.gen_uid('numa-0'),
            address=pool.external_address)

        yield pool, session_id, meta_api, storage_api, manager

        await MockStorageAPI.cleanup(pool.external_address)


def _gen_subtask(t, session_id):
    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())

    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())
    subtask = Subtask(new_task_id(), session_id,
                      new_task_id(), chunk_graph)

    return subtask


@pytest.mark.asyncio
async def test_subtask_success(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    a = mt.ones((10, 10), chunk_size=10)
    b = a + 1

    subtask = _gen_subtask(b, session_id)
    subtask_runner: SubtaskRunnerActor = await manager.get_free_slot()
    asyncio.create_task(subtask_runner.run_subtask(subtask))
    await asyncio.sleep(0)
    await subtask_runner.wait_subtask()
    result = await subtask_runner.get_subtask_result()
    assert result.status == SubtaskStatus.succeeded

    # check storage
    expected = np.ones((10, 10)) + 1
    result_key = subtask.chunk_graph.results[0].key
    result = await storage_api.get(result_key)
    np.testing.assert_array_equal(expected, result)

    # check meta
    chunk_meta = await meta_api.get_chunk_meta(result_key)
    assert chunk_meta is not None
    assert chunk_meta['bands'][0] == (pool.external_address, 'numa-0')
    assert await manager.is_slot_free(subtask_runner) is True


@pytest.mark.asyncio
async def test_subtask_failure(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    # test execution error
    with mt.errstate(divide='raise'):
        a = mt.ones((10, 10), chunk_size=10)
        c = a / 0

    subtask = _gen_subtask(c, session_id)
    subtask_runner: SubtaskRunnerActor = await manager.get_free_slot()
    await subtask_runner.run_subtask(subtask)
    result = await subtask_runner.get_subtask_result()
    assert result.status == SubtaskStatus.errored
    assert isinstance(result.error, FloatingPointError)
    assert await manager.is_slot_free(subtask_runner) is True


@pytest.mark.asyncio
async def test_cancel_subtask(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    def sleep(timeout: int):
        time.sleep(timeout)
        return timeout

    a = mr.spawn(sleep, 2)

    subtask = _gen_subtask(a, session_id)
    subtask_runner: SubtaskRunnerActor = await manager.get_free_slot()
    asyncio.create_task(subtask_runner.run_subtask(subtask))
    await asyncio.sleep(0.2)
    with Timer() as timer:
        # normal cancel by cancel asyncio Task
        await manager.free_slot(subtask_runner, timeout=5)
    # do not need to wait 5 sec
    assert timer.duration < 5
    assert await manager.is_slot_free(subtask_runner) is True

    b = mr.spawn(sleep, 100)

    subtask2 = _gen_subtask(b, session_id)
    subtask_runner: SubtaskRunnerActor = await manager.get_free_slot()
    asyncio.create_task(subtask_runner.run_subtask(subtask2))
    await asyncio.sleep(0.2)
    with Timer() as timer:
        # normal cancel by cancel asyncio Task
        aio_task = asyncio.create_task(manager.free_slot(subtask_runner, timeout=1))
        assert await manager.is_slot_free(subtask_runner) is False
        await aio_task
    # need 1 sec to reach timeout, then killing actor and wait for auto recovering
    # the time would not be over 5 sec
    assert timer.duration < 5
    assert await manager.is_slot_free(subtask_runner) is True
