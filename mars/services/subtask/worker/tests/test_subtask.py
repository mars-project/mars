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
import os
import sys
import time

import numpy as np
import pytest

import mars.oscar as mo
import mars.tensor as mt
import mars.remote as mr
from mars.core.context import get_context
from mars.core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from mars.services.cluster import MockClusterAPI
from mars.services.lifecycle import MockLifecycleAPI
from mars.services.meta import MockMetaAPI
from mars.services.scheduling import MockSchedulingAPI
from mars.services.session import MockSessionAPI
from mars.services.storage import MockStorageAPI
from mars.services.subtask import Subtask, SubtaskStatus, SubtaskResult
from mars.services.subtask.worker.manager import SubtaskManagerActor
from mars.services.subtask.worker.runner import SubtaskRunnerActor, SubtaskRunnerRef
from mars.services.task import new_task_id
from mars.services.task.supervisor.manager import TaskManagerActor, TaskConfigurationActor
from mars.utils import Timer


class FakeTaskManager(TaskManagerActor):
    def set_subtask_result(self, subtask_result: SubtaskResult):
        return


@pytest.fixture
async def actor_pool():
    start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
        if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', n_process=3,
                                      labels=['main'] + ['numa-0'] * 2 + ['io'],
                                      subprocess_start_method=start_method)

    async with pool:
        session_id = 'test_session'
        # create mock APIs
        await MockClusterAPI.create(pool.external_address, band_to_slots={'numa-0': 2})
        await MockSessionAPI.create(
            pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        await MockLifecycleAPI.create(session_id, pool.external_address)
        storage_api = await MockStorageAPI.create(session_id, pool.external_address)
        await MockSchedulingAPI.create(session_id, pool.external_address)

        # create configuration
        await mo.create_actor(TaskConfigurationActor, dict(),
                              uid=TaskConfigurationActor.default_uid(),
                              address=pool.external_address)
        await mo.create_actor(
            FakeTaskManager, session_id,
            uid=FakeTaskManager.gen_uid(session_id),
            address=pool.external_address)
        manager = await mo.create_actor(
            SubtaskManagerActor, None,
            uid=SubtaskManagerActor.default_uid(),
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
    subtask_runner: SubtaskRunnerRef = await mo.actor_ref(
        SubtaskRunnerActor.gen_uid('numa-0', 0), address=pool.external_address)
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
    assert await subtask_runner.is_runner_free() is True


@pytest.mark.asyncio
async def test_subtask_failure(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    # test execution error
    with mt.errstate(divide='raise'):
        a = mt.ones((10, 10), chunk_size=10)
        c = a / 0

    subtask = _gen_subtask(c, session_id)
    subtask_runner: SubtaskRunnerRef = await mo.actor_ref(
        SubtaskRunnerActor.gen_uid('numa-0', 0), address=pool.external_address)
    with pytest.raises(FloatingPointError):
        await subtask_runner.run_subtask(subtask)
    result = await subtask_runner.get_subtask_result()
    assert result.status == SubtaskStatus.errored
    assert isinstance(result.error, FloatingPointError)
    assert await subtask_runner.is_runner_free() is True


@pytest.mark.asyncio
async def test_cancel_subtask(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool
    subtask_runner: SubtaskRunnerRef = await mo.actor_ref(
        SubtaskRunnerActor.gen_uid('numa-0', 0), address=pool.external_address)

    def sleep(timeout: int):
        time.sleep(timeout)
        return timeout

    b = mr.spawn(sleep, 100)

    subtask = _gen_subtask(b, session_id)
    asyncio.create_task(subtask_runner.run_subtask(subtask))
    await asyncio.sleep(0.2)
    with Timer() as timer:
        # normal cancel by cancel asyncio Task
        aio_task = asyncio.create_task(asyncio.wait_for(
            subtask_runner.cancel_subtask(), timeout=1))
        assert await subtask_runner.is_runner_free() is False
        with pytest.raises(asyncio.TimeoutError):
            await aio_task
    # need 1 sec to reach timeout, then killing actor and wait for auto recovering
    # the time would not be over 5 sec
    assert timer.duration < 5

    async def wait_slot_restore():
        while True:
            try:
                assert await subtask_runner.is_runner_free() is True
            except (mo.ServerClosed, ConnectionRefusedError, mo.ActorNotExist):
                await asyncio.sleep(0.5)
            else:
                break

    await mo.kill_actor(subtask_runner)
    await wait_slot_restore()

    a = mr.spawn(sleep, 2)

    subtask2 = _gen_subtask(a, session_id)
    asyncio.create_task(subtask_runner.run_subtask(subtask2))
    await asyncio.sleep(0.2)
    with Timer() as timer:
        # normal cancel by cancel asyncio Task
        await asyncio.wait_for(subtask_runner.cancel_subtask(), timeout=6)
    # do not need to wait 10 sec
    assert timer.duration < 10
    assert await subtask_runner.is_runner_free() is True


@pytest.mark.asyncio
async def test_subtask_op_progress(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool
    subtask_runner: SubtaskRunnerRef = await mo.actor_ref(
        SubtaskRunnerActor.gen_uid('numa-0', 0), address=pool.external_address)

    def progress_sleep(interval: float, count: int):
        for idx in range(count):
            time.sleep(interval)
            get_context().set_progress((1 + idx) * 1.0 / count)

    b = mr.spawn(progress_sleep, args=(0.75, 2))

    subtask = _gen_subtask(b, session_id)
    aio_task = asyncio.create_task(subtask_runner.run_subtask(subtask))
    try:
        await asyncio.sleep(0.5)
        result = await subtask_runner.get_subtask_result()
        assert result.progress == 0.0

        await asyncio.sleep(0.75)
        result = await subtask_runner.get_subtask_result()
        assert result.progress == 0.5
    finally:
        await aio_task

    result = await subtask_runner.get_subtask_result()
    assert result.progress == 1.0
