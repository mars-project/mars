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
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

import mars.dataframe as md
import mars.oscar as mo
import mars.remote as mr
import mars.tensor as mt
from mars.core import Tileable, TileableGraph, TileableGraphBuilder
from mars.oscar.backends.allocate_strategy import MainPool
from mars.services.cluster import MockClusterAPI
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.storage.api import StorageAPI, MockStorageAPI
from mars.services.task.core import TaskStatus, TaskResult
from mars.services.task.supervisor.task_manager import \
    TaskConfigurationActor, TaskManagerActor
from mars.services.task.worker.subtask import BandSubtaskManagerActor
from mars.utils import Timer, merge_chunks


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
        storage_api = await MockStorageAPI.create(session_id, pool.external_address)

        # create configuration
        await mo.create_actor(TaskConfigurationActor, dict(),
                              uid=TaskConfigurationActor.default_uid(),
                              address=pool.external_address)
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

        await MockStorageAPI.cleanup(pool.external_address)


async def _merge_data(fetch_tileable: Tileable,
                      storage_api: StorageAPI):
    gets = []
    for i, chunk in enumerate(fetch_tileable.chunks):
        gets.append(storage_api.get.delay(chunk.key))
    data = await storage_api.get.batch(*gets)
    index_data = [(c.index, d) for c, d
                  in zip(fetch_tileable.chunks, data)]
    return merge_chunks(index_data)


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
    assert await manager.get_task_progress(task_id) == 1.0

    result_tileables = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(result_tileables, storage_api)
    np.testing.assert_array_equal(result, raw + 1)


@pytest.mark.asyncio
async def test_error_task(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    with mt.errstate(divide='raise'):
        a = mt.ones((10, 10), chunk_size=10)
        c = a / 0

    graph = TileableGraph([c.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert task_result.error is not None
    assert isinstance(task_result.error, FloatingPointError)


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

    assert timer.duration < 15


@pytest.mark.asyncio
async def test_iterative_tiling(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    rs = np.random.RandomState(0)
    raw_a = rs.rand(10, 10)
    raw_b = rs.rand(10, 10)
    a = mt.tensor(raw_a, chunk_size=5)
    b = mt.tensor(raw_b, chunk_size=5)

    d = a[a[:, 0] < 3] + b[b[:, 0] < 3]
    graph = TileableGraph([d.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert task_result.error is None
    assert await manager.get_task_progress(task_id) == 1.0

    expect = raw_a[raw_a[:, 0] < 3] + raw_b[raw_b[:, 0] < 3]
    result_tileables = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(result_tileables, storage_api)
    np.testing.assert_array_equal(result, expect)


@pytest.mark.asyncio
async def test_shuffle(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    raw = np.random.rand(10, 10)
    raw2 = np.random.randint(10, size=(10,))
    a = mt.tensor(raw, chunk_size=5)
    b = mt.tensor(raw2, chunk_size=5)
    c = a[b]

    graph = TileableGraph([c.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert task_result.error is None
    assert await manager.get_task_progress(task_id) == 1.0

    expect = raw[raw2]
    result_tileables = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(result_tileables, storage_api)
    np.testing.assert_array_equal(result, expect)


@pytest.mark.asyncio
async def test_numexpr(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    raw = np.random.rand(10, 10)
    t = mt.tensor(raw, chunk_size=5)
    t2 = (t + 1) * 2 - 0.3

    graph = TileableGraph([t2.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph,
                                                  fuse_enabled=True)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert task_result.error is None
    assert await manager.get_task_progress(task_id) == 1.0

    expect = (raw + 1) * 2 - 0.3
    result_tileables = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(result_tileables, storage_api)
    np.testing.assert_array_equal(result, expect)


@pytest.mark.asyncio
async def test_optimization(actor_pool):
    pool, session_id, meta_api, storage_api, manager = actor_pool

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, 'test.csv')

        pdf = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                           'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                           'c': list('aabaaddce'),
                           'd': list('abaaaddce')})
        pdf.to_csv(file_path, index=False)

        df = md.read_csv(file_path)
        df2 = df.groupby('c').agg({'a': 'sum'})
        df3 = df[['b', 'a']]

        graph = TileableGraph([df2.data, df3.data])
        next(TileableGraphBuilder(graph).build())

        task_id = await manager.submit_tileable_graph(graph)
        assert isinstance(task_id, str)

        await manager.wait_task(task_id)
        task_result: TaskResult = await manager.get_task_result(task_id)

        assert task_result.status == TaskStatus.terminated
        assert task_result.error is None
        assert await manager.get_task_progress(task_id) == 1.0

        expect = pdf.groupby('c').agg({'a': 'sum'})
        result_tileables = (await manager.get_task_result_tileables(task_id))
        result1 = result_tileables[0]
        result = await _merge_data(result1, storage_api)
        np.testing.assert_array_equal(result, expect)

        expect = pdf[['b', 'a']]
        result2 = result_tileables[1]
        result = await _merge_data(result2, storage_api)
        np.testing.assert_array_equal(result, expect)
