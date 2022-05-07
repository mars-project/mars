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
import gc
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from ..... import dataframe as md
from ..... import oscar as mo
from ..... import remote as mr
from ..... import tensor as mt
from .....core import Tileable, TileableGraph, TileableGraphBuilder
from .....core.operand import Fetch
from .....oscar.backends.allocate_strategy import MainPool
from .....resource import Resource
from .....storage import StorageLevel
from .....utils import Timer, merge_chunks
from ....cluster import MockClusterAPI
from ....lifecycle import MockLifecycleAPI
from ....meta import MetaAPI, MockMetaAPI, MockWorkerMetaAPI
from ....scheduling import MockSchedulingAPI
from ....session import MockSessionAPI
from ....storage import StorageAPI, MockStorageAPI
from ....subtask import MockSubtaskAPI
from ....mutable import MockMutableAPI
from ...core import TaskStatus, TaskResult
from ...execution.api import Fetcher, ExecutionConfig
from ..manager import TaskConfigurationActor, TaskManagerActor


@pytest.fixture
async def actor_pool(request):
    param = getattr(request, "param", {})
    backend = param.get("backend", "mars")
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await mo.create_actor_pool(
        "127.0.0.1",
        n_process=3,
        labels=["main"] + ["numa-0"] * 2 + ["io"],
        subprocess_start_method=start_method,
    )

    async with pool:
        session_id = "test_session"
        # create mock APIs
        await MockClusterAPI.create(
            pool.external_address, band_to_resource={"numa-0": Resource(num_cpus=2)}
        )
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        await MockWorkerMetaAPI.create(session_id, pool.external_address)
        lifecycle_api = await MockLifecycleAPI.create(session_id, pool.external_address)
        storage_api = await MockStorageAPI.create(session_id, pool.external_address)
        await MockSchedulingAPI.create(session_id, pool.external_address)
        await MockSubtaskAPI.create(pool.external_address)
        await MockMutableAPI.create(session_id, pool.external_address)

        # create configuration
        config = ExecutionConfig.from_params(backend=backend, n_worker=1, n_cpu=2)
        await mo.create_actor(
            TaskConfigurationActor,
            dict(),
            config.get_execution_config(),
            uid=TaskConfigurationActor.default_uid(),
            address=pool.external_address,
        )
        # create task manager
        manager = await mo.create_actor(
            TaskManagerActor,
            session_id,
            uid=TaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
            allocate_strategy=MainPool(),
        )
        try:
            yield backend, pool, session_id, meta_api, lifecycle_api, storage_api, manager
        finally:
            await MockStorageAPI.cleanup(pool.external_address)
            await MockClusterAPI.cleanup(pool.external_address)
            await MockMutableAPI.cleanup(session_id, pool.external_address)


async def _merge_data(
    execution_backend: str,
    fetch_tileable: Tileable,
    meta_api: MetaAPI,
    storage_api: StorageAPI,
):
    async def _get_storage_api(band):
        return storage_api

    fetcher = Fetcher.create(execution_backend, get_storage_api=_get_storage_api)
    get_metas = []
    for chunk in fetch_tileable.chunks:
        get_metas.append(
            meta_api.get_chunk_meta.delay(chunk.key, fields=fetcher.required_meta_keys)
        )
    metas = await meta_api.get_chunk_meta.batch(*get_metas)
    for chunk, meta in zip(fetch_tileable.chunks, metas):
        await fetcher.append(chunk.key, meta)
    data = await fetcher.get()
    index_data = [(c.index, d) for c, d in zip(fetch_tileable.chunks, data)]
    return merge_chunks(index_data)


@pytest.mark.asyncio
async def test_run_task(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

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
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    np.testing.assert_array_equal(result, raw + 1)

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts([b.key]))[0] == 1
    assert (
        await lifecycle_api.get_chunk_ref_counts(
            [c.key for c in result_tileable.chunks]
        )
    ) == [1] * len(result_tileable.chunks)


@pytest.mark.asyncio
async def test_run_tasks_with_same_name(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1
    c = a * 2

    for t, e in zip([b, c], [raw + 1, raw * 2]):
        graph = TileableGraph([t.data])
        next(TileableGraphBuilder(graph).build())

        task_id = await manager.submit_tileable_graph(
            graph, task_name="my_task", fuse_enabled=False
        )
        assert isinstance(task_id, str)

        await manager.wait_task(task_id)
        task_result: TaskResult = await manager.get_task_result(task_id)

        assert task_result.status == TaskStatus.terminated
        if task_result.error is not None:
            raise task_result.error.with_traceback(task_result.traceback)
        assert await manager.get_task_progress(task_id) == 1.0

        result_tileable = (await manager.get_task_result_tileables(task_id))[0]
        result = await _merge_data(
            execution_backend, result_tileable, meta_api, storage_api
        )
        np.testing.assert_array_equal(result, e)


@pytest.mark.asyncio
async def test_error_task(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    with mt.errstate(divide="raise"):
        a = mt.ones((10, 10), chunk_size=5)
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

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts([c.key]))[0] == 0
    assert len(await lifecycle_api.get_all_chunk_ref_counts()) == 0


@pytest.mark.asyncio
async def test_cancel_task(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    def func():
        time.sleep(200)

    rs = [mr.spawn(func) for _ in range(10)]

    graph = TileableGraph([r.data for r in rs])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=False)
    assert isinstance(task_id, str)

    await asyncio.sleep(0.5)

    with Timer() as timer:
        await manager.cancel_task(task_id)
        await manager.wait_task(task_id)
        result = await manager.get_task_result(task_id)
        assert result.status == TaskStatus.terminated

    assert timer.duration < 25

    keys = [r.key for r in rs]
    del rs
    gc.collect()
    await asyncio.sleep(0.5)

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts(keys)) == [0] * len(keys)


@pytest.mark.asyncio
async def test_iterative_tiling(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

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
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    expect = raw_a[raw_a[:, 0] < 3] + raw_b[raw_b[:, 0] < 3]
    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    np.testing.assert_array_equal(result, expect)

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts([d.key]))[0] == 1
    assert (
        await lifecycle_api.get_chunk_ref_counts(
            [c.key for c in result_tileable.chunks]
        )
    ) == [1] * len(result_tileable.chunks)


@pytest.mark.asyncio
async def test_prune_in_iterative_tiling(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    raw = pd.DataFrame(np.random.RandomState(0).rand(1000, 10))
    df = md.DataFrame(raw, chunk_size=100)
    df2 = df.groupby(0).agg("sum")

    graph = TileableGraph([df2.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=True)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    expect = raw.groupby(0).agg("sum")
    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    pd.testing.assert_frame_equal(expect, result)

    subtask_graphs = await manager.get_subtask_graphs(task_id)
    assert len(subtask_graphs) == 2

    # the first subtask graph should have only 2 subtasks after pruning
    assert len(subtask_graphs[0]) == 2
    nodes = [
        n
        for st in subtask_graphs[0]
        for n in st.chunk_graph
        if not isinstance(n.op, Fetch)
    ]
    assert len(nodes) == 8
    result_nodes = [n for st in subtask_graphs[0] for n in st.chunk_graph.results]
    assert len(result_nodes) == 4
    assert all("GroupByAgg" in str(n.op) for n in result_nodes)

    # second subtask graph
    assert len(subtask_graphs[1]) == 6
    all_nodes = nodes + [
        n
        for st in subtask_graphs[1]
        for n in st.chunk_graph
        if not isinstance(n.op, Fetch)
    ]
    assert len(all_nodes) == 28
    assert len({n.key for n in all_nodes}) == 28

    df3 = df[df[0] < 1].rechunk(200)

    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=True)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    pd.testing.assert_frame_equal(raw, result)

    subtask_graphs = await manager.get_subtask_graphs(task_id)
    assert len(subtask_graphs) == 2

    # the first subtask graph
    assert len(subtask_graphs[0]) == 5
    nodes = [
        n
        for st in subtask_graphs[0]
        for n in st.chunk_graph
        if not isinstance(n.op, Fetch)
    ]
    assert len(nodes) == 40
    result_nodes = [n for st in subtask_graphs[0] for n in st.chunk_graph.results]
    assert len(result_nodes) == 10

    # second subtask graph
    assert len(subtask_graphs[1]) == 5
    all_nodes = nodes + [
        n
        for st in subtask_graphs[1]
        for n in st.chunk_graph
        if not isinstance(n.op, Fetch)
    ]
    assert len(all_nodes) == 45
    assert len({n.key for n in all_nodes}) == 45


@pytest.mark.asyncio
async def test_shuffle(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    rs = np.random.RandomState(0)
    raw = rs.rand(10, 10)
    raw2 = rs.randint(10, size=(10,))
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
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    expect = raw[raw2]
    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    np.testing.assert_array_equal(result, expect)

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts([c.key]))[0] == 1
    assert (
        await lifecycle_api.get_chunk_ref_counts(
            [c.key for c in result_tileable.chunks]
        )
    ) == [1] * len(result_tileable.chunks)
    await lifecycle_api.decref_tileables([c.key])
    ref_counts = await lifecycle_api.get_all_chunk_ref_counts()
    assert len(ref_counts) == 0

    # test if exists in storage
    assert len(await storage_api.list(level=StorageLevel.MEMORY)) == 0


@pytest.mark.asyncio
async def test_numexpr(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    raw = np.random.rand(10, 10)
    t = mt.tensor(raw, chunk_size=5)
    t2 = (t + 1) * 2 - 0.3

    graph = TileableGraph([t2.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(graph, fuse_enabled=True)
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)
    task_result: TaskResult = await manager.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)
    assert await manager.get_task_progress(task_id) == 1.0

    expect = (raw + 1) * 2 - 0.3
    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    np.testing.assert_array_equal(result, expect)

    # test ref counts
    assert (await lifecycle_api.get_tileable_ref_counts([t2.key]))[0] == 1
    assert (
        await lifecycle_api.get_chunk_ref_counts(
            [c.key for c in result_tileable.chunks]
        )
    ) == [1] * len(result_tileable.chunks)


@pytest.mark.asyncio
async def test_optimization(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test.csv")

        pdf = pd.DataFrame(
            {
                "a": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "b": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c": list("aabaaddce"),
                "d": list("abaaaddce"),
            }
        )
        pdf.to_csv(file_path, index=False)

        df = md.read_csv(file_path, incremental_index=True)
        df2 = df.groupby("c").agg({"a": "sum"})
        df3 = df[["b", "a"]]

        graph = TileableGraph([df2.data, df3.data])
        next(TileableGraphBuilder(graph).build())

        task_id = await manager.submit_tileable_graph(graph)
        assert isinstance(task_id, str)

        await manager.wait_task(task_id)
        task_result: TaskResult = await manager.get_task_result(task_id)

        assert task_result.status == TaskStatus.terminated
        if task_result.error is not None:
            raise task_result.error.with_traceback(task_result.traceback)
        assert await manager.get_task_progress(task_id) == 1.0

        expect = pdf.groupby("c").agg({"a": "sum"})
        result_tileables = await manager.get_task_result_tileables(task_id)
        result1 = result_tileables[0]
        result = await _merge_data(execution_backend, result1, meta_api, storage_api)
        np.testing.assert_array_equal(result, expect)

        expect = pdf[["b", "a"]]
        result2 = result_tileables[1]
        result = await _merge_data(execution_backend, result2, meta_api, storage_api)
        np.testing.assert_array_equal(result, expect)

        # test ref counts
        assert (await lifecycle_api.get_tileable_ref_counts([df3.key]))[0] == 1
        assert (
            await lifecycle_api.get_chunk_ref_counts(
                [c.key for c in result_tileables[1].chunks]
            )
        ) == [1] * len(result_tileables[1].chunks)

        # test ref counts
        assert (await lifecycle_api.get_tileable_ref_counts([df3.key]))[0] == 1
        assert (
            await lifecycle_api.get_chunk_ref_counts(
                [c.key for c in result_tileables[1].chunks]
            )
        ) == [1] * len(result_tileables[1].chunks)


@pytest.mark.asyncio
async def test_dump_subtask_graph(actor_pool):
    (
        execution_backend,
        pool,
        session_id,
        meta_api,
        lifecycle_api,
        storage_api,
        manager,
    ) = actor_pool

    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        {
            "c1": rs.randint(20, size=100),
            "c2": rs.choice(["a", "b", "c"], (100,)),
            "c3": rs.rand(100),
        }
    )
    mdf = md.DataFrame(raw, chunk_size=20)
    # groupby will generate multiple tasks
    r = mdf.groupby("c2").agg("sum")
    graph = TileableGraph([r.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await manager.submit_tileable_graph(
        graph,
        task_name="my_task",
        fuse_enabled=True,
        extra_config={"dump_subtask_graph": True},
    )
    assert isinstance(task_id, str)

    await manager.wait_task(task_id)

    result_tileable = (await manager.get_task_result_tileables(task_id))[0]
    result = await _merge_data(
        execution_backend, result_tileable, meta_api, storage_api
    )
    pd.testing.assert_frame_equal(result.sort_index(), raw.groupby("c2").agg("sum"))

    # read dot file
    file_path = os.path.join(tempfile.gettempdir(), f"mars-{task_id}")
    with open(file_path) as f:
        text = f.read()
        assert "style=bold" in text
        assert 'color="/spectral9/' in text
        for c in result_tileable.chunks:
            assert c.key[:5] in text
    os.remove(file_path)

    pdf_path = os.path.join(tempfile.gettempdir(), f"mars-{task_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
