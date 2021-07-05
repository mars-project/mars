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

import mars.oscar as mo
import mars.remote as mr
from mars.core import TileableGraph, TileableGraphBuilder
from mars.core.context import get_context
from mars.services import start_services, NodeRole
from mars.services.session import SessionAPI
from mars.services.storage import MockStorageAPI
from mars.services.web import WebActor
from mars.services.meta import MetaAPI
from mars.services.task import TaskAPI, TaskStatus, WebTaskAPI
from mars.utils import Timer


@pytest.fixture
async def actor_pools():
    async def start_pool(is_worker: bool):
        if is_worker:
            kw = dict(
                n_process=3,
                labels=['main'] + ['numa-0'] * 2 + ['io'],
                subprocess_start_method='spawn'
            )
        else:
            kw = dict(n_process=0,
                      subprocess_start_method='spawn')
        pool = await mo.create_actor_pool('127.0.0.1', **kw)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(
        start_pool(False), start_pool(True)
    )
    yield sv_pool, worker_pool
    await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=[False, True])
async def start_test_service(actor_pools, request):
    sv_pool, worker_pool = actor_pools

    config = {
        "services": ["cluster", "session", "lifecycle", "meta", "lifecycle", "scheduling",
                     "task", "subtask"],
        "cluster": {
            "backend": "fixed",
            "lookup_address": sv_pool.external_address,
            "resource": {"numa-0": 2}
        },
        "meta": {
            "store": "dict"
        },
        "scheduling": {},
        "task": {},
    }
    if request:
        config['services'].append('web')

    await start_services(
        NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(
        NodeRole.WORKER, config, address=worker_pool.external_address)

    session_id = 'test_session'
    session_api = await SessionAPI.create(sv_pool.external_address)
    await session_api.create_session(session_id)

    if not request:
        task_api = await TaskAPI.create(session_id,
                                        sv_pool.external_address)
    else:
        web_actor = await mo.actor_ref(WebActor.default_uid(),
                                       address=sv_pool.external_address)
        web_address = await web_actor.get_web_address()
        task_api = WebTaskAPI(session_id, web_address)

    assert await task_api.get_task_results() == []

    # create mock meta and storage APIs
    _ = await MetaAPI.create(session_id,
                             sv_pool.external_address)
    storage_api = await MockStorageAPI.create(session_id,
                                              worker_pool.external_address)

    try:
        yield task_api, storage_api
    finally:
        await MockStorageAPI.cleanup(worker_pool.external_address)


@pytest.mark.asyncio
async def test_task_execution(start_test_service):
    task_api, storage_api = start_test_service

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
    assert await task_api.get_last_idle_time() is None
    assert isinstance(task_id, str)

    await task_api.wait_task(task_id)
    task_result = await task_api.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated
    assert await task_api.get_last_idle_time() is not None
    if task_result.error is not None:
        raise task_result.error.with_traceback(task_result.traceback)

    result_tileable = (await task_api.get_fetch_tileables(task_id))[0]
    data_key = result_tileable.chunks[0].key
    assert await storage_api.get(data_key) == 45


@pytest.mark.asyncio
async def test_task_cancel(start_test_service):
    task_api, storage_api = start_test_service

    # test job cancel
    def f1():
        time.sleep(100)

    rs = [mr.spawn(f1) for _ in range(10)]

    graph = TileableGraph([r.data for r in rs])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)
    await asyncio.sleep(.5)
    with Timer() as timer:
        await task_api.cancel_task(task_id)
        result = await task_api.get_task_result(task_id)
        assert result.status == TaskStatus.terminated
    assert timer.duration < 20
    await asyncio.sleep(.1)
    assert await task_api.get_last_idle_time() is not None

    results = await task_api.get_task_results(progress=True)
    assert all(result.status == TaskStatus.terminated for result in results)


@pytest.mark.asyncio
async def test_task_progress(start_test_service):
    task_api, storage_api = start_test_service

    def f1(interval: float, count: int):
        for idx in range(count):
            time.sleep(interval)
            get_context().set_progress((1 + idx) * 1.0 / count)

    r = mr.spawn(f1, args=(0.75, 2))

    graph = TileableGraph([r.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await asyncio.sleep(0.5)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 0.0

    await asyncio.sleep(0.75)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 0.5

    await task_api.wait_task(task_id, timeout=10)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 1.0
