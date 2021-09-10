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

from .... import dataframe as md
from .... import oscar as mo
from .... import remote as mr
from ....core import TileableGraph, TileableGraphBuilder
from ....core.context import get_context
from ....utils import Timer
from ... import start_services, stop_services, NodeRole
from ...session import SessionAPI
from ...storage import MockStorageAPI
from ...subtask import SubtaskStatus, SubtaskDisplayNodeStatus
from ...web import WebActor
from ...meta import MetaAPI
from .. import TaskAPI, TaskStatus, WebTaskAPI
from ..errors import TaskNotExist
from mars.services import subtask


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
    try:
        yield sv_pool, worker_pool
    finally:
        await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=[False, True])
async def start_test_service(actor_pools, request):
    sv_pool, worker_pool = actor_pools

    config = {
        "services": ["cluster", "session", "meta", "lifecycle",
                     "scheduling", "subtask", "task"],
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

    if not request.param:
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
        yield sv_pool.external_address, task_api, storage_api
    finally:
        await MockStorageAPI.cleanup(worker_pool.external_address)
        await stop_services(NodeRole.WORKER, config, worker_pool.external_address)
        await stop_services(NodeRole.SUPERVISOR, config, sv_pool.external_address)


@pytest.mark.asyncio
async def test_task_execution(start_test_service):
    _sv_pool_address, task_api, storage_api = start_test_service

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
async def test_task_error(start_test_service):
    _sv_pool_address, task_api, storage_api = start_test_service

    # test job cancel
    def f1():
        raise SystemError

    rs = [mr.spawn(f1) for _ in range(10)]

    graph = TileableGraph([r.data for r in rs])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await task_api.wait_task(task_id, timeout=10)
    results = await task_api.get_task_results(progress=True)
    assert type(results[0].error) is SystemError


@pytest.mark.asyncio
async def test_task_cancel(start_test_service):
    _sv_pool_address, task_api, storage_api = start_test_service

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


class _ProgressController:
    def __init__(self):
        self._step_event = asyncio.Event()

    async def wait(self):
        await self._step_event.wait()
        self._step_event.clear()

    def set(self):
        self._step_event.set()


@pytest.mark.asyncio
async def test_task_progress(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service

    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, 'progress_controller', _ProgressController)

    def f1(count: int):
        progress_controller = get_context().get_remote_object('progress_controller')
        for idx in range(count):
            progress_controller.wait()
            get_context().set_progress((1 + idx) * 1.0 / count)

    r = mr.spawn(f1, args=(2,))

    graph = TileableGraph([r.data])
    next(TileableGraphBuilder(graph).build())

    await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await asyncio.sleep(0.2)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 0.0

    await ref.set()
    await asyncio.sleep(1)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 0.5

    await ref.set()
    await asyncio.sleep(1)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 1.0


@pytest.mark.asyncio
async def test_get_tileable_graph(start_test_service):
    _sv_pool_address, task_api, storage_api = start_test_service

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

    with pytest.raises(TaskNotExist):
        await task_api.get_tileable_graph_as_json('non_exist')

    tileable_detail = await task_api.get_tileable_graph_as_json(task_id)

    num_tileable = len(tileable_detail.get('tileables'))
    num_dependencies = len(tileable_detail.get('dependencies'))
    assert num_tileable > 0
    assert num_dependencies <= (num_tileable / 2) * (num_tileable / 2)

    assert (num_tileable == 1 and num_dependencies == 0) or (num_tileable > 1 and num_dependencies > 0)

    graph_nodes = []
    graph_dependencies = []
    for node in graph.iter_nodes():
        graph_nodes.append(node.key)

        for node_successor in graph.iter_successors(node):
            graph_dependencies.append({
                'fromTileableId': node.key,
                'toTileableId': node_successor.key,
                'linkType': 0,
            })

    for tileable in tileable_detail.get('tileables'):
        graph_nodes.remove(tileable.get('tileableId'))

    assert len(graph_nodes) == 0

    for i in range(num_dependencies):
        dependency = tileable_detail.get('dependencies')[i]
        assert graph_dependencies[i] == dependency


@pytest.mark.asyncio
async def test_get_tileable_details(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service

    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, 'progress_controller', _ProgressController)

    with pytest.raises(TaskNotExist):
        await task_api.get_tileable_details('non_exist')

    def f(*_args, raises=False):
        get_context().set_progress(0.5)
        if raises:
            raise ValueError
        progress_controller = get_context().get_remote_object('progress_controller')
        progress_controller.wait()
        get_context().set_progress(1.0)

    # test non-fused DAGs
    r1 = mr.spawn(f)
    r2 = mr.spawn(f, args=(r1, 0))
    r3 = mr.spawn(f, args=(r1, 1))

    graph = TileableGraph([r2.data, r3.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    def _get_fields(details, field, wrapper=None):
        rs = [r1, r2, r3]
        ret = [details[r.key][field] for r in rs]
        if wrapper:
            ret = [wrapper(v) for v in ret]
        return ret

    await asyncio.sleep(1)
    details = await task_api.get_tileable_details(task_id)
    assert _get_fields(details, 'progress') == [0.5, 0.0, 0.0]
    assert _get_fields(details, 'status', SubtaskStatus) \
        == [SubtaskStatus.running] + [SubtaskStatus.pending] * 2

    await ref.set()
    await asyncio.sleep(1)
    details = await task_api.get_tileable_details(task_id)
    assert _get_fields(details, 'progress') == [1.0, 0.5, 0.5]
    assert _get_fields(details, 'status', SubtaskStatus) \
        == [SubtaskStatus.succeeded] + [SubtaskStatus.running] * 2

    await ref.set()
    await task_api.wait_task(task_id)

    # test fused DAGs
    r5 = mr.spawn(f, args=(0,))
    r6 = mr.spawn(f, args=(r5,))

    graph = TileableGraph([r6.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)

    await asyncio.sleep(1)
    details = await task_api.get_tileable_details(task_id)
    assert details[r5.key]['progress'] == details[r6.key]['progress'] == 0.25

    await ref.set()
    await asyncio.sleep(0.1)
    await ref.set()
    await task_api.wait_task(task_id)

    # test raises
    r7 = mr.spawn(f, kwargs={'raises': 1})

    graph = TileableGraph([r7.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)
    await task_api.wait_task(task_id)
    details = await task_api.get_tileable_details(task_id)
    assert details[r7.key]['status'] == SubtaskStatus.errored.value


@pytest.mark.asyncio
async def test_get_subtask_graph(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service
    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, 'progress_controller', _ProgressController)

    def a():
        return md.DataFrame([[1, 2], [3, 4]])

    def b():
        return md.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]])

    def c(a, b):
        return a.sum() * b.sum() / (a.product() * a.sum()) * b.product()

    ra = mr.spawn(a)
    rb = mr.spawn(b)
    rc = mr.spawn(c, args=(ra, rb))

    graph = TileableGraph([rc.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await asyncio.sleep(1)
    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_graph(task_id, tileable.get('tileableId'), True)

        with_input_output_subtask_size = len(with_input_output_subtask_details.get('subtasks'))
        with_input_output_dependency_size = len(with_input_output_subtask_details.get('dependencies'))
        assert with_input_output_subtask_size >= 0
        assert with_input_output_dependency_size <= (with_input_output_subtask_size / 2) * (with_input_output_subtask_size / 2)
        assert ((with_input_output_subtask_size == 0 or with_input_output_subtask_size == 1) and with_input_output_dependency_size == 0) \
            or (with_input_output_subtask_size > 1 and with_input_output_dependency_size > 0)

        subtask_ids = set()
        for subtask in with_input_output_subtask_details.get('subtasks'):
            assert subtask.get('subtaskId') not in subtask_ids
            subtask_ids.add(subtask.get('subtaskId'))

        for dependency in with_input_output_subtask_details.get('dependencies'):
            assert dependency.get('fromSubtaskId') in subtask_ids
            assert dependency.get('toSubtaskId') in subtask_ids

        no_input_output_subtask_details = await task_api.get_tileable_subtask_graph(task_id, tileable.get('tileableId'), False)

        no_input_output_subtask_size = len(no_input_output_subtask_details.get('subtasks'))
        no_input_output_dependency_size = len(no_input_output_subtask_details.get('dependencies'))

        subtask_ids = set()
        for subtask in no_input_output_subtask_details.get('subtasks'):
            assert subtask.get('subtaskId') not in subtask_ids
            subtask_ids.add(subtask.get('subtaskId'))

        for dependency in no_input_output_subtask_details.get('dependencies'):
            assert dependency.get('fromSubtaskId') in subtask_ids
            assert dependency.get('toSubtaskId') in subtask_ids

        assert with_input_output_subtask_size >= no_input_output_subtask_size
        assert with_input_output_dependency_size >= no_input_output_dependency_size


@pytest.mark.asyncio
async def test_get_subtask_detail(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service
    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, 'progress_controller', _ProgressController)

    def a():
        return md.DataFrame([[1, 2], [3, 4]])

    def b():
        return md.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]])

    def c(a, b):
        return a.sum() * b.sum() / (a.product() * a.sum()) * b.product()

    def is_input_or_output(subtask_detail):
        return (subtask_detail.get('status') == SubtaskDisplayNodeStatus.input_node.value \
                    and subtask_detail.get('progress') == SubtaskDisplayNodeStatus.input_node.value) \
                or (subtask_detail.get('status') == SubtaskDisplayNodeStatus.output_node.value \
                    and subtask_detail.get('progress') == SubtaskDisplayNodeStatus.output_node.value)

    ra = mr.spawn(a)
    rb = mr.spawn(b)
    rc = mr.spawn(c, args=(ra, rb))

    graph = TileableGraph([rc.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)

    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)

        subtask_ids = set()
        for subtask_id, subtask_detail in with_input_output_subtask_details.items():
            assert ( \
                (subtask_detail.get('status') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('status') <= SubtaskStatus.cancelled.value) \
                or is_input_or_output(subtask_detail))

            assert ((subtask_detail.get('progress') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('progress') <= SubtaskStatus.running.value) \
                or is_input_or_output(subtask_detail))

            assert subtask_id not in subtask_ids
            subtask_ids.add(subtask_id)

        no_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), False)

        subtask_ids = set()
        for subtask_id, subtask_detail in no_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('status') <= SubtaskStatus.cancelled.value))

            assert ((subtask_detail.get('progress') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('progress') <= SubtaskStatus.running.value))

            assert subtask_id not in subtask_ids
            subtask_ids.add(subtask_id)

    def f1(count: int):
        progress_controller = get_context().get_remote_object('progress_controller')
        for idx in range(count):
            progress_controller.wait()
            get_context().set_progress((1 + idx) * 1.0 / count)

    r = mr.spawn(f1, args=(2,))
    graph = TileableGraph([r.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)

    with pytest.raises(TaskNotExist):
        await task_api.get_tileable_subtask_detail('non_exist', 'non_exist', 'false')

    await asyncio.sleep(0.2)
    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)

        subtask_ids = set()
        for subtask_id, subtask_detail in with_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') == SubtaskStatus.pending.value) \
                or is_input_or_output(subtask_detail))

            assert ((subtask_detail.get('progress') == 0) \
                or is_input_or_output(subtask_detail))

            assert subtask_id not in subtask_ids
            subtask_ids.add(subtask_id)

        no_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), False)

        subtask_ids = set()
        for subtask_id, subtask_detail in no_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') == SubtaskStatus.pending.value))

            assert ((subtask_detail.get('progress') == 0))

            assert subtask_id not in subtask_ids
            subtask_ids.add(subtask_id)

    await ref.set()
    await asyncio.sleep(1)
    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)
        for subtask_id, subtask_detail in with_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('status') <= SubtaskStatus.cancelled.value) \
                or is_input_or_output(subtask_detail))

            assert ((subtask_detail.get('progress') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('progress') <= SubtaskStatus.cancelled.value) \
                or is_input_or_output(subtask_detail))

        no_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), False)
        for subtask_id, subtask_detail in no_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('status') <= SubtaskStatus.cancelled.value))

            assert ((subtask_detail.get('progress') >= SubtaskStatus.pending.value \
                    and subtask_detail.get('progress') <= SubtaskStatus.cancelled.value))

    await ref.set()
    await asyncio.sleep(1)
    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)
        for subtask_id, subtask_detail in with_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') == SubtaskStatus.succeeded.value) \
                or is_input_or_output(subtask_detail))

            assert ((subtask_detail.get('progress') == 1) \
                or is_input_or_output(subtask_detail))

        no_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), False)
        for subtask_id, subtask_detail in no_input_output_subtask_details.items():
            assert (subtask_detail.get('status') == SubtaskStatus.succeeded.value)
            assert (subtask_detail.get('progress') == 1)

    r_errored = mr.spawn(f1, kwargs={'raises': 1})
    graph = TileableGraph([r_errored.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)
    await task_api.wait_task(task_id)
    tileable_graph = await task_api.get_tileable_graph_as_json(task_id)
    for tileable in tileable_graph.get('tileables'):
        with_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)
        for subtask_id, subtask_detail in with_input_output_subtask_details.items():
            assert ((subtask_detail.get('status') == SubtaskStatus.errored.value) \
                or is_input_or_output(subtask_detail))

        no_input_output_subtask_details = await task_api.get_tileable_subtask_detail(task_id, tileable.get('tileableId'), True)
        for subtask_id, subtask_detail in no_input_output_subtask_details.items():
            assert (subtask_detail.get('status') == SubtaskStatus.errored.value)
