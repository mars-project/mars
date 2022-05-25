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
import pandas as pd
import pytest

from .... import dataframe as md
from .... import oscar as mo
from .... import remote as mr
from .... import tensor as mt
from ....core import TileableGraph, TileableGraphBuilder, TileStatus, recursive_tile
from ....core.context import get_context
from ....resource import Resource
from ....tensor.core import TensorOrder
from ....tensor.operands import TensorOperand, TensorOperandMixin
from ....utils import Timer, build_fetch
from ... import start_services, stop_services, NodeRole
from ...session import SessionAPI
from ...storage import MockStorageAPI
from ...subtask import SubtaskStatus
from ...web import WebActor
from ...meta import MetaAPI
from .. import TaskAPI, TaskStatus, WebTaskAPI
from ..supervisor.processor import TaskProcessor
from ..errors import TaskNotExist


@pytest.fixture
async def actor_pools():
    async def start_pool(is_worker: bool):
        if is_worker:
            kw = dict(
                n_process=3,
                labels=["main"] + ["numa-0"] * 2 + ["io"],
                subprocess_start_method="spawn",
            )
        else:
            kw = dict(n_process=1, subprocess_start_method="spawn")
        pool = await mo.create_actor_pool("127.0.0.1", **kw)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(start_pool(False), start_pool(True))
    try:
        yield sv_pool, worker_pool
    finally:
        await asyncio.gather(sv_pool.stop(), worker_pool.stop())


async def _start_services(
    supervisor_pool, worker_pool, request, task_processor_cls=None
):
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
            "lookup_address": supervisor_pool.external_address,
            "resource": {"numa-0": Resource(num_cpus=2)},
        },
        "meta": {"store": "dict"},
        "scheduling": {},
        "task": {},
    }
    if task_processor_cls:
        config["task"]["task_processor_cls"] = task_processor_cls
    if request:
        config["services"].append("web")
    await start_services(
        NodeRole.SUPERVISOR, config, address=supervisor_pool.external_address
    )
    await start_services(NodeRole.WORKER, config, address=worker_pool.external_address)

    session_id = "test_session"
    session_api = await SessionAPI.create(supervisor_pool.external_address)
    await session_api.create_session(session_id)

    if not request.param:
        task_api = await TaskAPI.create(session_id, supervisor_pool.external_address)
    else:
        web_actor = await mo.actor_ref(
            WebActor.default_uid(), address=supervisor_pool.external_address
        )
        web_address = await web_actor.get_web_address()
        task_api = WebTaskAPI(session_id, web_address)

    assert await task_api.get_task_results() == []

    # create mock meta and storage APIs
    _ = await MetaAPI.create(session_id, supervisor_pool.external_address)
    storage_api = await MockStorageAPI.create(session_id, worker_pool.external_address)
    return task_api, storage_api, config


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=[False, True])
async def start_test_service(actor_pools, request):
    sv_pool, worker_pool = actor_pools

    task_api, storage_api, config = await _start_services(sv_pool, worker_pool, request)

    try:
        yield sv_pool.external_address, task_api, storage_api
    finally:
        await MockStorageAPI.cleanup(worker_pool.external_address)
        await stop_services(NodeRole.WORKER, config, worker_pool.external_address)
        await stop_services(NodeRole.SUPERVISOR, config, sv_pool.external_address)


class MockTaskProcessor(TaskProcessor):
    @classmethod
    def _get_decref_stage_chunk_keys(cls, stage_processor):
        import time

        # time.sleep to block async thread
        time.sleep(5)
        return super()._get_decref_stage_chunk_keys(stage_processor)


@pytest.mark.parametrize(indirect=True)
@pytest.fixture(params=[True])
async def start_test_service_with_mock(actor_pools, request):
    sv_pool, worker_pool = actor_pools

    task_api, storage_api, config = await _start_services(
        sv_pool,
        worker_pool,
        request,
        task_processor_cls="mars.services.task.tests.test_service.MockTaskProcessor",
    )

    try:
        yield sv_pool.external_address, task_api, storage_api
    finally:
        await MockStorageAPI.cleanup(worker_pool.external_address)
        await stop_services(NodeRole.WORKER, config, worker_pool.external_address)
        await stop_services(NodeRole.SUPERVISOR, config, sv_pool.external_address)


@pytest.mark.asyncio
async def test_task_timeout_execution(start_test_service_with_mock):
    _sv_pool_address, task_api, storage_api = start_test_service_with_mock

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

    await task_api.wait_task(task_id, timeout=2)
    task_result = await task_api.get_task_result(task_id)

    assert task_result.status == TaskStatus.terminated


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
    assert isinstance(results[0].error, SystemError)


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
    await asyncio.sleep(0.5)
    with Timer() as timer:
        await task_api.cancel_task(task_id)
        result = await task_api.get_task_result(task_id)
        assert result.status == TaskStatus.terminated
    assert timer.duration < 20
    await asyncio.sleep(0.1)
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
        task_api._session_id, "progress_controller", _ProgressController
    )

    def f1(count: int):
        progress_controller = get_context().get_remote_object("progress_controller")
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


class _TileProgressOperand(TensorOperand, TensorOperandMixin):
    @classmethod
    def tile(cls, op: "_TileProgressOperand"):
        progress_controller = get_context().get_remote_object("progress_controller")

        t = yield from recursive_tile(mt.random.rand(10, 10, chunk_size=5))
        yield TileStatus(t.chunks, progress=0.25)
        progress_controller.wait()

        new_op = op.copy()
        params = op.outputs[0].params.copy()
        params["chunks"] = t.chunks
        params["nsplits"] = t.nsplits
        return new_op.new_tileables(t.inputs, kws=[params])


@pytest.mark.asyncio
async def test_task_tile_progress(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service

    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, "progress_controller", _ProgressController
    )

    t = _TileProgressOperand(dtype=np.dtype(np.float64)).new_tensor(
        None, (10, 10), order=TensorOrder.C_ORDER
    )

    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())

    await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await asyncio.sleep(1)
    results = await task_api.get_task_results(progress=True)
    assert results[0].progress == 0.25

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
    try:
        with pytest.raises(TaskNotExist):
            await task_api.get_tileable_graph_as_json("non_exist")

        tileable_detail = await task_api.get_tileable_graph_as_json(task_id)

        num_tileable = len(tileable_detail.get("tileables"))
        num_dependencies = len(tileable_detail.get("dependencies"))
        assert num_tileable > 0
        assert num_dependencies <= (num_tileable / 2) * (num_tileable / 2)

        assert (num_tileable == 1 and num_dependencies == 0) or (
            num_tileable > 1 and num_dependencies > 0
        )

        graph_nodes = []
        graph_dependencies = []
        for node in graph.iter_nodes():
            graph_nodes.append(node.key)

            for node_successor in graph.iter_successors(node):
                graph_dependencies.append(
                    {
                        "fromTileableId": node.key,
                        "toTileableId": node_successor.key,
                        "linkType": 0,
                    }
                )

        for tileable in tileable_detail.get("tileables"):
            graph_nodes.remove(tileable.get("tileableId"))

        assert len(graph_nodes) == 0

        for i in range(num_dependencies):
            dependency = tileable_detail.get("dependencies")[i]
            assert graph_dependencies[i] == dependency
    finally:
        await task_api.wait_task(task_id, timeout=120)


@pytest.mark.asyncio
async def test_get_tileable_details(start_test_service):
    sv_pool_address, task_api, storage_api = start_test_service

    session_api = await SessionAPI.create(address=sv_pool_address)
    ref = await session_api.create_remote_object(
        task_api._session_id, "progress_controller", _ProgressController
    )

    with pytest.raises(TaskNotExist):
        await task_api.get_tileable_details("non_exist")

    def f(*_args, raises=False):
        get_context().set_progress(0.5)
        if raises:
            raise ValueError
        progress_controller = get_context().get_remote_object("progress_controller")
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
    assert _get_fields(details, "progress") == [0.5, 0.0, 0.0]
    assert (
        _get_fields(details, "status", SubtaskStatus)
        == [SubtaskStatus.running] + [SubtaskStatus.pending] * 2
    )

    await ref.set()
    await asyncio.sleep(1)
    details = await task_api.get_tileable_details(task_id)
    assert _get_fields(details, "progress") == [1.0, 0.5, 0.5]
    assert (
        _get_fields(details, "status", SubtaskStatus)
        == [SubtaskStatus.succeeded] + [SubtaskStatus.running] * 2
    )

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
    assert details[r5.key]["progress"] == details[r6.key]["progress"] == 0.25

    await ref.set()
    await asyncio.sleep(0.1)
    await ref.set()
    await task_api.wait_task(task_id)

    # test raises
    r7 = mr.spawn(f, kwargs={"raises": 1})

    graph = TileableGraph([r7.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)
    await task_api.wait_task(task_id)
    details = await task_api.get_tileable_details(task_id)
    assert details[r7.key]["status"] == SubtaskStatus.errored.value

    for tileable in details.keys():
        for property_key, property_value in (
            details.get(tileable).get("properties").items()
        ):
            assert property_key != "key"
            assert property_key != "id"
            assert isinstance(property_value, (int, float, str))

    # test merge
    d1 = pd.DataFrame({"a": np.random.rand(100), "b": np.random.randint(3, size=100)})
    d2 = pd.DataFrame({"c": np.random.rand(100), "b": np.random.randint(3, size=100)})
    df1 = md.DataFrame(d1, chunk_size=10)
    df2 = md.DataFrame(d2, chunk_size=10)

    graph = TileableGraph([df1.data, df2.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)
    await task_api.wait_task(task_id)
    details = await task_api.get_tileable_details(task_id)
    assert details[df1.key]["progress"] == details[df2.key]["progress"] == 1.0

    f1 = build_fetch(df1)
    f2 = build_fetch(df2)
    df3 = f1.merge(f2, auto_merge="none", bloom_filter=False)
    graph = TileableGraph([df3.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=True)
    await task_api.wait_task(task_id)
    for _ in range(2):
        # get twice to ensure cache work
        details = await task_api.get_tileable_details(task_id)
        assert (
            details[df3.key]["progress"]
            == details[f1.key]["progress"]
            == details[f2.key]["progress"]
            == 1.0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_input_output", [False, True])
async def test_get_tileable_subtasks(start_test_service, with_input_output):
    sv_pool_address, task_api, storage_api = start_test_service

    def a():
        return md.DataFrame([[1, 2], [3, 4]])

    def b():
        return md.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]])

    def c(a, b):
        return (
            a.sum()
            * a.product()
            * b.sum()
            * a.sum()
            / a.sum()
            * b.product()
            / a.product()
        )

    ra = mr.spawn(a)
    rb = mr.spawn(b)
    rc = mr.spawn(c, args=(ra, rb))

    graph = TileableGraph([rc.data])
    next(TileableGraphBuilder(graph).build())

    task_id = await task_api.submit_tileable_graph(graph, fuse_enabled=False)

    await asyncio.sleep(1)

    try:
        tileable_graph_json = await task_api.get_tileable_graph_as_json(task_id)
        for tileable_json in tileable_graph_json["tileables"]:
            tileable_id = tileable_json["tileableId"]
            subtask_details = await task_api.get_tileable_subtasks(
                task_id, tileable_id, True
            )

            subtask_deps = []
            for subtask_id, subtask_detail in subtask_details.items():
                for from_subtask_id in subtask_detail.get("fromSubtaskIds", ()):
                    subtask_deps.append((from_subtask_id, subtask_id))
            assert len(subtask_details) > 0

            for from_id, to_id in subtask_deps:
                assert from_id in subtask_details
                assert to_id in subtask_details

            if with_input_output:
                tileable_inputs = [
                    dep["fromTileableId"]
                    for dep in tileable_graph_json["dependencies"]
                    if dep["toTileableId"] == tileable_id
                ]
                tileable_outputs = [
                    dep["toTileableId"]
                    for dep in tileable_graph_json["dependencies"]
                    if dep["fromTileableId"] == tileable_id
                ]
                if tileable_inputs:
                    assert any(
                        detail["nodeType"] == "Input"
                        for detail in subtask_details.values()
                    )
                if tileable_outputs:
                    assert any(
                        detail["nodeType"] == "Output"
                        for detail in subtask_details.values()
                    )
    finally:
        await task_api.wait_task(task_id, timeout=120)
