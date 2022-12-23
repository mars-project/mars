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

import pandas as pd
import pytest
import numpy as np

from collections import Counter

from ...... import dataframe as md
from ...... import tensor as mt
from ......config import Config
from ......core import TileContext
from ......core.context import get_context
from ......core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from ......core.operand import ShuffleFetchType
from ......lib.aio.isolation import new_isolation, stop_isolation
from ......resource import Resource
from ......serialization import serialize
from ......tests.core import require_ray, mock
from ......utils import lazy_import, get_chunk_params
from .....context import ThreadedServiceContext
from .....subtask import Subtask
from ....analyzer import GraphAnalyzer
from ....core import new_task_id, Task
from ..config import RayExecutionConfig
from ..context import (
    RayExecutionContext,
    RayExecutionWorkerContext,
    RayRemoteObjectManager,
    _RayRemoteObjectContext,
)
from ..executor import (
    execute_subtask,
    OrderedSet,
    RayTaskExecutor,
    RayTaskState,
    _RayChunkMeta,
    _RaySubtaskRuntime,
    _RaySlowSubtaskChecker,
)
from ..fetcher import RayFetcher
from ..shuffle import ShuffleManager

ray = lazy_import("ray")


def _gen_subtask_chunk_graph(t):
    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())
    return next(ChunkGraphBuilder(graph, fuse_enabled=False).build())


def _gen_subtask_graph(t):
    tileable_graph = t.build_graph(tile=False)
    chunk_graph = next(ChunkGraphBuilder(tileable_graph).build())
    bands = [(f"address_{i}", "numa-0") for i in range(4)]
    band_resource = dict((band, Resource(num_cpus=1)) for band in bands)
    task = Task("mock_task", "mock_session", tileable_graph)
    analyzer = GraphAnalyzer(
        chunk_graph,
        band_resource,
        task,
        Config(),
        dict(),
        shuffle_fetch_type=ShuffleFetchType.FETCH_BY_INDEX,
    )
    subtask_graph = analyzer.gen_subtask_graph()
    return chunk_graph, subtask_graph


class MockRayTaskExecutor(RayTaskExecutor):
    def __init__(self, *args, **kwargs):
        self._set_attrs = Counter()
        self._monitor_tasks = []
        super().__init__(*args, **kwargs)

    @classmethod
    async def _get_apis(cls, session_id: str, address: str):
        return None, None

    @staticmethod
    def _get_ray_executor():
        # Export remote function once.
        return None

    async def get_available_band_resources(self):
        return {}

    async def execute_subtask_graph(self, *args, **kwargs):
        self._monitor_tasks.clear()
        return await super().execute_subtask_graph(*args, **kwargs)

    async def _update_progress_and_collect_garbage(self, *args, **kwargs):
        # Infinite loop to test monitor task cancel.
        self._monitor_tasks.append(asyncio.current_task())
        return await super()._update_progress_and_collect_garbage(*args, **kwargs)

    def monitor_tasks(self):
        return self._monitor_tasks

    def set_attr_counter(self):
        return self._set_attrs

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self._set_attrs[key] += 1


class MockTileContext(TileContext):
    def get_all_progress(self) -> float:
        return 1.0


@require_ray
@pytest.mark.asyncio
@mock.patch("mars.services.task.execution.ray.executor.RayTaskState.create")
@mock.patch("mars.services.task.execution.ray.context.RayExecutionContext.init")
@mock.patch("ray.get")
async def test_ray_executor_create(
    mock_ray_get, mock_execution_context_init, mock_task_state_actor_create
):
    task = Task("mock_task", "mock_session", TileableGraph([]))

    # Create RayTaskState actor as needed by default.
    mock_config = RayExecutionConfig.from_execution_config({"backend": "ray"})
    executor = await MockRayTaskExecutor.create(
        mock_config,
        session_id="mock_session_id",
        address="mock_address",
        task=task,
        tile_context=TileContext(),
    )
    assert isinstance(executor, MockRayTaskExecutor)
    assert mock_task_state_actor_create.call_count == 0
    ctx = get_context()
    assert isinstance(ctx, RayExecutionContext)
    ctx.create_remote_object("abc", lambda: None)
    assert mock_ray_get.call_count == 1
    assert mock_task_state_actor_create.call_count == 1


@require_ray
@pytest.mark.asyncio
async def test_ray_executor_destroy():
    task = Task("mock_task", "mock_session", TileableGraph([]))
    mock_config = RayExecutionConfig.from_execution_config({"backend": "ray"})
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=TileContext(),
        task_context={},
        task_chunks_meta={},
        lifecycle_api=None,
        meta_api=None,
    )
    counter = executor.set_attr_counter()
    assert len(counter) > 0
    keys = executor.__dict__.keys()
    assert counter.keys() >= keys
    counter.clear()
    executor.destroy()
    keys = set(keys) - {"_set_attrs", "_monitor_tasks"}
    assert counter.keys() == keys, "Some keys are not reset in destroy()."
    for k, v in counter.items():
        assert v == 1
    assert await executor.get_progress() == 1.0


@require_ray
@mock.patch("ray.get_runtime_context")
def test_ray_execute_subtask_basic(_):
    raw = np.ones((10, 10))
    raw_expect = raw + 1
    a = mt.ones((10, 10), chunk_size=10)
    b = a + 1

    subtask_id = new_task_id()
    subtask_chunk_graph = _gen_subtask_chunk_graph(b)
    r = execute_subtask(subtask_id, serialize(subtask_chunk_graph), 0, False)
    np.testing.assert_array_equal(r, raw_expect)
    test_get_meta_chunk = subtask_chunk_graph.result_chunks[0]
    r = execute_subtask(subtask_id, serialize(subtask_chunk_graph), 1, False)
    assert len(r) == 2
    meta_dict, r = r
    assert len(meta_dict) == 1
    assert meta_dict[test_get_meta_chunk.key][0] == get_chunk_params(
        test_get_meta_chunk
    )
    np.testing.assert_array_equal(r, raw_expect)


@require_ray
@pytest.mark.asyncio
async def test_ray_fetcher(ray_start_regular_shared2):
    pd_value = pd.DataFrame(
        {
            "col1": [str(i) for i in range(10)],
            "col2": np.random.randint(0, 100, (10,)),
        }
    )
    pd_object_ref = ray.put(pd_value)
    np_value = np.asarray([1, 3, 6, 2, 4])
    np_object_ref = ray.put(np_value)
    # Test RayFetcher to fetch mixed values.
    fetcher = RayFetcher()
    await fetcher.append("pd_key", {"object_refs": [pd_object_ref]})
    await fetcher.append("np_key", {"object_refs": [np_object_ref]})
    await fetcher.append("pd_key", {"object_refs": [pd_object_ref]}, [slice(1, 3, 1)])
    await fetcher.append("np_key", {"object_refs": [np_object_ref]}, [slice(1, 3, 1)])
    results = await fetcher.get()
    pd.testing.assert_frame_equal(results[0], pd_value)
    np.testing.assert_array_equal(results[1], np_value)
    pd.testing.assert_frame_equal(results[2], pd_value.iloc[1:3])
    np.testing.assert_array_equal(results[3], np_value[1:3])


@require_ray
@pytest.mark.asyncio
async def test_ray_remote_object(ray_start_regular_shared2):
    class _TestRemoteObject:
        def __init__(self, i):
            self._i = i

        def value(self):
            return self._i

        def foo(self, a, b):
            return self._i + a + b

        async def bar(self, a, b):
            return self._i * a * b

    # Test RayTaskState reference
    state = RayTaskState.create()
    await state.create_remote_object.remote("aaa", _TestRemoteObject, 123)
    assert await state.call_remote_object.remote("aaa", "value") == 123
    state = RayTaskState.create()
    assert await state.call_remote_object.remote("aaa", "value") == 123

    # Test RayRemoteObjectManager
    name = "abc"
    manager = RayRemoteObjectManager()
    manager.create_remote_object(name, _TestRemoteObject, 2)
    r = await manager.call_remote_object(name, "foo", 3, 4)
    assert r == 9
    r = await manager.call_remote_object(name, "bar", 3, 4)
    assert r == 24
    manager.destroy_remote_object(name)
    with pytest.raises(KeyError):
        await manager.call_remote_object(name, "foo", 3, 4)

    # Test _RayRemoteObjectContext
    context = _RayRemoteObjectContext(lambda: RayTaskState.create())
    context.create_remote_object(name, _TestRemoteObject, 2)
    remote_object = context.get_remote_object(name)
    r = remote_object.foo(3, 4)
    assert r == 9
    r = remote_object.bar(3, 4)
    assert r == 24
    context.destroy_remote_object(name)
    with pytest.raises(KeyError):
        remote_object.foo(3, 4)

    class MyException(Exception):
        pass

    class _ErrorRemoteObject:
        def __init__(self):
            raise MyException()

    with pytest.raises(MyException):
        context.create_remote_object(name, _ErrorRemoteObject)

    handle = RayTaskState.get_handle()
    assert handle is not None


@require_ray
def test_ray_execution_context(ray_start_regular_shared2):
    value = 123
    o = ray.put(value)

    def fake_init(self):
        pass

    async def fake_get_chunks_meta_from_service(
        self, data_keys, fields=None, error="raise"
    ):
        mock_meta = {"meta_1": {fields[0]: 1}, "meta_3": {fields[0]: 3}}
        return [mock_meta[k] for k in data_keys]

    with mock.patch.object(
        ThreadedServiceContext, "__init__", new=fake_init
    ), mock.patch.object(
        RayExecutionContext,
        "_get_chunks_meta_from_service",
        new=fake_get_chunks_meta_from_service,
    ):
        mock_config = RayExecutionConfig.from_execution_config({"backend": "ray"})
        mock_worker_addresses = ["mock_worker_address"]
        isolation = new_isolation("test", threaded=True)
        try:
            context = RayExecutionContext(
                mock_config, {"abc": o}, {}, mock_worker_addresses, lambda: None
            )
            context._loop = isolation.loop
            r = context.get_chunks_result(["abc"])
            assert r == [value]

            r = context.get_worker_addresses()
            assert r == mock_worker_addresses

            r = context.get_chunks_meta(["meta_1"], fields=["memory_size"])
            assert r == [{"memory_size": 1}]

            context._task_chunks_meta["meta_1"] = _RayChunkMeta(memory_size=2)
            r = context.get_chunks_meta(["meta_1", "meta_3"], fields=["memory_size"])
            assert r == [{"memory_size": 2}, {"memory_size": 3}]
        finally:
            stop_isolation("test")


def test_ray_execution_worker_context():
    context = RayExecutionWorkerContext(lambda: None)
    with pytest.raises(NotImplementedError):
        context.set_running_operand_key("mock_session_id", "mock_op_key")
    with pytest.raises(NotImplementedError):
        context.register_custom_log_path(
            "mock_session_id",
            "mock_tileable_op_key",
            "mock_chunk_op_key",
            "mock_worker_address",
            "mock_log_path",
        )

    assert context.set_progress(0.1) is None
    assert context.new_custom_log_dir() is None
    assert context.get_storage_info("mock_address") == {}


@require_ray
@pytest.mark.asyncio
async def test_ray_execution_config(ray_start_regular_shared2):
    t1 = mt.random.randint(10, size=(100, 10), chunk_size=100)
    chunk_graph, subtask_graph = _gen_subtask_graph(t1)

    real_executor = RayTaskExecutor._get_ray_executor()

    class MockExecutor:
        opt = {}

        @classmethod
        def options(cls, *args, **kwargs):
            cls.opt = kwargs
            return real_executor.options(*args, **kwargs)

    task = Task("mock_task", "mock_session", TileableGraph([]))
    mock_config = RayExecutionConfig.from_execution_config(
        {
            "backend": "ray",
            "ray": {
                "monitor_interval_seconds": 0,
                "subtask_max_retries": 4,
                "subtask_num_cpus": 0.8,
                "subtask_memory": 1001,
                "n_cpu": 1,
                "n_worker": 1,
            },
        }
    )
    tile_context = MockTileContext()
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context={},
        task_chunks_meta={},
        lifecycle_api=None,
        meta_api=None,
    )
    executor._ray_executor = MockExecutor
    async with executor:
        await executor.execute_subtask_graph(
            "mock_stage", subtask_graph, chunk_graph, tile_context
        )

    assert MockExecutor.opt["num_cpus"] == 0.8
    assert MockExecutor.opt["max_retries"] == 4
    assert MockExecutor.opt["memory"] == 1001


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("gc_method", ["submitted", "completed"])
async def test_executor_context_gc(ray_start_regular_shared2, gc_method):
    popped_seq = []

    class MockTaskContext(dict):
        def pop(self, k, d=None):
            popped_seq.append(k)
            return super().pop(k, d)

    t1 = mt.random.randint(10, size=(100, 10), chunk_size=100)
    t2 = mt.random.randint(10, size=(100, 10), chunk_size=50)
    t3 = t2 + t1
    t4 = t3.sum(0)
    chunk_graph, subtask_graph = _gen_subtask_graph(t4)
    task = Task("mock_task", "mock_session", TileableGraph([]), fuse_enabled=True)
    mock_config = RayExecutionConfig.from_execution_config(
        {
            "backend": "ray",
            "ray": {
                "monitor_interval_seconds": 0,
                "log_interval_seconds": 0,
                "subtask_max_retries": 0,
                "n_cpu": 1,
                "n_worker": 1,
                "gc_method": gc_method,
            },
        }
    )
    tile_context = MockTileContext()
    task_context = MockTaskContext()
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context=task_context,
        task_chunks_meta={},
        lifecycle_api=None,
        meta_api=None,
    )
    executor._ray_executor = RayTaskExecutor._get_ray_executor()

    original_execute_subtask_graph = executor._execute_subtask_graph

    async def _wait_gc_execute_subtask_graph(*args, **kwargs):
        # Mock _execute_subtask_graph to wait the monitor task done.
        await original_execute_subtask_graph(*args, **kwargs)
        await executor.monitor_tasks()[0]

    with mock.patch.object(
        executor, "_execute_subtask_graph", _wait_gc_execute_subtask_graph
    ):
        async with executor:
            await executor.execute_subtask_graph(
                "mock_stage", subtask_graph, chunk_graph, tile_context
            )
            await asyncio.sleep(0)
            assert len(executor.monitor_tasks()) == 1
            assert executor.monitor_tasks()[0].done()

    assert len(task_context) == 1
    assert len(popped_seq) == 6
    subtasks = list(subtask_graph.topological_iter())
    chunk_keys1 = set(
        map(
            lambda c: c.key,
            (
                subtasks[0].chunk_graph.results
                + subtasks[1].chunk_graph.results
                + subtasks[3].chunk_graph.results
            ),
        )
    )
    chunk_keys2 = set(
        map(
            lambda c: c.key,
            (subtasks[2].chunk_graph.results + subtasks[4].chunk_graph.results),
        )
    )
    assert chunk_keys1 == set(popped_seq[0:4])
    assert chunk_keys2 == set(popped_seq[4:])

    task_context.clear()

    original_update_progress_and_collect_garbage = (
        executor._update_progress_and_collect_garbage
    )

    async def infinite_update_progress_and_collect_garbage(*args, **kwargs):
        # Mock _update_progress_and_collect_garbage that never done.
        await original_update_progress_and_collect_garbage(*args, **kwargs)
        while True:
            await asyncio.sleep(0)

    with mock.patch("logging.Logger.info") as log_patch, mock.patch.object(
        executor,
        "_update_progress_and_collect_garbage",
        infinite_update_progress_and_collect_garbage,
    ):
        async with executor:
            await executor.execute_subtask_graph(
                "mock_stage2", subtask_graph, chunk_graph, tile_context
            )
            await asyncio.sleep(0)
            assert len(executor.monitor_tasks()) == 1
            assert executor.monitor_tasks()[0].done()
        assert log_patch.call_count > 0
        args = [c.args[0] for c in log_patch.call_args_list]
        assert any("Submitted [%s/%s]" in a for a in args)
        assert any("Completed [%s/%s]" in a for a in args)

    assert len(task_context) == 1

    task_context.clear()

    # Test the monitor aiotask is done even an exception is raised.
    async def _raise_load_subtask_inputs(*args, **kwargs):
        # Mock _load_subtask_inputs to raise an exception.
        await asyncio.sleep(0)
        1 / 0

    with mock.patch.object(
        executor, "_load_subtask_inputs", _raise_load_subtask_inputs
    ):
        async with executor:
            with pytest.raises(ZeroDivisionError):
                await executor.execute_subtask_graph(
                    "mock_stage3", subtask_graph, chunk_graph, tile_context
                )
            await asyncio.sleep(0)
            assert len(executor.monitor_tasks()) == 1
            assert executor.monitor_tasks()[0].done()


@require_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("gc_method", ["submitted", "completed"])
async def test_execute_shuffle(ray_start_regular_shared2, gc_method):
    chunk_size, n_rows = 10, 50
    df = md.DataFrame(
        pd.DataFrame(np.random.rand(n_rows, 3), columns=list("abc")),
        chunk_size=chunk_size,
    )
    df2 = df.groupby(["a"]).apply(lambda x: x)
    chunk_graph, subtask_graph = _gen_subtask_graph(df2)
    task = Task("mock_task", "mock_session", TileableGraph([]), fuse_enabled=True)

    class MockRayExecutor:
        @staticmethod
        def options(**kwargs):
            num_returns = kwargs["num_returns"]

            class _Wrapper:
                @staticmethod
                def remote(*args):
                    args = [
                        ray.get(a) if isinstance(a, ray.ObjectRef) else a for a in args
                    ]
                    r = execute_subtask(*args)
                    assert len(r) == num_returns
                    return [ray.put(i) for i in r]

            return _Wrapper

    mock_config = RayExecutionConfig.from_execution_config(
        {
            "backend": "ray",
            "ray": {
                "monitor_interval_seconds": 0,
                "subtask_max_retries": 0,
                "n_cpu": 1,
                "n_worker": 1,
                "gc_method": gc_method,
            },
        }
    )
    tile_context = MockTileContext()
    task_context = {}
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context=task_context,
        task_chunks_meta={},
        lifecycle_api=None,
        meta_api=None,
    )
    executor._ray_executor = MockRayExecutor

    # Test ShuffleManager.remove_object_refs
    sm = ShuffleManager(subtask_graph)
    sm._mapper_output_refs[0].fill(1)
    sm.remove_object_refs(next(iter(sm._reducer_indices.keys())))
    assert pd.isnull(sm._mapper_output_refs[0][:, 0]).all()
    sm._mapper_output_refs[0].fill(1)
    sm.remove_object_refs(next(iter(sm._mapper_indices.keys())))
    assert pd.isnull(sm._mapper_output_refs[0][0]).all()
    with pytest.raises(ValueError):
        sm.remove_object_refs(None)

    original_execute_subtask_graph = executor._execute_subtask_graph

    async def _wait_gc_execute_subtask_graph(
        stage_id, subtask_graph, chunk_graph, monitor_context
    ):
        # Mock _execute_subtask_graph to wait the monitor task done.
        await original_execute_subtask_graph(
            stage_id, subtask_graph, chunk_graph, monitor_context
        )
        await executor.monitor_tasks()[0]
        assert pd.isnull(monitor_context.shuffle_manager._mapper_output_refs[0]).all()

    with mock.patch.object(
        executor, "_execute_subtask_graph", _wait_gc_execute_subtask_graph
    ), mock.patch("ray.get_runtime_context"):
        async with executor:
            await executor.execute_subtask_graph(
                "mock_stage", subtask_graph, chunk_graph, tile_context
            )
        await asyncio.sleep(0)
        assert len(executor.monitor_tasks()) == 1
        assert executor.monitor_tasks()[0].done()

    assert len(task_context) == len(chunk_graph.results)


@require_ray
@pytest.mark.asyncio
async def test_slow_subtask_checker():
    subtasks = [
        Subtask(str(i), logic_key=f"logic_key1", logic_parallelism=5) for i in range(5)
    ]
    for s in subtasks:
        s.runtime = _RaySubtaskRuntime()
    submitted = OrderedSet()
    completed = OrderedSet()
    now = time.time()
    checker = _RaySlowSubtaskChecker(5, submitted, completed)
    updater = checker.update()
    for s in subtasks:
        submitted.add(s)
    for _ in updater:
        break
    assert all(s.runtime.start_time >= now for s in subtasks)
    await asyncio.sleep(0.01)
    assert not any(checker.is_slow(s) for s in subtasks)
    completed.add(subtasks[0])
    completed.add(subtasks[1])
    for _ in updater:
        break
    await asyncio.sleep(0.01)
    completed.add(subtasks[2])
    assert not any(checker.is_slow(s) for s in subtasks[3:])
    completed.add(subtasks[3])
    for _ in updater:
        break
    assert not checker.is_slow(subtasks[4])
    await asyncio.sleep(0.1)
    assert checker.is_slow(subtasks[4])


@require_ray
@pytest.mark.asyncio
async def test_execute_slow_task(ray_start_regular_shared2):
    t1 = mt.random.randint(10, size=(100, 10), chunk_size=10)
    t2 = mt.random.randint(10, size=(100, 10), chunk_size=30)
    t3 = t2 + t1
    t4 = t3.sum(0)
    chunk_graph, subtask_graph = _gen_subtask_graph(t4)
    task = Task("mock_task", "mock_session", TileableGraph([]), fuse_enabled=True)
    mock_config = RayExecutionConfig.from_execution_config(
        {
            "backend": "ray",
            "ray": {
                "monitor_interval_seconds": 0,
                "log_interval_seconds": 0,
                "check_slow_subtasks_interval_seconds": 0,
                "subtask_max_retries": 0,
                "n_cpu": 1,
                "n_worker": 1,
            },
        }
    )
    tile_context = MockTileContext()
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context={},
        task_chunks_meta={},
        lifecycle_api=None,
        meta_api=None,
    )
    slow_subtask_id = list(subtask_graph)[-1].subtask_id

    def mock_execute_subtask(subtask_id, *args):
        if subtask_id == slow_subtask_id:
            time.sleep(1)
        return execute_subtask(subtask_id, *args)

    executor._ray_executor = ray.remote(mock_execute_subtask)

    with mock.patch("logging.Logger.info") as log_patch:
        async with executor:
            await executor.execute_subtask_graph(
                "mock_stage2", subtask_graph, chunk_graph, tile_context
            )
            await asyncio.sleep(0)
            assert len(executor.monitor_tasks()) == 1
            assert executor.monitor_tasks()[0].done()
        assert log_patch.call_count > 0
        slow_ray_object_refs = set()
        for c in log_patch.call_args_list:
            if c.args[0] == "Slow tasks(%s): %s":
                count, object_refs = c.args[1:]
                assert count >= 1
                slow_ray_object_refs.update(object_refs)
        assert len(slow_ray_object_refs) >= 1
