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
import pandas as pd
import pytest
import numpy as np

from collections import Counter
from ....analyzer import GraphAnalyzer
from .....subtask import SubtaskGraph
from ...... import tensor as mt
from ......config import Config

from ......core import TileContext, ChunkGraph
from ......core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder

from ......core.operand import Fetch

from ......resource import Resource
from ......serialization import serialize
from ......tests.core import require_ray, mock
from ......utils import lazy_import, get_chunk_params
from .....context import ThreadedServiceContext
from ....core import new_task_id, Task
from ..config import RayExecutionConfig
from ..context import (
    RayExecutionContext,
    RayExecutionWorkerContext,
    RayRemoteObjectManager,
    _RayRemoteObjectContext,
)
from ..executor import execute_subtask, RayTaskExecutor
from ..fetcher import RayFetcher

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
    analyzer = GraphAnalyzer(chunk_graph, band_resource, task, Config(), dict())
    subtask_graph = analyzer.gen_subtask_graph()
    return chunk_graph, subtask_graph


class MockRayTaskExecutor(RayTaskExecutor):
    def __init__(self, *args, **kwargs):
        self._set_attrs = Counter()
        super().__init__(*args, **kwargs)

    def set_attr_counter(self):
        return self._set_attrs

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self._set_attrs[key] += 1

    async def submit_subtask_graph(
        self,
        stage_id: str,
        subtask_graph: SubtaskGraph,
        chunk_graph: ChunkGraph,
    ):
        monitor_task = asyncio.create_task(
            self._update_progress_and_collect_garbage(
                subtask_graph, self._config.get_subtask_monitor_interval()
            )
        )

        result_meta_keys = {
            chunk.key
            for chunk in chunk_graph.result_chunks
            if not isinstance(chunk.op, Fetch)
        }

        for subtask in subtask_graph.topological_iter():
            subtask_chunk_graph = subtask.chunk_graph
            task_context = self._task_context
            key_to_input = await self._load_subtask_inputs(
                stage_id, subtask, subtask_chunk_graph, task_context
            )
            output_keys = self._get_subtask_output_keys(subtask_chunk_graph)
            output_meta_keys = result_meta_keys & output_keys
            output_count = len(output_keys) + bool(output_meta_keys)
            output_object_refs = self._ray_executor.options(
                num_returns=output_count
            ).remote(
                subtask.task_id,
                subtask.subtask_id,
                serialize(subtask_chunk_graph),
                output_meta_keys,
                list(key_to_input.keys()),
                *key_to_input.values(),
            )
            if output_count == 0:
                continue
            elif output_count == 1:
                output_object_refs = [output_object_refs]
            self._cur_stage_first_output_object_ref_to_subtask[
                output_object_refs[0]
            ] = subtask
            if output_meta_keys:
                meta_object_ref, *output_object_refs = output_object_refs
            task_context.update(zip(output_keys, output_object_refs))

        return monitor_task


class MockTileContext(TileContext):
    def get_all_progress(self) -> float:
        return 1.0


@pytest.mark.asyncio
async def test_ray_executor_destroy():
    task = Task("mock_task", "mock_session")
    mock_config = RayExecutionConfig.from_execution_config({"backend": "ray"})
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=TileContext(),
        task_context={},
        task_chunks_meta={},
        task_state_actor=None,
        lifecycle_api=None,
        meta_api=None,
    )
    counter = executor.set_attr_counter()
    assert len(counter) > 0
    keys = executor.__dict__.keys()
    assert counter.keys() >= keys
    counter.clear()
    executor.destroy()
    keys = set(keys) - {"_set_attrs"}
    assert counter.keys() == keys, "Some keys are not reset in destroy()."
    for k, v in counter.items():
        assert v == 1
    assert await executor.get_progress() == 1.0


def test_ray_execute_subtask_basic():
    raw = np.ones((10, 10))
    raw_expect = raw + 1
    a = mt.ones((10, 10), chunk_size=10)
    b = a + 1

    subtask_id = new_task_id()
    subtask_chunk_graph = _gen_subtask_chunk_graph(b)
    r = execute_subtask("", subtask_id, serialize(subtask_chunk_graph), set(), [])
    np.testing.assert_array_equal(r, raw_expect)
    test_get_meta_chunk = subtask_chunk_graph.result_chunks[0]
    r = execute_subtask(
        "", subtask_id, serialize(subtask_chunk_graph), {test_get_meta_chunk.key}, []
    )
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
    await fetcher.append("pd_key", {"object_refs": [pd_object_ref]}, [1, 3])
    await fetcher.append("np_key", {"object_refs": [np_object_ref]}, [1, 3])
    results = await fetcher.get()
    pd.testing.assert_frame_equal(results[0], pd_value)
    np.testing.assert_array_equal(results[1], np_value)
    pd.testing.assert_frame_equal(results[2], pd_value.iloc[[1, 3]])
    np.testing.assert_array_equal(results[3], np_value[[1, 3]])


@require_ray
@pytest.mark.asyncio
async def test_ray_remote_object(ray_start_regular_shared2):
    class _TestRemoteObject:
        def __init__(self, i):
            self._i = i

        def foo(self, a, b):
            return self._i + a + b

        async def bar(self, a, b):
            return self._i * a * b

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
    remote_manager = ray.remote(RayRemoteObjectManager).remote()
    context = _RayRemoteObjectContext(remote_manager)
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


@require_ray
def test_get_chunks_result(ray_start_regular_shared2):
    value = 123
    o = ray.put(value)

    def fake_init(self):
        pass

    with mock.patch.object(ThreadedServiceContext, "__init__", new=fake_init):
        mock_config = RayExecutionConfig.from_execution_config({"backend": "ray"})
        context = RayExecutionContext(mock_config, {"abc": o}, {}, None)
        r = context.get_chunks_result(["abc"])
        assert r == [value]


def test_ray_execution_worker_context():
    context = RayExecutionWorkerContext(None)
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
async def test_executor_context_gc():
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
    task = Task("mock_task", "mock_session", fuse_enabled=True)
    mock_config = RayExecutionConfig.from_execution_config(
        {"backend": "ray", "ray": {"subtask_monitor_interval": 0}}
    )
    tile_context = MockTileContext()
    task_context = MockTaskContext()
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context=task_context,
        task_chunks_meta={},
        task_state_actor=None,
        lifecycle_api=None,
        meta_api=None,
    )
    monitor_task = await executor.submit_subtask_graph(
        "mock_stage", subtask_graph, chunk_graph
    )
    await monitor_task

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
