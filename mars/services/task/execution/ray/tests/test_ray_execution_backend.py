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

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from ...... import tensor as mt

from ......core import TileContext
from ......core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from ......serialization import serialize
from ......tests.core import require_ray, mock
from ......utils import lazy_import, get_chunk_params
from .....context import ThreadedServiceContext
from ....core import new_task_id, Task
from ..config import RayExecutionConfig
from ..context import (
    RayExecutionContext,
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


class MockRayTaskExecutor(RayTaskExecutor):
    def __init__(self, *args, **kwargs):
        self._set_attrs = Counter()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_ray_executor():
        # Export remote function once.
        return None

    def set_attr_counter(self):
        return self._set_attrs

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self._set_attrs[key] += 1


def test_ray_executor_destroy():
    task = Task("mock_task", "mock_session")
    config = RayExecutionConfig.from_execution_config({"backend": "mars"})
    executor = MockRayTaskExecutor(
        config=config,
        task=task,
        tile_context=TileContext(),
        task_context={},
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
    assert meta_dict[test_get_meta_chunk.key] == get_chunk_params(test_get_meta_chunk)
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
        context = RayExecutionContext({"abc": o}, None)
        r = context.get_chunks_result(["abc"])
        assert r == [value]
