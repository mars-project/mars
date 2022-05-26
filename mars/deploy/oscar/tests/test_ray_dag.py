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
import copy
import os
import time

import pytest

from .... import get_context
from .... import tensor as mt
from ....config import Config
from ....core import (
    ChunkGraphBuilder,
    Tileable,
    TileContext,
    ChunkGraph,
    Chunk,
)
from ....core.operand import Fetch
from ....resource import Resource
from ....serialization import serialize
from ....services.subtask import SubtaskGraph
from ....services.task import Task
from ....services.task.analyzer import GraphAnalyzer
from ....services.task.execution import RayTaskExecutor, RayExecutionConfig
from ....tests.core import DICT_NOT_EMPTY, require_ray
from ....utils import lazy_import
from ..local import new_cluster
from ..session import new_session, get_default_async_session
from ..tests import test_local
from ..tests.session import new_test_session
from ..tests.test_local import _cancel_when_tile, _cancel_when_execute
from .modules.utils import (  # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import("ray")

EXPECT_PROFILING_STRUCTURE = {
    "supervisor": {
        "general": {
            "optimize": 0.0005879402160644531,
            "stage_*": {
                "tile(*)": 0.008243083953857422,
                "gen_subtask_graph(*)": 0.012202978134155273,
                "run": 0.27870702743530273,
                "total": 0.30318617820739746,
            },
            "total": 0.30951380729675293,
        },
        "serialization": {},
        "most_calls": DICT_NOT_EMPTY,
        "slow_calls": DICT_NOT_EMPTY,
        "band_subtasks": {},
        "slow_subtasks": {},
    }
}
EXPECT_PROFILING_STRUCTURE_NO_SLOW = copy.deepcopy(EXPECT_PROFILING_STRUCTURE)
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_calls"] = {}


def _build_subtask_graph(t: Tileable):
    tileable_graph = t.build_graph(tile=False)
    chunk_graph = next(ChunkGraphBuilder(tileable_graph).build())
    bands = [(f"address_{i}", "numa-0") for i in range(4)]
    band_resource = dict((band, Resource(num_cpus=1)) for band in bands)
    task = Task("mock_task", "mock_session", tileable_graph)
    analyzer = GraphAnalyzer(chunk_graph, band_resource, task, Config(), dict())
    subtask_graph = analyzer.gen_subtask_graph()
    return chunk_graph, subtask_graph


@pytest.mark.parametrize(indirect=True)
@pytest.fixture
async def create_cluster(request):
    start_method = os.environ.get("POOL_START_METHOD", None)
    client = await new_cluster(
        subprocess_start_method=start_method,
        backend="ray",
        n_worker=2,
        n_cpu=2,
        use_uvloop=False,
    )
    async with client:
        assert client.session.client is not None
        yield client, {}


@require_ray
@pytest.mark.parametrize("backend", ["ray"])
@pytest.mark.parametrize("_new_session", [new_session, new_test_session])
def test_new_session_backend(ray_start_regular_shared2, _new_session, backend):
    test_local.test_new_session_backend(_new_session, backend)


@require_ray
@pytest.mark.parametrize(
    "config",
    [
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 0,
                    "slow_subtasks_duration_threshold": 0,
                }
            },
            EXPECT_PROFILING_STRUCTURE,
        ],
        [
            {
                "enable_profiling": {
                    "slow_calls_duration_threshold": 1000,
                    "slow_subtasks_duration_threshold": 1000,
                }
            },
            EXPECT_PROFILING_STRUCTURE_NO_SLOW,
        ],
        [{}, {}],
    ],
)
@pytest.mark.asyncio
async def test_execute(ray_start_regular_shared2, create_cluster, config):
    await test_local.test_execute(create_cluster, config)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_start_regular_shared2, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@require_ray
@pytest.mark.parametrize("config", [{"backend": "ray"}])
def test_sync_execute(config):
    test_local.test_sync_execute(config)


@require_ray
@pytest.mark.asyncio
async def test_session_get_progress(ray_start_regular_shared2, create_cluster):
    await test_local.test_session_get_progress(create_cluster)


@require_ray
@pytest.mark.parametrize("test_func", [_cancel_when_execute, _cancel_when_tile])
def test_cancel(ray_start_regular_shared2, create_cluster, test_func):
    test_local.test_cancel(create_cluster, test_func)


@require_ray
@pytest.mark.parametrize("config", [{"backend": "ray"}])
def test_basic_context_gc(config):
    session = new_session(
        backend=config["backend"],
        n_cpu=2,
        web=False,
        use_uvloop=False,
        config={"task.execution_config.ray.subtask_monitor_interval": 0},
    )

    assert session._session.client.web_address is None
    assert session.get_web_endpoint() is None

    def f1(c):
        time.sleep(0.5)
        return c

    with session:
        t1 = mt.random.randint(10, size=(100, 10), chunk_size=100)
        t2 = mt.random.randint(10, size=(100, 10), chunk_size=50)
        t3 = t2 + t1
        t4 = t3.sum(0)
        t5 = t4.map_chunk(f1)
        r = t5.execute()
        result = r.fetch()
        assert result is not None
        assert len(result) == 10
        context = get_context()
        assert len(context._task_context) < 5

    session.stop_server()
    assert get_default_async_session() is None


class MockRayTaskExecutor(RayTaskExecutor):
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


@require_ray
@pytest.mark.asyncio
async def test_detail_context_gc():
    t1 = mt.random.randint(10, size=(100, 10), chunk_size=100)
    t2 = mt.random.randint(10, size=(100, 10), chunk_size=50)
    t3 = t2 + t1
    t4 = t3.sum(0)
    chunk_graph, subtask_graph = _build_subtask_graph(t4)

    task = Task("mock_task", "mock_session", fuse_enabled=True)
    mock_config = RayExecutionConfig.from_execution_config(
        {"backend": "ray", "ray": {"subtask_monitor_interval": 0}}
    )
    tile_context = MockTileContext()
    task_context = dict()
    task_chunks_meta = dict()
    executor = MockRayTaskExecutor(
        config=mock_config,
        task=task,
        tile_context=tile_context,
        task_context=task_context,
        task_chunks_meta=task_chunks_meta,
        task_state_actor=None,
        lifecycle_api=None,
        meta_api=None,
    )

    monitor_task = await executor.submit_subtask_graph(
        "mock_stage", subtask_graph, chunk_graph
    )
    sequence = await monitor_task

    assert len(task_context) == 1
    assert sequence == [1, 2, 1, 1, 1]
