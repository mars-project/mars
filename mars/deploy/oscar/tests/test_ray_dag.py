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

import copy
import os
import time

import pytest

from .... import get_context
from .... import tensor as mt
from ....tests import test_session
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


@pytest.mark.parametrize(indirect=True)
@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    start_method = os.environ.get("POOL_START_METHOD", None)
    client = await new_cluster(
        subprocess_start_method=start_method,
        backend="ray",
        n_worker=2,
        n_cpu=2,
        use_uvloop=False,
        config=param.get("config", None),
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
def test_sync_execute(ray_start_regular_shared2, config):
    test_local.test_sync_execute(config)


@require_ray
@pytest.mark.parametrize(
    "create_cluster",
    [{"config": {"task.execution_config.ray.monitor_interval_seconds": 0}}],
    indirect=True,
)
@pytest.mark.asyncio
async def test_session_get_progress(ray_start_regular_shared2, create_cluster):
    await test_local.test_session_get_progress(create_cluster)


@require_ray
@pytest.mark.parametrize("test_func", [_cancel_when_execute, _cancel_when_tile])
def test_cancel(ray_start_regular_shared2, create_cluster, test_func):
    test_local.test_cancel(create_cluster, test_func)


@require_ray
@pytest.mark.parametrize("config", [{"backend": "ray"}])
def test_executor_context_gc(ray_start_regular_shared2, config):
    session = new_session(
        backend=config["backend"],
        n_cpu=2,
        web=False,
        use_uvloop=False,
        config={"task.execution_config.ray.monitor_interval_seconds": 0},
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


@require_ray
@pytest.mark.asyncio
async def test_execute_describe(ray_start_regular_shared2, create_cluster):
    # `describe` contains multiple shuffle.
    await test_local.test_execute_describe(create_cluster)


@require_ray
@pytest.mark.parametrize("method", ["shuffle", "broadcast", None])
@pytest.mark.parametrize("auto_merge", ["after", "before"])
def test_merge_groupby(ray_start_regular_shared2, setup, method, auto_merge):
    # add ray_dag decorator to the test_merge_groupby makes the raylet crash.
    test_session.test_merge_groupby(setup, method, auto_merge)


@require_ray
@pytest.mark.asyncio
async def test_execute_apply_closure(ray_start_regular_shared2, create_cluster):
    await test_local.test_execute_apply_closure(create_cluster)


@require_ray
@pytest.mark.parametrize("multiplier", [1, 3, 4])
@pytest.mark.asyncio
async def test_execute_callable_closure(
    ray_start_regular_shared2, create_cluster, multiplier
):
    await test_local.test_execute_callable_closure(create_cluster, multiplier)


@require_ray
@pytest.mark.parametrize(
    "create_cluster",
    [
        {
            "config": {
                "task.task_preprocessor_cls": "mars.deploy.oscar.tests.test_clean_up_and_restore_func.RayBackendFuncTaskPreprocessor",
                "subtask.subtask_processor_cls": "mars.deploy.oscar.tests.test_clean_up_and_restore_func.RayBackendFuncSubtaskProcessor",
            }
        }
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_ray_dag_clean_up_and_restore_func(
    ray_start_regular_shared2, create_cluster
):
    await test_local.test_execute_apply_closure(create_cluster)
