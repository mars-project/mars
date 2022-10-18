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
import operator
import os
from functools import reduce

import numpy as np
import pytest

from .... import tensor as mt
from .... import dataframe as md
from ....oscar.errors import ReconstructWorkerError
from ....tests.core import require_ray, mock, DICT_NOT_EMPTY
from ....utils import lazy_import
from ..ray import (
    _load_config,
    ClusterStateActor,
    new_cluster,
)
from ..session import get_default_session, new_session
from ..tests import test_local
from .modules.utils import (  # noqa: F401  # pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import("ray")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "local_test_with_ray_config.yml")

EXPECT_PROFILING_STRUCTURE = {
    "supervisor": {
        "general": {
            "optimize": 0.0005879402160644531,
            "incref_fetch_tileables": 0.0010840892791748047,
            "stage_*": {
                "tile(*)": 0.008243083953857422,
                "gen_subtask_graph(*)": 0.012202978134155273,
                "run": 0.27870702743530273,
                "total": 0.30318617820739746,
            },
            "total": 0.30951380729675293,
        },
        "serialization": {
            "serialize": 0.014928340911865234,
            "deserialize": 0.0011813640594482422,
            "total": 0.016109704971313477,
        },
        "most_calls": DICT_NOT_EMPTY,
        "slow_calls": DICT_NOT_EMPTY,
        "band_subtasks": DICT_NOT_EMPTY,
        "slow_subtasks": DICT_NOT_EMPTY,
    }
}
EXPECT_PROFILING_STRUCTURE_NO_SLOW = copy.deepcopy(EXPECT_PROFILING_STRUCTURE)
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_calls"] = {}
EXPECT_PROFILING_STRUCTURE_NO_SLOW["supervisor"]["slow_subtasks"] = {}


@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config(CONFIG_FILE)
    ray_config.update(param.get("config", {}))
    client = await new_cluster(
        supervisor_mem=1 * 1024**3,
        worker_num=2,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        config=ray_config,
    )
    async with client:
        yield client, param


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
async def test_execute(ray_start_regular_shared, create_cluster, config):
    await test_local.test_execute(create_cluster, config)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_start_regular_shared, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_execute_describe(ray_start_regular_shared, create_cluster):
    await test_local.test_execute_describe(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_execute_apply_closure(ray_start_regular_shared, create_cluster):
    await test_local.test_execute_apply_closure(create_cluster)


@require_ray
@pytest.mark.parametrize("multiplier", [1, 3, 4])
@pytest.mark.asyncio
async def test_execute_callable_closure(
    ray_start_regular_shared, create_cluster, multiplier
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
async def test_ray_oscar_clean_up_and_restore_func(
    ray_start_regular_shared, create_cluster
):
    await test_local.test_execute_apply_closure(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_fetch_infos(ray_start_regular_shared, create_cluster):
    await test_local.test_fetch_infos(create_cluster)
    df = md.DataFrame(mt.random.RandomState(0).rand(5000, 1, chunk_size=1000))
    df.execute()
    fetched_infos = df.fetch_infos(fields=["object_refs"])
    object_refs = reduce(operator.concat, fetched_infos["object_refs"])
    assert len(fetched_infos) == 1
    assert len(object_refs) == 5


@require_ray
@pytest.mark.asyncio
def test_sync_execute(ray_start_regular_shared, create_cluster):
    client = create_cluster[0]
    assert client.session
    session = new_session(address=client.address)
    with session:
        raw = np.random.RandomState(0).rand(10, 5)
        a = mt.tensor(raw, chunk_size=5).sum(axis=1)
        b = a.execute(show_progress=False)
        assert b is a
        result = a.fetch()
        np.testing.assert_array_equal(result, raw.sum(axis=1))

        c = mt.tensor(raw, chunk_size=5).sum()
        d = session.execute(c)
        assert d is c
        assert abs(session.fetch(d) - raw.sum()) < 0.001

    assert get_default_session() is None


def _run_web_session(web_address):
    import asyncio

    asyncio.new_event_loop().run_until_complete(
        test_local._run_web_session_test(web_address)
    )
    return True


def _sync_web_session_test(web_address):
    new_session(web_address)
    raw = np.random.RandomState(0).rand(10, 5)
    a = mt.tensor(raw, chunk_size=5).sum(axis=1)
    b = a.execute(show_progress=False)
    assert b is a
    return True


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
async def test_web_session(ray_start_regular_shared, create_cluster, config):
    client = create_cluster[0]
    await test_local.test_web_session(create_cluster, config)
    web_address = client.web_address
    assert await ray.remote(_run_web_session).remote(web_address)
    assert await ray.remote(_sync_web_session_test).remote(web_address)


@require_ray
def test_load_config():
    default_config = _load_config()
    assert default_config["scheduling"]["autoscale"]["enabled"] is False
    default_config = _load_config({"scheduling": {"autoscale": {"enabled": True}}})
    assert default_config["scheduling"]["autoscale"]["enabled"] is True
    default_config = _load_config(
        {
            "scheduling.autoscale.enabled": True,
            "scheduling.autoscale.scheduler_backlog_timeout": 1,
        }
    )
    assert default_config["scheduling"]["autoscale"]["enabled"] is True
    assert default_config["scheduling"]["autoscale"]["scheduler_backlog_timeout"] == 1
    with pytest.raises(ValueError):
        _load_config({"scheduling.autoscale.enabled": True, "scheduling.autoscale": {}})
    assert _load_config(CONFIG_FILE)["session"]["custom_log_dir"] == "auto"


@require_ray
@pytest.mark.asyncio
@mock.patch("mars.deploy.oscar.ray.stop_worker")
async def test_reconstruct_worker_during_releasing_worker(fake_stop_worker):
    stop_worker = asyncio.Event()
    lock = asyncio.Event()

    async def _stop_worker(*args):
        stop_worker.set()
        await lock.wait()

    fake_stop_worker.side_effect = _stop_worker
    cluster_state = ClusterStateActor()
    release_task = asyncio.create_task(cluster_state.release_worker("abc"))
    await stop_worker.wait()
    with pytest.raises(ReconstructWorkerError, match="releasing"):
        await cluster_state.reconstruct_worker("abc")
    release_task.cancel()


@require_ray
@pytest.mark.asyncio
@mock.patch("mars.deploy.oscar.ray.stop_worker")
@mock.patch("ray.get_actor")
async def test_release_worker_during_reconstructing_worker(
    fake_get_actor, fake_stop_worker
):
    get_actor = asyncio.Event()
    lock = asyncio.Event()

    class FakeActorMethod:
        async def remote(self):
            get_actor.set()
            await lock.wait()

    class FakeActor:
        state = FakeActorMethod()

    def _get_actor(*args, **kwargs):
        return FakeActor

    async def _stop_worker(*args):
        await lock.wait()

    fake_get_actor.side_effect = _get_actor
    fake_stop_worker.side_effect = _stop_worker
    cluster_state = ClusterStateActor()
    reconstruct_task = asyncio.create_task(cluster_state.reconstruct_worker("abc"))
    await get_actor.wait()
    release_task = asyncio.create_task(cluster_state.release_worker("abc"))
    with pytest.raises(asyncio.CancelledError):
        await reconstruct_task
    release_task.cancel()


@require_ray
@pytest.mark.asyncio
def test_init_metrics_on_ray(ray_start_regular_shared, create_cluster):
    client = create_cluster[0]
    assert client.session
    from ....metrics import api

    assert client._cluster._config.get("metrics", {}).get("backend") == "ray"
    assert api._metric_backend == "ray"

    client.session.stop_server()
