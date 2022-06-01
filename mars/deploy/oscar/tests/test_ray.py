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

import mars
from .... import tensor as mt
from .... import dataframe as md
from ....oscar.errors import ReconstructWorkerError
from ....serialization.ray import register_ray_serializers
from ....tests.core import require_ray, mock, DICT_NOT_EMPTY
from ....utils import lazy_import
from ..ray import (
    _load_config,
    ClusterStateActor,
    new_cluster,
    new_cluster_in_ray,
    new_ray_session,
)
from ..session import get_default_session, new_session
from ..tests import test_local
from .modules.utils import (  # noqa: F401  # pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import("ray")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "local_test_with_ray_config.yml")
CONFIG_THIRD_PARTY_MODULES_TEST_FILE = os.path.join(
    os.path.dirname(__file__), "ray_test_with_third_parity_modules_config.yml"
)

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
    register_ray_serializers()
    import asyncio

    asyncio.new_event_loop().run_until_complete(
        test_local._run_web_session_test(web_address)
    )
    return True


def _sync_web_session_test(web_address):
    register_ray_serializers()
    new_session(web_address)
    raw = np.random.RandomState(0).rand(10, 5)
    a = mt.tensor(raw, chunk_size=5).sum(axis=1)
    b = a.execute(show_progress=False)
    assert b is a
    return True


@require_ray
def test_new_cluster_in_ray(stop_ray):
    cluster = new_cluster_in_ray(worker_num=2)
    mt.random.RandomState(0).rand(100, 5).sum().execute()
    cluster.session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    session = new_ray_session(address=cluster.address, session_id="abcd", default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    cluster.stop()


@require_ray
def test_new_ray_session(stop_ray):
    new_ray_session_test()


def new_ray_session_test():
    session = new_ray_session(session_id="abc", worker_num=2)
    mt.random.RandomState(0).rand(100, 5).sum().execute()
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    session = new_ray_session(session_id="abcd", worker_num=2, default=True)
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    df = md.DataFrame(mt.random.rand(100, 4), columns=list("abcd"))
    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    print(ds.schema(), ds.count())
    ds.filter(lambda row: row["a"] > 0.5).show(5)
    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds)
    print(df2.head(5).execute())
    # Test ray cluster exists after session got gc.
    del session
    import gc

    gc.collect()
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())


@require_ray
@pytest.mark.parametrize(
    "test_option",
    [
        [True, 0, ["ray://test_cluster/1/0", "ray://test_cluster/2/0"]],
        [False, 0, ["ray://test_cluster/0/1", "ray://test_cluster/1/0"]],
        [True, 2, ["ray://test_cluster/1/0", "ray://test_cluster/2/0"]],
        [False, 5, ["ray://test_cluster/0/6", "ray://test_cluster/1/0"]],
    ],
)
@pytest.mark.asyncio
async def test_optional_supervisor_node(ray_start_regular_shared, test_option):
    import logging

    logging.basicConfig(level=logging.INFO)
    supervisor_standalone, supervisor_sub_pool_num, worker_addresses = test_option
    config = _load_config()
    config["cluster"]["ray"]["supervisor"]["standalone"] = supervisor_standalone
    config["cluster"]["ray"]["supervisor"]["sub_pool_num"] = supervisor_sub_pool_num
    client = await new_cluster(
        "test_cluster",
        supervisor_mem=1 * 1024**3,
        worker_num=2,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        config=config,
    )
    async with client:
        assert client.address == "ray://test_cluster/0/0"
        assert client._cluster._worker_addresses == worker_addresses


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
@pytest.mark.parametrize(
    "config_exception",
    [
        [set(), pytest.raises(TypeError, match="set")],
        [
            {"supervisor": ["not_exists_for_supervisor"]},
            pytest.raises(ModuleNotFoundError, match="not_exists_for_supervisor"),
        ],
        [
            {"worker": ["not_exists_for_worker"]},
            pytest.raises(ModuleNotFoundError, match="not_exists_for_worker"),
        ],
    ],
)
@pytest.mark.asyncio
async def test_load_third_party_modules(ray_start_regular_shared, config_exception):
    third_party_modules_config, expected_exception = config_exception
    config = _load_config()

    config["third_party_modules"] = third_party_modules_config
    with expected_exception:
        await new_cluster(
            worker_num=1,
            worker_cpu=1,
            worker_mem=1 * 1024**3,
            config=config,
        )


@require_ray
@pytest.mark.parametrize(
    "create_cluster",
    [
        {
            "config": {
                "third_party_modules": {
                    "worker": ["mars.deploy.oscar.tests.modules.replace_op"]
                },
            },
        }
    ],
    indirect=True,
)
@pytest.mark.asyncio
def test_load_third_party_modules2(ray_start_regular_shared, create_cluster):
    client = create_cluster[0]
    assert client.session
    session = new_session(address=client.address)
    with session:
        raw = np.random.RandomState(0).rand(10, 10)
        a = mt.tensor(raw, chunk_size=5)
        b = a + 1
        b.execute(show_progress=False)
        result = b.fetch()

        np.testing.assert_equal(raw - 1, result)

    assert get_default_session() is None


@require_ray
@pytest.mark.asyncio
async def test_load_third_party_modules_from_config(
    ray_start_regular_shared, cleanup_third_party_modules_output  # noqa: F811
):
    client = await new_cluster(
        supervisor_mem=1 * 1024**3,
        worker_num=1,
        worker_cpu=1,
        worker_mem=1 * 1024**3,
        config=CONFIG_THIRD_PARTY_MODULES_TEST_FILE,
    )
    async with client:
        # 1 supervisor, 1 worker main pools, 1 worker sub pools.
        assert len(get_output_filenames()) == 3


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
