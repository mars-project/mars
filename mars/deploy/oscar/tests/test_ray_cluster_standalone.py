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

import pytest
import mars

from .... import tensor as mt
from .... import dataframe as md
from ....tests.core import require_ray, mock
from ....utils import lazy_import
from ..ray import (
    new_cluster_in_ray,
    new_ray_session,
    _load_config,
    new_cluster,
)

ray = lazy_import("ray")


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
@pytest.mark.parametrize(
    "backend",
    [
        "mars",
        "ray",
    ],
)
def test_new_ray_session(stop_ray, backend):
    new_ray_session_test(backend)


def new_ray_session_test(backend):
    session = new_ray_session(
        session_id="abc", worker_num=2, worker_mem=512 * 1024**2, backend=backend
    )
    mt.random.RandomState(0).rand(100, 5).sum().execute()
    session.execute(mt.random.RandomState(0).rand(100, 5).sum())
    mars.execute(mt.random.RandomState(0).rand(100, 5).sum())
    session = new_ray_session(
        session_id="abcd",
        worker_num=2,
        default=True,
        worker_mem=512 * 1024**2,
        backend=backend,
    )
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
async def test_optional_supervisor_node(ray_start_regular, test_option):
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
@pytest.mark.asyncio
async def test_new_ray_session_config(stop_ray):
    original_placement_group = ray.util.placement_group
    with mock.patch.object(
        ray.util, "placement_group", autospec=True
    ) as mock_placement_group:

        def _wrap_original_placement_group(*args, **kwargs):
            assert {"CPU": 3} in kwargs["bundles"]
            return original_placement_group(*args, **kwargs)

        mock_placement_group.side_effect = _wrap_original_placement_group
        mars.new_ray_session(
            supervisor_cpu=3,
            worker_cpu=5,
            backend="ray",
            default=True,
            config={
                "third_party_modules": [
                    "mars.deploy.oscar.tests.modules.check_ray_remote_function_options"
                ]
            },
        )
        mt.random.RandomState(0).rand(100, 5).sum().execute()

        # It seems crashes CI.
        # mars.stop_server()
        #
        # actors = ray.state.actors()
        # assert len(actors) == 1
        # assert list(actors.values())[0]["State"] == "DEAD"

        mars.new_ray_session(
            supervisor_cpu=3,
            worker_cpu=4,
            backend="ray",
            default=True,
            config={
                "third_party_modules": [
                    "mars.deploy.oscar.tests.modules.check_ray_remote_function_options"
                ]
            },
        )
        with pytest.raises(AssertionError):
            mt.random.RandomState(0).rand(100, 5).sum().execute()
