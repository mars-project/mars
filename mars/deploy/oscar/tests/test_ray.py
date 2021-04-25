# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import numpy as np
import pytest

import mars.tensor as mt
from mars.core.session import get_default_session, new_session
from mars.deploy.oscar.ray import new_cluster
from mars.serialization.ray import register_ray_serializers
from mars.tests.core import require_ray
from ....utils import lazy_import
from . import test_local

ray = lazy_import('ray')


@pytest.fixture(scope="module")
def ray_cluster():
    try:
        from ray.cluster_utils import Cluster
    except ModuleNotFoundError:
        from ray._private.cluster_utils import Cluster
    cluster = Cluster()
    remote_nodes = []
    num_nodes = 1
    for i in range(num_nodes):
        remote_nodes.append(cluster.add_node(num_cpus=10))
        if len(remote_nodes) == 1:
            ray.init(address=cluster.address)
    yield
    ray.shutdown()


@pytest.fixture
async def create_cluster():
    client = await new_cluster('test_cluster',
                               worker_num=2,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3)
    async with client:
        yield client


@require_ray
@pytest.mark.asyncio
async def test_execute(ray_cluster, create_cluster):
    await test_local.test_execute(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_cluster, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@require_ray
@pytest.mark.asyncio
def test_sync_execute(ray_cluster, create_cluster):
    assert create_cluster.session
    session = new_session(address=create_cluster.address, backend='oscar', default=True)
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


@require_ray
@pytest.mark.asyncio
async def test_web_session(ray_cluster, create_cluster):
    await test_local.test_web_session(create_cluster)
    supervisor_address, web_address = create_cluster.supervisor_address, create_cluster.web_address
    assert await ray.remote(_run_web_session).remote(supervisor_address, web_address)
    assert await ray.remote(_sync_web_session_test).remote(supervisor_address, web_address)


def _run_web_session(supervisor_address, web_address):
    register_ray_serializers()
    import asyncio
    asyncio.new_event_loop().run_until_complete(test_local.web_session_test(supervisor_address, web_address))
    return True


def _sync_web_session_test(supervisor_address, web_address):
    register_ray_serializers()
    new_session(supervisor_address, backend='oscar', default=True, web_address=web_address)
    raw = np.random.RandomState(0).rand(10, 5)
    a = mt.tensor(raw, chunk_size=5).sum(axis=1)
    b = a.execute(show_progress=False)
    assert b is a
    return True
