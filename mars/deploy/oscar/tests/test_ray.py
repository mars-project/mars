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
from mars.deploy.oscar.session import Session
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
async def mars_cluster():
    client = await new_cluster('test_cluster',
                               worker_num=2,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3)
    async with client:
        yield client


@require_ray
@pytest.mark.asyncio
async def test_execute(ray_cluster, mars_cluster):
    await test_local.test_execute(mars_cluster)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_cluster, mars_cluster):
    await test_local.test_iterative_tiling(mars_cluster)


@require_ray
@pytest.mark.asyncio
def test_sync_execute(ray_cluster, mars_cluster):
    assert mars_cluster.session
    session = new_session(address=mars_cluster.address, backend='oscar', default=True)
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
async def test_web_session(ray_cluster, mars_cluster):
    import uuid
    session_id = str(uuid.uuid4())
    supervisor_address, web_address = mars_cluster.cluster.supervisor_address, mars_cluster.cluster.web_address
    session = await Session.init(supervisor_address, session_id, web_api=True, web_address=web_address)
    session.as_default()
    await test_local.test_execute(mars_cluster)
    await test_local.test_iterative_tiling(mars_cluster)
    Session.reset_default()
    await session.destroy()

    await _web_session_test(supervisor_address, web_address)
    await ray.remote(_run_web_session).remote(supervisor_address, web_address)


def _run_web_session(supervisor_address, web_address):
    register_ray_serializers()
    import asyncio
    asyncio.new_event_loop().run_until_complete(_web_session_test(supervisor_address, web_address))


async def _web_session_test(supervisor_address, web_address):
    import uuid
    session_id = str(uuid.uuid4())
    session = await Session.init(supervisor_address, session_id, web_api=True, web_address=web_address)
    session.as_default()
    raw = np.random.RandomState(0).rand(10, 10)
    a = mt.tensor(raw, chunk_size=5)
    b = a + 1

    info = await session.execute(b)
    await info
    assert info.result() is None
    assert info.exception() is None
    assert info.progress() == 1
    np.testing.assert_equal(raw + 1, (await session.fetch(b))[0])
    Session.reset_default()
    await session.destroy()
