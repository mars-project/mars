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

import numpy as np
import pandas as pd
import pytest

import mars.oscar as mo
import mars.tensor as mt
import mars.dataframe as md
from mars.core.session import get_default_session, new_session
from mars.deploy.oscar.ray import new_cluster, _load_config
from mars.deploy.oscar.tests import test_local
from mars.serialization.ray import register_ray_serializers
from mars.tests.conftest import *  # noqa
from mars.tests.core import require_ray
from mars.utils import lazy_import

ray = lazy_import('ray')


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
async def test_execute(ray_large_cluster, create_cluster):
    await test_local.test_execute(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_iterative_tiling(ray_large_cluster, create_cluster):
    await test_local.test_iterative_tiling(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_execute_describe(ray_large_cluster, create_cluster):
    # await test_local.test_execute_describe(create_cluster)
    r = md.Series(list(range(8)), chunk_size=1).apply(lambda x: x*x).sum()
    print(create_cluster.session)
    info = await create_cluster.session.execute(r)
    print(f'info {info} {info.ex}')
    print(await create_cluster.session.fetch(r))


@require_ray
@pytest.mark.asyncio
def test_sync_execute(ray_large_cluster, create_cluster):
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


def _run_web_session(web_address):
    register_ray_serializers()
    import asyncio
    asyncio.new_event_loop().run_until_complete(
        test_local._run_web_session_test(web_address))
    return True


def _sync_web_session_test(web_address):
    register_ray_serializers()
    new_session(web_address, backend='oscar', default=True)
    raw = np.random.RandomState(0).rand(10, 5)
    a = mt.tensor(raw, chunk_size=5).sum(axis=1)
    b = a.execute(show_progress=False)
    assert b is a
    return True


@require_ray
@pytest.mark.parametrize('test_option',
                         [[True, 0, ['ray://test_cluster/1/0', 'ray://test_cluster/2/0']],
                          [False, 0, ['ray://test_cluster/0/1', 'ray://test_cluster/1/0']],
                          [True, 2, ['ray://test_cluster/1/0', 'ray://test_cluster/2/0']],
                          [False, 5, ['ray://test_cluster/0/6', 'ray://test_cluster/1/0']]])
@pytest.mark.asyncio
async def test_optional_supervisor_node(ray_large_cluster, test_option):
    import logging
    logging.basicConfig(level=logging.INFO)
    supervisor_standalone, supervisor_sub_pool_num, worker_addresses = test_option
    config = _load_config()
    config['cluster']['ray']['supervisor']['standalone'] = supervisor_standalone
    config['cluster']['ray']['supervisor']['sub_pool_num'] = supervisor_sub_pool_num
    client = await new_cluster('test_cluster',
                               worker_num=2,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3,
                               config=config)
    async with client:
        assert client.address == 'ray://test_cluster/0/0'
        assert client._cluster._worker_addresses == worker_addresses


@require_ray
@pytest.mark.asyncio
async def test_web_session(ray_large_cluster, create_cluster):
    await test_local.test_web_session(create_cluster)
    web_address = create_cluster.web_address
    assert await ray.remote(_run_web_session).remote(web_address)
    assert await ray.remote(_sync_web_session_test).remote(web_address)


@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 10}], indirect=True)
@pytest.mark.parametrize('init_workers', [0, 1])
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_out(ray_large_cluster, init_workers: int):
    config = _load_config()
    config['scheduling']['autoscale']['enable'] = True
    config['scheduling']['autoscale']['scheduler_check_interval'] = 1
    config['scheduling']['autoscale']['scheduler_backlog_timeout'] = 1
    config['scheduling']['autoscale']['sustained_scheduler_backlog_timeout'] = 1
    config['scheduling']['autoscale']['worker_idle_timeout'] = 10000000
    config['scheduling']['autoscale']['max_workers'] = 9
    client = await new_cluster('test_cluster',
                               worker_num=init_workers,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config=config)
    async with client:
        def time_consuming(x):
            time.sleep(2)
            return x * x

        series_size = 100
        assert md.Series(list(range(series_size)), chunk_size=1).apply(time_consuming).sum().execute().fetch() ==\
               pd.Series(list(range(series_size))).apply(lambda x: x*x).sum()
        from mars.services.scheduling.supervisor.autoscale import AutoscalerActor
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        assert await autoscaler_ref.get_dynamic_worker_nums() > 0


@pytest.mark.timeout(timeout=60)
@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 4}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_in(ray_large_cluster):
    config = _load_config()
    config['scheduling']['autoscale']['enable'] = True
    config['scheduling']['autoscale']['scheduler_check_interval'] = 1
    config['scheduling']['autoscale']['worker_idle_timeout'] = 1
    config['scheduling']['autoscale']['max_workers'] = 4
    config['scheduling']['autoscale']['min_workers'] = 0
    client = await new_cluster('test_cluster',
                               worker_num=0,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config=config)
    async with client:
        from mars.services.scheduling.supervisor.autoscale import AutoscalerActor
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        new_worker_nums = 3
        await asyncio.gather(*[autoscaler_ref.request_worker_node() for _ in range(new_worker_nums)])
        series_size = 100
        assert md.Series(list(range(series_size)), chunk_size=20).sum().execute().fetch() == \
               pd.Series(list(range(series_size))).sum()
        while await autoscaler_ref.get_dynamic_worker_nums() > 0:
            print(f'Waiting workers {await autoscaler_ref.get_dynamic_workers()} to be released.')
            await asyncio.sleep(1)


@pytest.mark.timeout(timeout=60)
@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 4}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_ownership_when_scale_in(ray_large_cluster):
    config = _load_config()
    config['scheduling']['autoscale']['enable'] = True
    config['scheduling']['autoscale']['scheduler_check_interval'] = 1
    config['scheduling']['autoscale']['scheduler_backlog_timeout'] = 1
    config['scheduling']['autoscale']['worker_idle_timeout'] = 5
    config['scheduling']['autoscale']['max_workers'] = 4
    config['scheduling']['autoscale']['min_workers'] = 0
    client = await new_cluster('test_cluster',
                               worker_num=0,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config=config)
    async with client:
        from mars.services.scheduling.supervisor.autoscale import AutoscalerActor
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        df = md.DataFrame(mt.random.rand(100, 4, chunk_size=10), columns=list('abcd'))
        print(df.sum().execute())
        assert await autoscaler_ref.get_dynamic_worker_nums() > 0
        while await autoscaler_ref.get_dynamic_worker_nums() > 0:
            print(f'Waiting workers {await autoscaler_ref.get_dynamic_workers()} to be released.')
            await asyncio.sleep(1)
        # Test data on node of released worker can still be fetched
        r = df.groupby('a').sum().execute()
        print(r, type(r))
