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

import os
import numpy as np
import pandas as pd
import pytest

from .... import oscar as mo
from .... import tensor as mt
from .... import dataframe as md
from ....serialization.ray import register_ray_serializers
from ....services.scheduling.supervisor.autoscale import AutoscalerActor
from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import new_cluster, _load_config
from ..session import get_default_session, new_session
from ..tests import test_local
from .modules.utils import (  # noqa: F401; pylint: disable=unused-variable
    cleanup_third_party_modules_output,
    get_output_filenames,
)

ray = lazy_import('ray')

CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), 'local_test_with_ray_config.yml')
CONFIG_THIRD_PARTY_MODULES_TEST_FILE = os.path.join(
    os.path.dirname(__file__), 'ray_test_with_third_parity_modules_config.yml')


@pytest.fixture
async def create_cluster(request):
    param = getattr(request, "param", {})
    ray_config = _load_config(CONFIG_FILE)
    ray_config.update(param.get('config', {}))
    client = await new_cluster('test_cluster',
                               worker_num=2,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3,
                               config=ray_config)
    async with client:
        yield client, param


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
    await test_local.test_execute_describe(create_cluster)


@require_ray
@pytest.mark.asyncio
async def test_fetch_infos(ray_large_cluster, create_cluster):
    await test_local.test_fetch_infos(create_cluster)


@require_ray
@pytest.mark.asyncio
def test_sync_execute(ray_large_cluster, create_cluster):
    client = create_cluster[0]
    assert client.session
    session = new_session(address=client.address, backend='oscar')
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
    new_session(web_address, backend='oscar')
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
    client = create_cluster[0]
    await test_local.test_web_session(create_cluster)
    web_address = client.web_address
    assert await ray.remote(_run_web_session).remote(web_address)
    assert await ray.remote(_sync_web_session_test).remote(web_address)


@require_ray
@pytest.mark.parametrize('config_exception',
                         [[set(),
                           pytest.raises(TypeError, match='set')],
                          [{'supervisor': ['not_exists_for_supervisor']},
                           pytest.raises(ModuleNotFoundError, match='not_exists_for_supervisor')],
                          [{'worker': ['not_exists_for_worker']},
                           pytest.raises(ModuleNotFoundError, match='not_exists_for_worker')]])
@pytest.mark.asyncio
async def test_load_third_party_modules(ray_large_cluster, config_exception):
    third_party_modules_config, expected_exception = config_exception
    config = _load_config()

    config['third_party_modules'] = third_party_modules_config
    with expected_exception:
        await new_cluster('test_cluster',
                          worker_num=2,
                          worker_cpu=2,
                          worker_mem=1 * 1024 ** 3,
                          config=config)


@require_ray
@pytest.mark.parametrize('create_cluster',
                         [{
                             'config': {
                                 'third_party_modules': {
                                     'worker': ['mars.deploy.oscar.tests.modules.replace_op']},
                             },
                         }],
                         indirect=True)
@pytest.mark.asyncio
def test_load_third_party_modules2(ray_large_cluster, create_cluster):
    client = create_cluster[0]
    assert client.session
    session = new_session(address=client.address, backend='oscar')
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
async def test_load_third_party_modules_from_config(ray_large_cluster,
                                                    cleanup_third_party_modules_output):  # noqa: F811
    client = await new_cluster('test_cluster',
                               worker_num=2,
                               worker_cpu=2,
                               worker_mem=1 * 1024 ** 3,
                               config=CONFIG_THIRD_PARTY_MODULES_TEST_FILE)
    async with client:
        # 1 supervisor, 2 worker main pools, 4 worker sub pools.
        assert len(get_output_filenames()) == 7


@require_ray
def test_load_config():
    default_config = _load_config()
    assert default_config['scheduling']['autoscale']['enabled'] is False
    default_config = _load_config({'scheduling': {'autoscale': {'enabled': True}}})
    assert default_config['scheduling']['autoscale']['enabled'] is True
    default_config = _load_config({
        'scheduling.autoscale.enabled': True,
        'scheduling.autoscale.scheduler_backlog_timeout': 1
    })
    assert default_config['scheduling']['autoscale']['enabled'] is True
    assert default_config['scheduling']['autoscale']['scheduler_backlog_timeout'] == 1
    with pytest.raises(ValueError):
        _load_config({
            'scheduling.autoscale.enabled': True,
            'scheduling.autoscale': {}
        })
    assert _load_config(CONFIG_FILE)['session']['custom_log_dir'] == 'auto'


@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 3, 'num_cpus': 1}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_request_worker(ray_large_cluster):
    worker_cpu, worker_mem = 1, 100 * 1024 ** 2
    client = await new_cluster('test_cluster', worker_num=0, worker_cpu=worker_cpu, worker_mem=worker_mem)
    async with client:
        cluster_state_ref = client._cluster._cluster_backend.get_cluster_state_ref()
        # Note that supervisor took one node
        workers = await asyncio.gather(*[cluster_state_ref.request_worker(timeout=5) for _ in range(2)])
        assert all(worker is not None for worker in workers)
        assert not await cluster_state_ref.request_worker(timeout=5)
        await asyncio.gather(*[cluster_state_ref.release_worker(worker) for worker in workers])
        assert await cluster_state_ref.request_worker(timeout=5)


@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 4, 'num_cpus': 2}], indirect=True)
@pytest.mark.parametrize('init_workers', [0, 1])
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_out(ray_large_cluster, init_workers: int):
    client = await new_cluster('test_cluster',
                               worker_num=init_workers,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config={
                                   'scheduling.autoscale.enabled': True,
                                   'scheduling.autoscale.scheduler_backlog_timeout': 1,
                                   'scheduling.autoscale.worker_idle_timeout': 10000000,
                                   'scheduling.autoscale.max_workers': 10
                               })
    async with client:
        def time_consuming(x):
            time.sleep(1)
            return x * x

        series_size = 100
        assert md.Series(list(range(series_size)), chunk_size=1).apply(time_consuming).sum().execute().fetch() ==\
               pd.Series(list(range(series_size))).apply(lambda x: x*x).sum()
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        assert await autoscaler_ref.get_dynamic_worker_nums() > 0


@pytest.mark.timeout(timeout=120)
@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 4}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_in(ray_large_cluster):
    config = _load_config()
    config['scheduling']['autoscale']['enabled'] = True
    config['scheduling']['autoscale']['worker_idle_timeout'] = 1
    config['scheduling']['autoscale']['max_workers'] = 4
    config['scheduling']['autoscale']['min_workers'] = 2
    client = await new_cluster('test_cluster',
                               worker_num=0,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config=config)
    async with client:
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        new_worker_nums = 3
        await asyncio.gather(*[autoscaler_ref.request_worker() for _ in range(new_worker_nums)])
        series_size = 100
        assert md.Series(list(range(series_size)), chunk_size=20).sum().execute().fetch() == \
               pd.Series(list(range(series_size))).sum()
        while await autoscaler_ref.get_dynamic_worker_nums() > 2:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            print(f'Waiting workers {dynamic_workers} to be released.')
            await asyncio.sleep(1)
        await asyncio.sleep(1)
        assert await autoscaler_ref.get_dynamic_worker_nums() == 2


@pytest.mark.timeout(timeout=120)
@pytest.mark.parametrize('ray_large_cluster', [{'num_nodes': 4}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_ownership_when_scale_in(ray_large_cluster):
    client = await new_cluster('test_cluster',
                               worker_num=0,
                               worker_cpu=2,
                               worker_mem=100 * 1024 ** 2,
                               config={
                                   'scheduling.autoscale.enabled': True,
                                   'scheduling.autoscale.scheduler_check_interval': 1,
                                   'scheduling.autoscale.scheduler_backlog_timeout': 1,
                                   'scheduling.autoscale.worker_idle_timeout': 10,
                                   'scheduling.autoscale.min_workers': 1,
                                   'scheduling.autoscale.max_workers': 4
                               })
    async with client:
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(), address=client._cluster.supervisor_address)
        await asyncio.gather(*[autoscaler_ref.request_worker() for _ in range(2)])
        df = md.DataFrame(mt.random.rand(100, 4, chunk_size=2), columns=list('abcd'))
        print(df.execute())
        assert await autoscaler_ref.get_dynamic_worker_nums() > 1
        while await autoscaler_ref.get_dynamic_worker_nums() > 1:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            print(f'Waiting workers {dynamic_workers} to be released.')
            await asyncio.sleep(1)
        # Test data on node of released worker can still be fetched
        pd_df = df.to_pandas()
        groupby_sum_df = df.rechunk(40).groupby('a').sum()
        print(groupby_sum_df.execute())
        while await autoscaler_ref.get_dynamic_worker_nums() > 1:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            print(f'Waiting workers {dynamic_workers} to be released.')
            await asyncio.sleep(1)
        assert df.to_pandas().to_dict() == pd_df.to_dict()
        assert groupby_sum_df.to_pandas().to_dict() == pd_df.groupby('a').sum().to_dict()
