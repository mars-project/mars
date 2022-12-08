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
import logging
import os

import numpy as np
import pytest
import pandas as pd
import time

from .... import dataframe as md
from .... import oscar as mo
from .... import tensor as mt
from ....oscar.backends.ray.utils import (
    process_address_to_placement,
    process_placement_to_address,
    kill_and_wait,
)
from ....services.cluster import ClusterAPI
from ....services.scheduling.supervisor.autoscale import AutoscalerActor
from ....tests.core import require_ray
from ....utils import lazy_import
from ..ray import new_cluster, _load_config
from ..tests import test_local

ray = lazy_import("ray")

logger = logging.getLogger(__name__)


@pytest.fixture
async def speculative_cluster():
    client = await new_cluster(
        "test_cluster",
        worker_num=5,
        worker_cpu=2,
        worker_mem=512 * 1024**2,
        supervisor_mem=100 * 1024**2,
        config={
            "scheduling": {
                "speculation": {
                    "enabled": True,
                    "dry": False,
                    "interval": 0.5,
                    "threshold": 0.2,
                    "min_task_runtime": 2,
                    "multiplier": 1.5,
                },
                # used to kill hanged subtask to release slot.
                "subtask_cancel_timeout": 0.1,
            },
        },
    )
    async with client:
        yield client


@pytest.mark.parametrize("ray_large_cluster", [{"num_nodes": 2}], indirect=True)
@pytest.mark.timeout(timeout=500)
@require_ray
@pytest.mark.asyncio
async def test_task_speculation_execution(ray_large_cluster, speculative_cluster):
    await test_local.test_task_speculation_execution(speculative_cluster)


@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 1, "num_cpus": 3}], indirect=True
)
@require_ray
@pytest.mark.asyncio
async def test_request_worker(ray_large_cluster):
    worker_cpu, worker_mem = 1, 100 * 1024**2
    client = await new_cluster(
        worker_num=0, worker_cpu=worker_cpu, worker_mem=worker_mem
    )
    async with client:
        cluster_state_ref = client._cluster._cluster_backend.get_cluster_state_ref()
        # Note that supervisor took one node
        workers = await asyncio.gather(
            *[cluster_state_ref.request_worker(timeout=5) for _ in range(2)]
        )
        assert all(worker is not None for worker in workers)
        assert not await cluster_state_ref.request_worker(timeout=5)
        release_workers = [
            cluster_state_ref.release_worker(worker) for worker in workers
        ]
        # Duplicate release workers requests should be handled.
        release_workers.extend(
            [cluster_state_ref.release_worker(worker) for worker in workers]
        )
        await asyncio.gather(*release_workers)
        assert await cluster_state_ref.request_worker(timeout=5)
        cluster_state_ref.reconstruct_worker()


@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 1, "num_cpus": 3}], indirect=True
)
@require_ray
@pytest.mark.asyncio
async def test_reconstruct_worker(ray_large_cluster):
    worker_cpu, worker_mem = 1, 100 * 1024**2
    client = await new_cluster(
        worker_num=0, worker_cpu=worker_cpu, worker_mem=worker_mem
    )
    async with client:
        cluster_api = await ClusterAPI.create(client._cluster.supervisor_address)
        worker = await cluster_api.request_worker(timeout=5)
        pg_name, bundle_index, process_index = process_address_to_placement(worker)
        worker_sub_pool = process_placement_to_address(
            pg_name, bundle_index, process_index + 1
        )

        worker_actor = ray.get_actor(worker)
        worker_pid = await worker_actor.getpid.remote()
        # the worker pool actor should be destroyed even we get actor.
        worker_sub_pool_actor = ray.get_actor(worker_sub_pool)
        worker_sub_pool_pid = await worker_sub_pool_actor.getpid.remote()

        # kill worker main pool
        await kill_and_wait(ray.get_actor(worker))

        # duplicated reconstruct worker request can be handled.
        await asyncio.gather(
            cluster_api.reconstruct_worker(worker),
            cluster_api.reconstruct_worker(worker),
        )
        worker_actor = ray.get_actor(worker)
        new_worker_pid = await worker_actor.getpid.remote()
        worker_sub_pool_actor = ray.get_actor(worker_sub_pool)
        new_worker_sub_pool_pid = await worker_sub_pool_actor.getpid.remote()
        assert new_worker_pid != worker_pid
        assert new_worker_sub_pool_pid != worker_sub_pool_pid

        # the compute should be ok after the worker is reconstructed.
        raw = np.random.RandomState(0).rand(10, 5)
        a = mt.tensor(raw, chunk_size=5).sum(axis=1)
        b = a.execute(show_progress=False)
        assert b is a
        result = a.fetch()
        np.testing.assert_array_equal(result, raw.sum(axis=1))


@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 2, "num_cpus": 4}], indirect=True
)
@pytest.mark.parametrize("init_workers", [0, 1])
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_out(ray_large_cluster, init_workers: int):
    client = await new_cluster(
        worker_num=init_workers,
        worker_cpu=2,
        worker_mem=200 * 1024**2,
        supervisor_mem=1 * 1024**3,
        config={
            "scheduling.autoscale.enabled": True,
            "scheduling.autoscale.scheduler_backlog_timeout": 1,
            "scheduling.autoscale.worker_idle_timeout": 10000000,
            "scheduling.autoscale.max_workers": 10,
        },
    )
    async with client:

        def time_consuming(x):
            time.sleep(1)
            return x * x

        series_size = 100
        assert (
            md.Series(list(range(series_size)), chunk_size=1)
            .apply(time_consuming)
            .sum()
            .execute()
            .fetch()
            == pd.Series(list(range(series_size))).apply(lambda x: x * x).sum()
        )
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(),
            address=client._cluster.supervisor_address,
        )
        assert await autoscaler_ref.get_dynamic_worker_nums() > 0


@pytest.mark.timeout(timeout=600)
@pytest.mark.parametrize(
    "ray_large_cluster", [{"num_nodes": 2, "num_cpus": 4}], indirect=True
)
@require_ray
@pytest.mark.asyncio
async def test_auto_scale_in(ray_large_cluster):
    config = _load_config()
    config["scheduling"]["autoscale"]["enabled"] = True
    config["scheduling"]["autoscale"]["worker_idle_timeout"] = 1
    config["scheduling"]["autoscale"]["max_workers"] = 4
    config["scheduling"]["autoscale"]["min_workers"] = 2
    client = await new_cluster(
        worker_num=0,
        worker_cpu=2,
        worker_mem=200 * 1024**2,
        supervisor_mem=1 * 1024**3,
        config=config,
    )
    async with client:
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(),
            address=client._cluster.supervisor_address,
        )
        new_worker_nums = 3
        await asyncio.gather(
            *[autoscaler_ref.request_worker() for _ in range(new_worker_nums)]
        )
        series_size = 100
        assert (
            md.Series(list(range(series_size)), chunk_size=20).sum().execute().fetch()
            == pd.Series(list(range(series_size))).sum()
        )
        while await autoscaler_ref.get_dynamic_worker_nums() > 2:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            logger.info(f"Waiting %s workers to be released.", dynamic_workers)
            await asyncio.sleep(1)
        await asyncio.sleep(1)
        assert await autoscaler_ref.get_dynamic_worker_nums() == 2


@pytest.mark.timeout(timeout=500)
@pytest.mark.parametrize("ray_large_cluster", [{"num_nodes": 4}], indirect=True)
@require_ray
@pytest.mark.asyncio
async def test_ownership_when_scale_in(ray_large_cluster):
    client = await new_cluster(
        worker_num=0,
        worker_cpu=2,
        worker_mem=1 * 1024**3,
        supervisor_mem=200 * 1024**2,
        config={
            "scheduling.autoscale.enabled": True,
            "scheduling.autoscale.scheduler_check_interval": 0.1,
            "scheduling.autoscale.scheduler_backlog_timeout": 0.5,
            "scheduling.autoscale.worker_idle_timeout": 1,
            "scheduling.autoscale.min_workers": 1,
            "scheduling.autoscale.max_workers": 4,
        },
    )
    async with client:
        autoscaler_ref = mo.create_actor_ref(
            uid=AutoscalerActor.default_uid(),
            address=client._cluster.supervisor_address,
        )
        num_chunks, chunk_size = 10, 4
        df = md.DataFrame(
            mt.random.rand(num_chunks * chunk_size, 4, chunk_size=chunk_size),
            columns=list("abcd"),
        )
        latch_actor = ray.remote(CountDownLatch).remote(1)
        pid = os.getpid()

        def f(pdf, latch):
            if os.getpid() != pid:
                # type inference will call this function too
                ray.get(latch.wait.remote())
            return pdf

        df = df.map_chunk(
            f,
            args=(latch_actor,),
        )
        info = df.execute(wait=False)
        while await autoscaler_ref.get_dynamic_worker_nums() <= 1:
            logger.info("Waiting workers to be created.")
            await asyncio.sleep(1)
        await latch_actor.count_down.remote()
        await info
        assert info.exception() is None
        assert info.progress() == 1
        logger.info("df execute succeed.")

        while await autoscaler_ref.get_dynamic_worker_nums() > 1:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            logger.info("Waiting workers %s to be released.", dynamic_workers)
            await asyncio.sleep(1)
        # Test data on node of released worker can still be fetched
        pd_df = df.fetch()
        groupby_sum_df = (
            df.rechunk(chunk_size * 2).groupby("a").apply(lambda pdf: pdf.sum())
        )
        logger.info(groupby_sum_df.execute())
        while await autoscaler_ref.get_dynamic_worker_nums() > 1:
            dynamic_workers = await autoscaler_ref.get_dynamic_workers()
            logger.info(f"Waiting workers %s to be released.", dynamic_workers)
            await asyncio.sleep(1)
        assert df.to_pandas().to_dict() == pd_df.to_dict()
        assert (
            groupby_sum_df.to_pandas().to_dict()
            == pd_df.groupby("a").apply(lambda pdf: pdf.sum()).to_dict()
        )


class CountDownLatch:
    def __init__(self, cnt):
        self.cnt = cnt

    def count_down(self):
        self.cnt -= 1

    def get_count(self):
        return self.cnt

    async def wait(self):
        while self.cnt != 0:
            await asyncio.sleep(0.01)
