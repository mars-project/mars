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

import pytest

from ..... import oscar as mo
from .....resource import Resource
from ....cluster import ClusterAPI, MockClusterAPI
from ....session import MockSessionAPI
from ...supervisor import GlobalResourceManagerActor


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        await MockClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)

        global_resource_ref = await mo.create_actor(
            GlobalResourceManagerActor,
            uid=GlobalResourceManagerActor.default_uid(),
            address=pool.external_address,
        )

        try:
            yield pool, session_id, global_resource_ref
        finally:
            await mo.destroy_actor(global_resource_ref)
            await MockClusterAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_global_resource(actor_pool):
    pool, session_id, global_resource_ref = actor_pool

    cluster_api = await ClusterAPI.create(pool.external_address)
    bands = await cluster_api.get_all_bands()
    band = (pool.external_address, "numa-0")
    band_resource = bands[band]

    assert band in await global_resource_ref.get_idle_bands(0)
    assert ["subtask0"] == await global_resource_ref.apply_subtask_resources(
        band, session_id, ["subtask0"], [Resource(num_cpus=1)]
    )
    assert band not in await global_resource_ref.get_idle_bands(0)

    await global_resource_ref.update_subtask_resources(
        band, session_id, "subtask0", band_resource
    )
    assert [] == await global_resource_ref.apply_subtask_resources(
        band, session_id, ["subtask1"], [Resource(num_cpus=1)]
    )

    wait_coro = global_resource_ref.wait_band_idle(band)
    (done, pending) = await asyncio.wait([wait_coro], timeout=0.5)
    assert not done
    await global_resource_ref.release_subtask_resource(band, session_id, "subtask0")
    (done, pending) = await asyncio.wait([wait_coro], timeout=0.5)
    assert done
    assert band in await global_resource_ref.get_idle_bands(0)
    assert ["subtask1"] == await global_resource_ref.apply_subtask_resources(
        band, session_id, ["subtask1"], [Resource(num_cpus=1)]
    )
    assert (await global_resource_ref.get_remaining_resources())[
        band
    ] == band_resource - Resource(num_cpus=1)
