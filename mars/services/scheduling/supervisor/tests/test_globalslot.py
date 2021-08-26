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

import mars.oscar as mo
from mars.services.cluster import ClusterAPI, MockClusterAPI
from mars.services.session import MockSessionAPI
from mars.services.scheduling.supervisor import GlobalSlotManagerActor


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        session_id = 'test_session'
        await MockClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)

        global_slot_ref = await mo.create_actor(
            GlobalSlotManagerActor, uid=GlobalSlotManagerActor.default_uid(),
            address=pool.external_address)

        yield pool, session_id, global_slot_ref

        await mo.destroy_actor(global_slot_ref)


@pytest.mark.asyncio
async def test_global_slot(actor_pool):
    pool, session_id, global_slot_ref = actor_pool

    cluster_api = await ClusterAPI.create(pool.external_address)
    bands = await cluster_api.get_all_bands()
    band = (pool.external_address, 'numa-0')
    band_slots = bands[band]

    assert ['subtask0'] == await global_slot_ref.apply_subtask_slots(
        band, session_id, ['subtask0'], [1])

    await global_slot_ref.update_subtask_slots(
        band, session_id, 'subtask0', band_slots)
    assert [] == await global_slot_ref.apply_subtask_slots(
        band, session_id, ['subtask1'], [1])

    await global_slot_ref.release_subtask_slots(
        band, session_id, 'subtask0')
    assert ['subtask1'] == await global_slot_ref.apply_subtask_slots(
        band, session_id, ['subtask1'], [1])
