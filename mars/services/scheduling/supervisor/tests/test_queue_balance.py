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
from typing import Tuple, List

from ..... import oscar as mo
from ....cluster import ClusterAPI
from ....cluster.core import NodeRole, NodeStatus
from ....cluster.uploader import NodeInfoUploaderActor
from ....cluster.supervisor.locator import SupervisorPeerLocatorActor
from ....cluster.supervisor.node_info import NodeInfoCollectorActor
from ....subtask import Subtask
from ...supervisor import (
    AssignerActor,
    SubtaskManagerActor,
    SubtaskQueueingActor,
    GlobalSlotManagerActor,
)


class MockNodeInfoCollectorActor(NodeInfoCollectorActor):
    def __init__(self, timeout=None, check_interval=None):
        super().__init__(timeout=timeout, check_interval=check_interval)
        self.ready_nodes = {
            ("address0", "numa-0"): 2,
            ("address1", "numa-0"): 2,
            ("address2", "numa-0"): 2,
        }

    async def update_node_info(
        self, address, role, env=None, resource=None, detail=None, status=None
    ):
        if "address" in address and status == NodeStatus.STOPPING:
            del self.ready_nodes[(address, "numa-0")]
        await super().update_node_info(address, role, env, resource, detail, status)

    def get_all_bands(self, role=None, statuses=None):
        if statuses == {NodeStatus.READY}:
            return self.ready_nodes
        else:
            return {
                ("address0", "numa-0"): 2,
                ("address1", "numa-0"): 2,
                ("address2", "numa-0"): 2,
            }


class FakeClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls, address: str, **kw):
        dones, _ = await asyncio.wait(
            [
                mo.create_actor(
                    SupervisorPeerLocatorActor,
                    "fixed",
                    address,
                    uid=SupervisorPeerLocatorActor.default_uid(),
                    address=address,
                ),
                mo.create_actor(
                    MockNodeInfoCollectorActor,
                    uid=NodeInfoCollectorActor.default_uid(),
                    address=address,
                ),
                mo.create_actor(
                    NodeInfoUploaderActor,
                    NodeRole.WORKER,
                    interval=kw.get("upload_interval"),
                    band_to_slots=kw.get("band_to_slots"),
                    use_gpu=kw.get("use_gpu", False),
                    uid=NodeInfoUploaderActor.default_uid(),
                    address=address,
                ),
            ]
        )

        for task in dones:
            try:
                task.result()
            except mo.ActorAlreadyExist:  # pragma: no cover
                pass

        api = await super().create(address=address)
        await api.mark_node_ready()
        return api


class MockSlotsActor(mo.Actor):
    def apply_subtask_slots(
        self,
        band: Tuple,
        session_id: str,
        subtask_ids: List[str],
        subtask_slots: List[int],
    ):
        return subtask_ids


class MockAssignerActor(mo.Actor):
    def assign_subtasks(
        self, subtasks: List[Subtask], exclude_bands=None, exclude_bands_force=False
    ):
        return [subtask.expect_bands[0] for subtask in subtasks]

    def reassign_subtasks(self, band_num_queued_subtasks):
        if len(band_num_queued_subtasks.keys()) == 1:
            [(band, _)] = band_num_queued_subtasks.items()
            return {band: 0}
        return {
            ("address1", "numa-0"): -8,
            ("address0", "numa-0"): 0,
            ("address2", "numa-0"): 8,
        }


class MockSubtaskManagerActor(mo.Actor):
    def __init__(self):
        self._subtask_ids, self._bands = [], []

    @mo.extensible
    def submit_subtask_to_band(self, subtask_id: str, band: Tuple):
        self._subtask_ids.append(subtask_id)
        self._bands.append(band)

    def dump_data(self):
        return self._subtask_ids, self._bands


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)

    async with pool:
        session_id = "test_session"
        cluster_api = await FakeClusterAPI.create(pool.external_address)

        # create assigner actor
        await mo.create_actor(
            MockAssignerActor,
            uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        # create queueing actor
        manager_ref = await mo.create_actor(
            MockSubtaskManagerActor,
            uid=SubtaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )
        # create slots actor
        slots_ref = await mo.create_actor(
            MockSlotsActor,
            uid=GlobalSlotManagerActor.default_uid(),
            address=pool.external_address,
        )
        # create queueing actor
        queueing_ref = await mo.create_actor(
            SubtaskQueueingActor,
            session_id,
            1,
            uid=SubtaskQueueingActor.gen_uid(session_id),
            address=pool.external_address,
        )

        yield pool, session_id, cluster_api, queueing_ref, slots_ref, manager_ref

        await mo.destroy_actor(queueing_ref)


async def _queue_subtasks(num_subtasks, expect_bands, queueing_ref):
    if not num_subtasks:
        return
    subtasks = [Subtask(expect_bands[0] + "-" + str(i)) for i in range(num_subtasks)]
    for subtask in subtasks:
        subtask.expect_bands = [expect_bands]
    priorities = [(i,) for i in range(num_subtasks)]

    await queueing_ref.add_subtasks(subtasks, priorities)


@pytest.mark.asyncio
async def test_subtask_queueing(actor_pool):
    _pool, session_id, cluster_api, queueing_ref, slots_ref, manager_ref = actor_pool
    nums_subtasks = [9, 8, 1]
    expects_bands = [
        ("address0", "numa-0"),
        ("address1", "numa-0"),
        ("address2", "numa-0"),
    ]
    for num_subtasks, expect_bands in zip(nums_subtasks, expects_bands):
        await _queue_subtasks(num_subtasks, expect_bands, queueing_ref)

    await cluster_api.set_node_status(
        node="address1", role=NodeRole.WORKER, status=NodeStatus.STOPPING
    )

    # 9 subtasks on ('address0', 'numa-0')
    await queueing_ref.submit_subtasks(band=("address0", "numa-0"), limit=10)
    commited_subtask_ids, _commited_bands = await manager_ref.dump_data()
    assert len(commited_subtask_ids) == 9

    # 0 subtasks on ('address1', 'numa-0')
    await queueing_ref.submit_subtasks(band=("address1", "numa-0"), limit=10)
    commited_subtask_ids, _commited_bands = await manager_ref.dump_data()
    assert len(commited_subtask_ids) == 9

    # 9 subtasks on ('address2', 'numa-0')
    await queueing_ref.submit_subtasks(band=("address2", "numa-0"), limit=10)
    commited_subtask_ids, _commited_bands = await manager_ref.dump_data()
    assert len(commited_subtask_ids) == 18
