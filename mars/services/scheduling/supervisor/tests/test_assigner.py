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

import numpy as np
import asyncio
import pytest

import mars.oscar as mo
from mars.core import ChunkGraph
from mars.services.cluster import ClusterAPI
from mars.services.cluster.core import NodeRole, NodeStatus
from mars.services.cluster.uploader import NodeInfoUploaderActor
from mars.services.cluster.supervisor.locator import SupervisorPeerLocatorActor
from mars.services.cluster.supervisor.node_info import NodeInfoCollectorActor
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.scheduling.supervisor import AssignerActor
from mars.services.subtask import Subtask
from mars.tensor.fetch import TensorFetch
from mars.tensor.arithmetic import TensorTreeAdd


class MockNodeInfoCollectorActor(NodeInfoCollectorActor):
    def __init__(self, timeout=None, check_interval=None):
        super().__init__(timeout=timeout, check_interval=check_interval)
        self.ready_nodes = {('address0', 'numa-0'): 2,
                            ('address1', 'numa-0'): 2,
                            ('address2', 'numa-0'): 2,
                            ('address3', 'numa-0'): 2}

    async def update_node_info(self, address, role, env=None,
                               resource=None, detail=None, status=None):
        if 'address' in address and status == NodeStatus.STOPPING:
            del self.ready_nodes[(address, 'numa-0')]
        await super().update_node_info(address, role, env,
                                       resource, detail, status)

    def get_all_bands(self, role=None, statuses=None):
        if statuses == {NodeStatus.READY}:
            return self.ready_nodes
        else:
            return {('address0', 'numa-0'): 2,
                    ('address1', 'numa-0'): 2,
                    ('address2', 'numa-0'): 2,
                    ('address3', 'numa-0'): 2}


class FakeClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls, address: str, **kw):
        dones, _ = await asyncio.wait([
            mo.create_actor(SupervisorPeerLocatorActor, 'fixed', address,
                            uid=SupervisorPeerLocatorActor.default_uid(),
                            address=address),
            mo.create_actor(MockNodeInfoCollectorActor,
                            uid=NodeInfoCollectorActor.default_uid(),
                            address=address),
            mo.create_actor(NodeInfoUploaderActor, NodeRole.WORKER,
                            interval=kw.get('upload_interval'),
                            band_to_slots=kw.get('band_to_slots'),
                            use_gpu=kw.get('use_gpu', False),
                            uid=NodeInfoUploaderActor.default_uid(),
                            address=address),
        ])

        for task in dones:
            try:
                task.result()
            except mo.ActorAlreadyExist:  # pragma: no cover
                pass

        api = await super().create(address=address)
        await api.mark_node_ready()
        return api


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        session_id = 'test_session'
        cluster_api = await FakeClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        assigner_ref = await mo.create_actor(
            AssignerActor, session_id, uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address)

        yield pool, session_id, assigner_ref, cluster_api, meta_api

        await mo.destroy_actor(assigner_ref)


@pytest.mark.asyncio
async def test_assigner(actor_pool):
    pool, session_id, assigner_ref, cluster_api, meta_api = actor_pool

    input1 = TensorFetch(key='a', source_key='a', dtype=np.dtype(int)).new_chunk([])
    input2 = TensorFetch(key='b', source_key='b', dtype=np.dtype(int)).new_chunk([])
    input3 = TensorFetch(key='c', source_key='c', dtype=np.dtype(int)).new_chunk([])
    result_chunk = TensorTreeAdd(args=[input1, input2, input3]) \
        .new_chunk([input1, input2, input3])

    chunk_graph = ChunkGraph([result_chunk])
    chunk_graph.add_node(input1)
    chunk_graph.add_node(input2)
    chunk_graph.add_node(input3)
    chunk_graph.add_node(result_chunk)
    chunk_graph.add_edge(input1, result_chunk)
    chunk_graph.add_edge(input2, result_chunk)
    chunk_graph.add_edge(input3, result_chunk)

    await meta_api.set_chunk_meta(input1, memory_size=200, store_size=200,
                                  bands=[('address0', 'numa-0')])
    await meta_api.set_chunk_meta(input2, memory_size=400, store_size=400,
                                  bands=[('address1', 'numa-0')])
    await meta_api.set_chunk_meta(input3, memory_size=400, store_size=400,
                                  bands=[('address2', 'numa-0')])

    await cluster_api.set_node_status(
        node='address1', role=NodeRole.WORKER, status=NodeStatus.STOPPING)
    await cluster_api.set_node_status(
        node='address3', role=NodeRole.WORKER, status=NodeStatus.STOPPING)

    subtask = Subtask('test_task', session_id, chunk_graph=chunk_graph)
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result in (('address0', 'numa-0'), ('address2', 'numa-0'))

    subtask.expect_bands = [('address0', 'numa-0')]
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result == ('address0', 'numa-0')

    subtask.expect_bands = [('address0', 'numa-0'), ('address1', 'numa-0')]
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result == ('address0', 'numa-0')

    subtask.expect_bands = [('address1', 'numa-0')]
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result in (('address0', 'numa-0'), ('address2', 'numa-0'))


@pytest.mark.asyncio
async def test_reassign_subtasks(actor_pool):
    pool, session_id, assigner_ref, cluster_api, meta_api = actor_pool

    # ('address0', 'numa-0'), ('address1', 'numa-0'), ('address2', 'numa-0') are ready
    await cluster_api.set_node_status(
        node='address3', role=NodeRole.WORKER, status=NodeStatus.STOPPING)

    band_num_queued_subtasks = {('address0', 'numa-0'): 3, ('address1', 'numa-0'): 4}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks in \
        ({('address1', 'numa-0'): -1, ('address0', 'numa-0'): -1, ('address2', 'numa-0'): 2},
         {('address1', 'numa-0'): -2, ('address0', 'numa-0'): 0,  ('address2', 'numa-0'): 2},
         {('address1', 'numa-0'): -2, ('address0', 'numa-0'): -1, ('address2', 'numa-0'): 3})

    # ('address0', 'numa-0'), ('address2', 'numa-0') are ready
    await cluster_api.set_node_status(
        node='address1', role=NodeRole.WORKER, status=NodeStatus.STOPPING)

    band_num_queued_subtasks = {('address0', 'numa-0'): 9, ('address1', 'numa-0'): 7,
                                ('address2', 'numa-0'): 0}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks in \
        ({('address1', 'numa-0'): -7, ('address0', 'numa-0'): 3, ('address2', 'numa-0'): 4},
         {('address1', 'numa-0'): -7, ('address0', 'numa-0'): 4, ('address2', 'numa-0'): 3})

    band_num_queued_subtasks = {('address0', 'numa-0'): 9, ('address1', 'numa-0'): 7}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks == \
        {('address1', 'numa-0'): -7, ('address0', 'numa-0'): -1, ('address2', 'numa-0'): 8}

    band_num_queued_subtasks = {('address1', 'numa-0'): 8}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks == \
        {('address1', 'numa-0'): -8, ('address0', 'numa-0'): 4, ('address2', 'numa-0'): 4}

    band_num_queued_subtasks = {('address1', 'numa-0'): 0}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks == {('address1', 'numa-0'): 0}

    # only ('address0', 'numa-0') is ready, i.e. there's only one band initially
    await cluster_api.set_node_status(
        node='address2', role=NodeRole.WORKER, status=NodeStatus.STOPPING)
    band_num_queued_subtasks = {('address0', 'numa-0'): 8}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks == {('address0', 'numa-0'): 0}

    band_num_queued_subtasks = {('address1', 'numa-0'): 8}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks == {('address1', 'numa-0'): -8, ('address0', 'numa-0'): 8}
