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

from ..... import oscar as mo
from .....core import ChunkGraph
from .....tensor.fetch import TensorFetch
from .....tensor.arithmetic import TensorTreeAdd
from ....cluster import ClusterAPI
from ....cluster.core import NodeRole, NodeStatus
from ....cluster.uploader import NodeInfoUploaderActor
from ....cluster.supervisor.locator import SupervisorPeerLocatorActor
from ....cluster.supervisor.node_info import NodeInfoCollectorActor
from ....meta import MockMetaAPI
from ....session import MockSessionAPI
from ....subtask import Subtask
from ...supervisor import AssignerActor
from ...errors import NoMatchingSlots


class MockNodeInfoCollectorActor(NodeInfoCollectorActor):
    def __init__(self, timeout=None, check_interval=None, with_gpu=False):
        super().__init__(timeout=timeout, check_interval=check_interval)
        self.ready_bands = {('address0', 'numa-0'): 2,
                            ('address1', 'numa-0'): 2,
                            ('address2', 'numa-0'): 2}
        if with_gpu:
            self.ready_bands[('address0', 'gpu-0')] = 1
        self.all_bands = self.ready_bands.copy()

    async def update_node_info(self, address, role, env=None,
                               resource=None, detail=None, status=None):
        if 'address' in address and status == NodeStatus.STOPPING:
            del self.ready_bands[(address, 'numa-0')]
        await super().update_node_info(address, role, env,
                                       resource, detail, status)

    def get_all_bands(self, role=None, statuses=None):
        if statuses == {NodeStatus.READY}:
            return self.ready_bands
        else:
            return self.all_bands


class FakeClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls, address: str, **kw):
        dones, _ = await asyncio.wait([
            mo.create_actor(SupervisorPeerLocatorActor, 'fixed', address,
                            uid=SupervisorPeerLocatorActor.default_uid(),
                            address=address),
            mo.create_actor(MockNodeInfoCollectorActor,
                            with_gpu=kw.get('with_gpu', False),
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
async def actor_pool(request):
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)
    with_gpu = request.param

    async with pool:
        session_id = 'test_session'
        await FakeClusterAPI.create(
            pool.external_address, with_gpu=with_gpu)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        assigner_ref = await mo.create_actor(
            AssignerActor, session_id, uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address)

        yield pool, session_id, assigner_ref, meta_api

        await mo.destroy_actor(assigner_ref)


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [False], indirect=True)
async def test_assign_cpu_tasks(actor_pool):
    pool, session_id, assigner_ref, meta_api = actor_pool

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

    subtask = Subtask('test_task', session_id, chunk_graph=chunk_graph)
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result in (('address1', 'numa-0'), ('address2', 'numa-0'))

    result_chunk.op.gpu = True
    subtask = Subtask('test_task', session_id, chunk_graph=chunk_graph)
    with pytest.raises(NoMatchingSlots) as err:
        await assigner_ref.assign_subtasks([subtask])
    assert 'gpu' in str(err.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [True], indirect=True)
async def test_assign_gpu_tasks(actor_pool):
    pool, session_id, assigner_ref, meta_api = actor_pool

    input1 = TensorFetch(key='a', source_key='a', dtype=np.dtype(int)).new_chunk([])
    input2 = TensorFetch(key='b', source_key='b', dtype=np.dtype(int)).new_chunk([])
    result_chunk = TensorTreeAdd(args=[input1, input2], gpu=True) \
        .new_chunk([input1, input2])

    chunk_graph = ChunkGraph([result_chunk])
    chunk_graph.add_node(input1)
    chunk_graph.add_node(input2)
    chunk_graph.add_node(result_chunk)
    chunk_graph.add_edge(input1, result_chunk)
    chunk_graph.add_edge(input2, result_chunk)

    await meta_api.set_chunk_meta(input1, memory_size=200, store_size=200,
                                  bands=[('address0', 'numa-0')])
    await meta_api.set_chunk_meta(input2, memory_size=200, store_size=200,
                                  bands=[('address0', 'numa-0')])

    subtask = Subtask('test_task', session_id, chunk_graph=chunk_graph)
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result[1].startswith('gpu')
