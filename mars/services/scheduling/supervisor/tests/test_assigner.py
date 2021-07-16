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
import pytest

import mars.oscar as mo
from mars.core import ChunkGraph
from mars.services.cluster import MockClusterAPI
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.scheduling.supervisor import AssignerActor, GlobalSlotManagerActor
from mars.services.subtask import Subtask
from mars.tensor.fetch import TensorFetch
from mars.tensor.arithmetic import TensorTreeAdd


class MockSlotsActor(mo.Actor):
    def watch_available_bands(self):
        return {('address0', 'numa-0'): 2,
                ('address2', 'numa-0'): 2}

    def get_available_bands(self):
        return {('address0', 'numa-0'): 2,
                ('address2', 'numa-0'): 2}


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool('127.0.0.1', n_process=0)

    async with pool:
        session_id = 'test_session'
        await MockClusterAPI.create(pool.external_address)
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        await mo.create_actor(MockSlotsActor,
                              uid=GlobalSlotManagerActor.default_uid(),
                              address=pool.external_address)
        assigner_ref = await mo.create_actor(
            AssignerActor, session_id, uid=AssignerActor.gen_uid(session_id),
            address=pool.external_address)

        yield pool, session_id, assigner_ref, meta_api

        await mo.destroy_actor(assigner_ref)


@pytest.mark.asyncio
async def test_assigner(actor_pool):
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
    assert result in (('address0', 'numa-0'), ('address2', 'numa-0'))

    subtask.expect_bands = [('address1', 'numa-0')]
    [result] = await assigner_ref.assign_subtasks([subtask])
    assert result in (('address0', 'numa-0'), ('address2', 'numa-0'))

    band_num_queued_subtasks = {('address0', 'numa-0'): 9, ('address1', 'numa-0'): 8,
                                ('address2', 'numa-0'): 0}
    move_queued_subtasks = await assigner_ref.reassign_subtasks(band_num_queued_subtasks)
    assert move_queued_subtasks in ({('address1', 'numa-0'): -8,
                                     ('address0', 'numa-0'): -1,
                                     ('address2', 'numa-0'): 9},
                                    {('address1', 'numa-0'): -8,
                                     ('address0', 'numa-0'): 0,
                                     ('address2', 'numa-0'): 8})
