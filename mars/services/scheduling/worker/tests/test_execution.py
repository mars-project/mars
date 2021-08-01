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
import uuid
from contextlib import asynccontextmanager
from typing import Tuple

import numpy as np
import pytest

import mars.oscar as mo
import mars.remote as mr
from mars.core import ChunkGraph, ChunkGraphBuilder, \
    TileableGraph, TileableGraphBuilder
from mars.remote.core import RemoteFunction
from mars.services.scheduling.worker import SubtaskExecutionActor, \
    QuotaActor, BandSlotManagerActor
from mars.services.cluster import MockClusterAPI
from mars.services.lifecycle import MockLifecycleAPI
from mars.services.meta import MockMetaAPI
from mars.services.session import MockSessionAPI
from mars.services.storage import MockStorageAPI
from mars.services.storage.handler import StorageHandlerActor
from mars.services.subtask import MockSubtaskAPI, Subtask
from mars.services.task.supervisor.manager import TaskManagerActor
from mars.tensor.fetch import TensorFetch
from mars.tensor.arithmetic import TensorTreeAdd
from mars.utils import Timer


class CancelDetectActorMixin:
    @asynccontextmanager
    async def _delay_method(self):
        delay_fetch_time = getattr(self, '_delay_fetch_time', None)
        try:
            if delay_fetch_time is not None:
                await asyncio.sleep(delay_fetch_time)
            yield
        except asyncio.CancelledError:
            self._is_cancelled = True
            raise

    def set_delay_fetch_time(self, delay: float):
        setattr(self, '_delay_fetch_time', delay)

    def get_is_cancelled(self):
        return getattr(self, '_is_cancelled', False)


class MockStorageHandlerActor(StorageHandlerActor, CancelDetectActorMixin):
    async def fetch_batch(self, *args, **kwargs):
        async with self._delay_method():
            return super().fetch_batch(*args, **kwargs)


class MockQuotaActor(QuotaActor, CancelDetectActorMixin):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._batch_quota_reqs = []

    async def request_batch_quota(self, batch):
        self._batch_quota_reqs.append(batch)
        async with self._delay_method():
            return super().request_batch_quota(batch)

    def get_batch_quota_reqs(self):
        return self._batch_quota_reqs


class MockBandSlotManagerActor(BandSlotManagerActor, CancelDetectActorMixin):
    async def acquire_free_slot(self, session_stid: Tuple[str, str]):
        async with self._delay_method():
            return super().acquire_free_slot(session_stid)


class MockTaskManager(mo.Actor):
    def __init__(self):
        self._results = []

    def set_subtask_result(self, result):
        self._results.append(result)

    def get_results(self):
        return self._results


@pytest.fixture
async def actor_pool(request):
    n_slots, enable_kill = request.param
    pool = await mo.create_actor_pool('127.0.0.1',
                                      labels=[None] + ['numa-0'] * n_slots,
                                      n_process=n_slots)

    async with pool:
        session_id = 'test_session'
        await MockClusterAPI.create(pool.external_address, band_to_slots={'numa-0': n_slots})
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        await MockLifecycleAPI.create(session_id, pool.external_address)
        await MockSubtaskAPI.create(pool.external_address)
        storage_api = await MockStorageAPI.create(session_id, pool.external_address,
                                                  storage_handler_cls=MockStorageHandlerActor)

        # create assigner actor
        execution_ref = await mo.create_actor(SubtaskExecutionActor,
                                              subtask_max_retries=0,
                                              enable_kill_slot=enable_kill,
                                              uid=SubtaskExecutionActor.default_uid(),
                                              address=pool.external_address)
        # create quota actor
        await mo.create_actor(MockQuotaActor, 'numa-0', 102400,
                              uid=QuotaActor.gen_uid('numa-0'),
                              address=pool.external_address)
        # create dispatcher actor
        await mo.create_actor(MockBandSlotManagerActor,
                              (pool.external_address, 'numa-0'), n_slots,
                              uid=BandSlotManagerActor.gen_uid('numa-0'),
                              address=pool.external_address)
        # create mock task manager actor
        await mo.create_actor(MockTaskManager,
                              uid=TaskManagerActor.gen_uid(session_id),
                              address=pool.external_address)

        yield pool, session_id, meta_api, storage_api, execution_ref


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [(1, True)], indirect=True)
async def test_execute_tensor(actor_pool):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool

    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)

    input1 = TensorFetch(key='input1', source_key='input2', dtype=np.dtype(int)).new_chunk([])
    input2 = TensorFetch(key='input2', source_key='input2', dtype=np.dtype(int)).new_chunk([])
    result_chunk = TensorTreeAdd(args=[input1, input2]) \
        .new_chunk([input1, input2], shape=data1.shape, dtype=data1.dtype)

    await meta_api.set_chunk_meta(input1, memory_size=data1.nbytes, store_size=data1.nbytes,
                                  bands=[(pool.external_address, 'numa-0')])
    await meta_api.set_chunk_meta(input2, memory_size=data1.nbytes, store_size=data2.nbytes,
                                  bands=[(pool.external_address, 'numa-0')])
    # todo use different storage level when storage ready
    await storage_api.put(input1.key, data1)
    await storage_api.put(input2.key, data2)

    chunk_graph = ChunkGraph([result_chunk])
    chunk_graph.add_node(input1)
    chunk_graph.add_node(input2)
    chunk_graph.add_node(result_chunk)
    chunk_graph.add_edge(input1, result_chunk)
    chunk_graph.add_edge(input2, result_chunk)

    subtask = Subtask('test_task', session_id=session_id, chunk_graph=chunk_graph)
    await execution_ref.run_subtask(subtask, 'numa-0', pool.external_address)

    # check if results are correct
    result = await storage_api.get(result_chunk.key)
    np.testing.assert_array_equal(data1 + data2, result)

    # check if quota computations are correct
    quota_ref = await mo.actor_ref(QuotaActor.gen_uid('numa-0'),
                                   address=pool.external_address)
    [quota] = await quota_ref.get_batch_quota_reqs()
    assert quota[(subtask.subtask_id, subtask.subtask_id)] == data1.nbytes

    # check if metas are correct
    result_meta = await meta_api.get_chunk_meta(result_chunk.key)
    assert result_meta['object_id'] == result_chunk.key
    assert result_meta['shape'] == result.shape


_cancel_phases = [
    'prepare',
    'quota',
    'slot',
    'execute',
]


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool,cancel_phase',
                         [((1, True), phase) for phase in _cancel_phases],
                         indirect=['actor_pool'])
async def test_execute_with_cancel(actor_pool, cancel_phase):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool

    # config for different phases
    ref_to_delay = None
    if cancel_phase == 'prepare':
        ref_to_delay = await mo.actor_ref(
            StorageHandlerActor.default_uid(), address=pool.external_address)
    elif cancel_phase == 'quota':
        ref_to_delay = await mo.actor_ref(
            QuotaActor.gen_uid('numa-0'), address=pool.external_address)
    elif cancel_phase == 'slot':
        ref_to_delay = await mo.actor_ref(
            BandSlotManagerActor.gen_uid('numa-0'), address=pool.external_address)
    if ref_to_delay:
        await ref_to_delay.set_delay_fetch_time(100)

    def delay_fun(delay, _inp1):
        time.sleep(delay)
        return delay

    input1 = TensorFetch(key='input1', source_key='input1', dtype=np.dtype(int)).new_chunk([])
    remote_result = RemoteFunction(function=delay_fun, function_args=[100, input1],
                                   function_kwargs={}, n_output=1) \
        .new_chunk([input1])

    data1 = np.random.rand(10, 10)
    await meta_api.set_chunk_meta(input1, memory_size=data1.nbytes, store_size=data1.nbytes,
                                  bands=[(pool.external_address, 'numa-0')])
    await storage_api.put(input1.key, data1)

    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(input1)
    chunk_graph.add_node(remote_result)
    chunk_graph.add_edge(input1, remote_result)

    subtask = Subtask(f'test_task_{uuid.uuid4()}', session_id=session_id,
                      chunk_graph=chunk_graph)
    aiotask = asyncio.create_task(execution_ref.run_subtask(
        subtask, 'numa-0', pool.external_address))
    await asyncio.sleep(1)

    with Timer() as timer:
        await execution_ref.cancel_subtask(subtask.subtask_id, kill_timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(aiotask, timeout=30)
    assert timer.duration < 15

    # check for different phases
    if ref_to_delay is not None:
        assert await ref_to_delay.get_is_cancelled()
        await ref_to_delay.set_delay_fetch_time(0)

    # test if slot is restored
    remote_tileable = mr.spawn(delay_fun, args=(0.5, None))
    graph = TileableGraph([remote_tileable.data])
    next(TileableGraphBuilder(graph).build())

    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())

    subtask = Subtask(f'test_task2_{uuid.uuid4()}', session_id=session_id, chunk_graph=chunk_graph)
    await asyncio.wait_for(
        execution_ref.run_subtask(subtask, 'numa-0', pool.external_address),
        timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize('actor_pool', [(1, False)], indirect=True)
async def test_cancel_without_kill(actor_pool):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool

    def delay_fun(delay):
        import mars
        time.sleep(delay)
        mars._slot_marker = 1
        return delay

    def check_fun():
        import mars
        return getattr(mars, '_slot_marker', False)

    remote_result = RemoteFunction(function=delay_fun, function_args=[2],
                                   function_kwargs={}).new_chunk([])
    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(remote_result)

    subtask = Subtask(f'test_task_{uuid.uuid4()}', session_id=session_id,
                      chunk_graph=chunk_graph)
    aiotask = asyncio.create_task(execution_ref.run_subtask(
        subtask, 'numa-0', pool.external_address))
    await asyncio.sleep(0.5)

    await execution_ref.cancel_subtask(subtask.subtask_id, kill_timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(aiotask, timeout=30)

    remote_result = RemoteFunction(function=check_fun, function_args=[],
                                   function_kwargs={}).new_chunk([])
    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(remote_result)

    subtask = Subtask(f'test_task_{uuid.uuid4()}', session_id=session_id,
                      chunk_graph=chunk_graph)
    await execution_ref.run_subtask(
        subtask, 'numa-0', pool.external_address)

    # check if results are correct
    assert await storage_api.get(remote_result.key)
