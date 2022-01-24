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
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import pytest

from ...... import oscar as mo
from ...... import remote as mr
from ......core import (
    ChunkGraph,
    TileableGraph,
    TileableGraphBuilder,
    ChunkGraphBuilder,
)
from ......core.base import MarsError
from ......remote.core import RemoteFunction
from ......tensor.fetch import TensorFetch
from ......tensor.arithmetic import TensorTreeAdd
from ......utils import Timer, build_fetch_chunk
from .....cluster import MockClusterAPI
from .....lifecycle import MockLifecycleAPI
from .....meta import MockMetaAPI
from .....mutable import MockMutableAPI
from .....session import MockSessionAPI
from .....storage import MockStorageAPI
from .....subtask import MockSubtaskAPI, Subtask, SubtaskStatus
from .....task.supervisor.manager import TaskManagerActor
from ...quota import QuotaActor
from ...queues import SubtaskPrepareQueueActor, SubtaskExecutionQueueActor
from ...slotmanager import SlotManagerActor
from ..actor import SubtaskExecutionActor


class CancelDetectActorMixin:
    @asynccontextmanager
    async def _delay_method(self):
        delay_fetch_event = getattr(self, "_delay_fetch_event", None)
        delay_wait_event = getattr(self, "_delay_wait_event", None)
        try:
            if delay_fetch_event is not None:
                delay_fetch_event.set()
            if delay_wait_event is not None:
                await delay_wait_event.wait()
            yield
        except asyncio.CancelledError:
            self._is_cancelled = True
            raise

    def set_delay_fetch_event(
        self, fetch_event: asyncio.Event, wait_event: asyncio.Event
    ):
        setattr(self, "_delay_fetch_event", fetch_event)
        setattr(self, "_delay_wait_event", wait_event)

    def get_is_cancelled(self):
        return getattr(self, "_is_cancelled", False)


class MockQuotaActor(QuotaActor, CancelDetectActorMixin):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._batch_quota_reqs = {}

    async def request_batch_quota(self, batch, insufficient: str = "enqueue"):
        async with self._delay_method():
            res = super().request_batch_quota(batch, insufficient=insufficient)
        self._batch_quota_reqs.update(batch)
        return res

    def get_batch_quota_reqs(self, quota_key):
        return self._batch_quota_reqs[quota_key]


class MockSubtaskPrepareQueueActor(SubtaskPrepareQueueActor, CancelDetectActorMixin):
    async def get(self, band_name: str):
        val = await super().get(band_name)
        async with self._delay_method():
            return val


class MockSubtaskExecutionQueueActor(
    SubtaskExecutionQueueActor, CancelDetectActorMixin
):
    async def get(self, band_name: str):
        val = await super().get(band_name)
        async with self._delay_method():
            return val


class MockTaskManager(mo.Actor):
    def __init__(self):
        self._results = []

    def set_subtask_result(self, result):
        self._results.append(result)

    def get_results(self):
        return self._results


@pytest.fixture
async def actor_pool(request):
    n_slots, enable_kill, max_retries = request.param
    pool = await mo.create_actor_pool(
        "127.0.0.1", labels=[None] + ["numa-0"] * n_slots, n_process=n_slots
    )

    async with pool:
        session_id = "test_session"
        await MockClusterAPI.create(
            pool.external_address, band_to_slots={"numa-0": n_slots}
        )
        await MockSessionAPI.create(pool.external_address, session_id=session_id)
        meta_api = await MockMetaAPI.create(session_id, pool.external_address)
        await MockLifecycleAPI.create(session_id, pool.external_address)
        await MockSubtaskAPI.create(pool.external_address)
        await MockMutableAPI.create(session_id, pool.external_address)
        storage_api = await MockStorageAPI.create(
            session_id,
            pool.external_address,
        )

        slot_manager_ref = await mo.create_actor(
            SlotManagerActor,
            uid=SlotManagerActor.default_uid(),
            address=pool.external_address,
        )

        prepare_queue_ref = await mo.create_actor(
            MockSubtaskPrepareQueueActor,
            uid=SubtaskPrepareQueueActor.default_uid(),
            address=pool.external_address,
        )
        exec_queue_ref = await mo.create_actor(
            MockSubtaskExecutionQueueActor,
            uid=SubtaskExecutionQueueActor.default_uid(),
            address=pool.external_address,
        )
        # create quota actor
        quota_ref = await mo.create_actor(
            MockQuotaActor,
            "numa-0",
            102400,
            uid=QuotaActor.gen_uid("numa-0"),
            address=pool.external_address,
        )

        # create mock task manager actor
        task_manager_ref = await mo.create_actor(
            MockTaskManager,
            uid=TaskManagerActor.gen_uid(session_id),
            address=pool.external_address,
        )

        # create assigner actor
        execution_ref = await mo.create_actor(
            SubtaskExecutionActor,
            subtask_max_retries=max_retries,
            enable_kill_slot=enable_kill,
            uid=SubtaskExecutionActor.default_uid(),
            address=pool.external_address,
        )

        try:
            yield pool, session_id, meta_api, storage_api, execution_ref
        finally:
            await mo.destroy_actor(execution_ref)
            await mo.destroy_actor(prepare_queue_ref)
            await mo.destroy_actor(exec_queue_ref)
            await mo.destroy_actor(quota_ref)
            await mo.destroy_actor(task_manager_ref)
            await mo.destroy_actor(slot_manager_ref)
            await MockStorageAPI.cleanup(pool.external_address)
            await MockSubtaskAPI.cleanup(pool.external_address)
            await MockClusterAPI.cleanup(pool.external_address)
            await MockMutableAPI.cleanup(session_id, pool.external_address)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [(1, True, 0)], indirect=True)
async def test_execute_tensor(actor_pool):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool

    async def build_test_subtask(subtask_name):
        data1 = np.random.rand(10, 10)
        data2 = np.random.rand(10, 10)

        input1_key = f"{subtask_name}_input1"
        input1 = TensorFetch(
            key=input1_key, source_key=input1_key, dtype=np.dtype(int)
        ).new_chunk([])

        input2_key = f"{subtask_name}_input2"
        input2 = TensorFetch(
            key=input2_key, source_key=input2_key, dtype=np.dtype(int)
        ).new_chunk([])

        result_chunk = TensorTreeAdd(args=[input1, input2]).new_chunk(
            [input1, input2], shape=data1.shape, dtype=data1.dtype
        )

        await meta_api.set_chunk_meta(
            input1,
            memory_size=data1.nbytes,
            store_size=data1.nbytes,
            bands=[(pool.external_address, "numa-0")],
        )
        await meta_api.set_chunk_meta(
            input2,
            memory_size=data1.nbytes,
            store_size=data2.nbytes,
            bands=[(pool.external_address, "numa-0")],
        )
        # todo use different storage level when storage ready
        await storage_api.put(input1.key, data1)
        await storage_api.put(input2.key, data2)

        chunk_graph = ChunkGraph([result_chunk])
        chunk_graph.add_node(input1)
        chunk_graph.add_node(input2)
        chunk_graph.add_node(result_chunk)
        chunk_graph.add_edge(input1, result_chunk)
        chunk_graph.add_edge(input2, result_chunk)

        return (
            Subtask(subtask_name, session_id=session_id, chunk_graph=chunk_graph),
            (data1, data2),
        )

    subtask1, data_group1 = await build_test_subtask("subtask1")
    subtask2, data_group2 = await build_test_subtask("subtask2")

    await execution_ref.submit_subtasks(
        [subtask1, subtask2], [(0,), (0,)], pool.external_address, "numa-0"
    )
    await execution_ref.wait_subtasks([subtask1.subtask_id, subtask2.subtask_id])

    for subtask, (data1, data2) in [(subtask1, data_group1), (subtask2, data_group2)]:
        result_chunk = subtask.chunk_graph.result_chunks[0]
        # check if results are correct
        result = await storage_api.get(result_chunk.key)
        np.testing.assert_array_equal(data1 + data2, result)

        # check if quota computations are correct
        quota_ref = await mo.actor_ref(
            QuotaActor.gen_uid("numa-0"), address=pool.external_address
        )
        quota = await quota_ref.get_batch_quota_reqs(
            (subtask.session_id, subtask.subtask_id)
        )
        assert quota == data1.nbytes

        # check if metas are correct
        result_meta = await meta_api.get_chunk_meta(result_chunk.key)
        assert result_meta["object_id"] == result_chunk.key
        assert result_meta["shape"] == result.shape


_cancel_phases = [
    "immediately",
    "prepare-queue",
    "prepare",
    "execute-queue",
    "execute",
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actor_pool,cancel_phase",
    [((1, True, 0), phase) for phase in _cancel_phases],
    indirect=["actor_pool"],
)
async def test_execute_with_cancel(actor_pool, cancel_phase):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool
    delay_fetch_event = asyncio.Event()
    delay_wait_event = asyncio.Event()

    # config for different phases
    ref_to_delay = None
    if cancel_phase == "prepare-queue":
        ref_to_delay = await mo.actor_ref(
            SubtaskPrepareQueueActor.default_uid(),
            address=pool.external_address,
        )
    elif cancel_phase == "prepare":
        ref_to_delay = await mo.actor_ref(
            QuotaActor.gen_uid("numa-0"), address=pool.external_address
        )
    elif cancel_phase == "execute-queue":
        ref_to_delay = await mo.actor_ref(
            SubtaskExecutionQueueActor.default_uid(),
            address=pool.external_address,
        )
    if ref_to_delay is not None:
        await ref_to_delay.set_delay_fetch_event(delay_fetch_event, delay_wait_event)
    else:
        delay_fetch_event.set()

    def delay_fun(delay, _inp1):
        if not ref_to_delay:
            time.sleep(delay)
        return (delay,)

    input1 = TensorFetch(
        key="input1", source_key="input1", dtype=np.dtype(int)
    ).new_chunk([])
    remote_result = RemoteFunction(
        function=delay_fun, function_args=[100, input1], function_kwargs={}, n_output=1
    ).new_chunk([input1])

    data1 = np.random.rand(10, 10)
    await meta_api.set_chunk_meta(
        input1,
        memory_size=data1.nbytes,
        store_size=data1.nbytes,
        bands=[(pool.external_address, "numa-0")],
    )
    await storage_api.put(input1.key, data1)

    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(input1)
    chunk_graph.add_node(remote_result)
    chunk_graph.add_edge(input1, remote_result)

    subtask = Subtask(
        f"test_subtask_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph
    )
    await execution_ref.submit_subtasks(
        [subtask], [(0,)], pool.external_address, "numa-0"
    )
    aiotask = asyncio.create_task(execution_ref.wait_subtasks([subtask.subtask_id]))
    if ref_to_delay is not None:
        await delay_fetch_event.wait()
    else:
        if cancel_phase != "immediately":
            await asyncio.sleep(1)

    with Timer() as timer:
        await asyncio.wait_for(
            execution_ref.cancel_subtasks([subtask.subtask_id], kill_timeout=1),
            timeout=30,
        )
        [r] = await asyncio.wait_for(aiotask, timeout=30)
        assert r.status == SubtaskStatus.cancelled
    assert timer.duration < 15

    # check for different phases
    if ref_to_delay is not None:
        if not cancel_phase.endswith("queue"):
            assert await ref_to_delay.get_is_cancelled()
        delay_wait_event.set()

    # test if slot is restored
    remote_tileable = mr.spawn(delay_fun, args=(0.5, None))
    graph = TileableGraph([remote_tileable.data])
    next(TileableGraphBuilder(graph).build())

    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())

    subtask = Subtask(
        f"test_subtask2_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph
    )
    await execution_ref.submit_subtasks(
        [subtask], [(0,)], pool.external_address, "numa-0"
    )
    await asyncio.wait_for(execution_ref.wait_subtasks([subtask]), timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [(1, False, 0)], indirect=True)
async def test_cancel_without_kill(actor_pool):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool
    executed_file = os.path.join(
        tempfile.gettempdir(), f"mars_test_cancel_without_kill_{os.getpid()}.tmp"
    )

    def delay_fun(delay):
        import mars

        open(executed_file, "w").close()
        time.sleep(delay)
        mars._slot_marker = 1
        return delay

    def check_fun():
        import mars

        return getattr(mars, "_slot_marker", False)

    remote_result = RemoteFunction(
        function=delay_fun, function_args=[2], function_kwargs={}
    ).new_chunk([])
    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(remote_result)

    subtask = Subtask(
        f"test_subtask_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph
    )
    await execution_ref.submit_subtasks(
        [subtask], [(0,)], pool.external_address, "numa-0"
    )
    aiotask = asyncio.create_task(execution_ref.wait_subtasks([subtask.subtask_id]))
    await asyncio.sleep(0.5)

    await asyncio.wait_for(
        execution_ref.cancel_subtasks([subtask.subtask_id], kill_timeout=1),
        timeout=30,
    )
    [r] = await asyncio.wait_for(aiotask, timeout=30)
    assert r.status == SubtaskStatus.cancelled

    remote_result = RemoteFunction(
        function=check_fun, function_args=[], function_kwargs={}
    ).new_chunk([])
    chunk_graph = ChunkGraph([remote_result])
    chunk_graph.add_node(remote_result)

    subtask = Subtask(
        f"test_subtask_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph
    )
    await execution_ref.submit_subtasks(
        [subtask], [(0,)], pool.external_address, "numa-0"
    )
    await execution_ref.wait_subtasks([subtask.subtask_id])

    # check if slots not killed (or slot assignment may be cancelled)
    if os.path.exists(executed_file):
        assert await storage_api.get(remote_result.key)
        os.unlink(executed_file)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [(1, False, 2)], indirect=True)
async def test_retry_execution(actor_pool):
    with tempfile.TemporaryDirectory(prefix="mars_test_retry_exec_") as tempdir:
        pool, session_id, meta_api, storage_api, execution_ref = actor_pool

        def error_fun(count_file, max_fail):
            file_path = os.path.join(tempdir, count_file)
            try:
                cnt = int(open(file_path, "r").read())
            except OSError:
                cnt = 0

            if cnt < max_fail:
                with open(file_path, "w") as f:
                    f.write(str(cnt + 1))
                raise MarsError(f"Error No {cnt}")

        remote_result = RemoteFunction(
            function=error_fun, function_args=["subtask1", 1], function_kwargs={}
        ).new_chunk([])
        chunk_graph = ChunkGraph([remote_result])
        chunk_graph.add_node(remote_result)

        subtask = Subtask(
            f"test_subtask_{uuid.uuid4()}",
            session_id=session_id,
            chunk_graph=chunk_graph,
        )
        await execution_ref.submit_subtasks(
            [subtask], [(0,)], pool.external_address, "numa-0"
        )
        [r] = await execution_ref.wait_subtasks([subtask.subtask_id])
        assert r.status == SubtaskStatus.succeeded
        assert int(open(os.path.join(tempdir, "subtask1"), "r").read()) == 1

        remote_result = RemoteFunction(
            function=error_fun, function_args=["subtask2", 2], function_kwargs={}
        ).new_chunk([])
        chunk_graph = ChunkGraph([remote_result])
        chunk_graph.add_node(remote_result)

        subtask = Subtask(
            f"test_subtask_{uuid.uuid4()}",
            session_id=session_id,
            chunk_graph=chunk_graph,
        )
        await execution_ref.submit_subtasks(
            [subtask], [(0,)], pool.external_address, "numa-0"
        )
        [r] = await execution_ref.wait_subtasks([subtask.subtask_id])
        assert r.status == SubtaskStatus.errored
        assert int(open(os.path.join(tempdir, "subtask2"), "r").read()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_pool", [(1, False, 0)], indirect=True)
async def test_cached_successors(actor_pool):
    pool, session_id, meta_api, storage_api, execution_ref = actor_pool

    data1 = np.random.rand(5, 5)
    data2 = np.random.rand(5, 5)

    def func1():
        return (data1,)

    def func2(v):
        return (v + data2,)

    result1 = RemoteFunction(function=func1, n_output=1).new_chunk([])
    chunk_graph1 = ChunkGraph([result1])
    chunk_graph1.add_node(result1)
    subtask1 = Subtask(
        f"test_subtask_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph1
    )

    fetch1 = build_fetch_chunk(result1)
    result2 = RemoteFunction(
        function=func2, function_args=[fetch1], n_output=1
    ).new_chunk([fetch1])
    chunk_graph2 = ChunkGraph([result2])
    chunk_graph2.add_node(fetch1)
    chunk_graph2.add_node(result2)
    chunk_graph2.add_edge(fetch1, result2)
    subtask2 = Subtask(
        f"test_subtask_{uuid.uuid4()}", session_id=session_id, chunk_graph=chunk_graph2
    )

    await execution_ref.cache_subtasks(
        [subtask2], [(0,)], pool.external_address, "numa-0"
    )
    await execution_ref.submit_subtasks(
        [subtask1], [(0,)], pool.external_address, "numa-0"
    )
    [r] = await execution_ref.wait_subtasks([subtask1.subtask_id])
    [r] = await execution_ref.wait_subtasks([subtask2.subtask_id])
