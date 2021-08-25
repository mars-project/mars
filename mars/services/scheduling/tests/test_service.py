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
from collections import defaultdict

import numpy as np
import pytest

import mars.oscar as mo
import mars.remote as mr
import mars.tensor as mt
from mars.core.graph import TileableGraph, TileableGraphBuilder, ChunkGraphBuilder
from mars.services import start_services, stop_services, NodeRole
from mars.services.session import SessionAPI
from mars.services.scheduling import SchedulingAPI
from mars.services.scheduling.supervisor import GlobalSlotManagerActor
from mars.services.storage import StorageAPI, MockStorageAPI
from mars.services.subtask import Subtask, SubtaskResult, SubtaskStatus
from mars.services.task import new_task_id
from mars.services.task.supervisor.manager import TaskManagerActor


class FakeTaskManager(TaskManagerActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._events = defaultdict(list)
        self._results = dict()

    def set_subtask_result(self, subtask_result: SubtaskResult):
        self._results[subtask_result.subtask_id] = subtask_result
        for event in self._events[subtask_result.subtask_id]:
            event.set()
        self._events.pop(subtask_result.subtask_id, None)

    def _return_result(self, subtask_id: str):
        result = self._results[subtask_id]
        if result.status == SubtaskStatus.cancelled:
            raise asyncio.CancelledError
        elif result.status == SubtaskStatus.errored:
            raise result.error.with_traceback(result.traceback)
        return result

    async def wait_subtask_result(self, subtask_id: str):
        if subtask_id in self._results:
            return self._return_result(subtask_id)

        event = asyncio.Event()
        self._events[subtask_id].append(event)

        async def waiter():
            await event.wait()
            return self._return_result(subtask_id)

        return waiter()


def _gen_subtask(t, session_id):
    graph = TileableGraph([t.data])
    next(TileableGraphBuilder(graph).build())

    chunk_graph = next(ChunkGraphBuilder(graph, fuse_enabled=False).build())
    subtask = Subtask(new_task_id(), session_id,
                      new_task_id(), chunk_graph)

    return subtask


@pytest.fixture
async def actor_pools():
    async def start_pool(is_worker: bool):
        if is_worker:
            kw = dict(
                n_process=2,
                labels=['main'] + ['numa-0'] * 2,
                subprocess_start_method='spawn'
            )
        else:
            kw = dict(n_process=0,
                      subprocess_start_method='spawn')
        pool = await mo.create_actor_pool('127.0.0.1', **kw)
        await pool.start()
        return pool

    sv_pool, worker_pool = await asyncio.gather(
        start_pool(False), start_pool(True)
    )

    config = {
        "services": ["cluster", "session", "meta", "lifecycle",
                     "scheduling", "subtask", "task"],
        "cluster": {
            "backend": "fixed",
            "lookup_address": sv_pool.external_address,
            "resource": {"numa-0": 2}
        },
        "meta": {
            "store": "dict"
        },
        "scheduling": {},
        "subtask": {},
    }
    await start_services(
        NodeRole.SUPERVISOR, config, address=sv_pool.external_address)
    await start_services(
        NodeRole.WORKER, config, address=worker_pool.external_address)

    session_id = 'test_session'
    session_api = await SessionAPI.create(sv_pool.external_address)
    await session_api.create_session(session_id)
    ref = await mo.actor_ref(FakeTaskManager.gen_uid(session_id),
                             address=sv_pool.external_address)
    await mo.destroy_actor(ref)
    task_manager_ref = await mo.create_actor(
        FakeTaskManager, session_id,
        uid=FakeTaskManager.gen_uid(session_id),
        address=sv_pool.external_address)
    await MockStorageAPI.create(session_id, worker_pool.external_address)

    try:
        yield sv_pool, worker_pool, session_id, task_manager_ref
    finally:
        await session_api.delete_session(session_id)
        await MockStorageAPI.cleanup(worker_pool.external_address)
        await stop_services(
            NodeRole.WORKER, config, address=worker_pool.external_address)
        await stop_services(
            NodeRole.SUPERVISOR, config, address=sv_pool.external_address)

        await asyncio.gather(sv_pool.stop(), worker_pool.stop())


@pytest.mark.asyncio
async def test_schedule_success(actor_pools):
    sv_pool, worker_pool, session_id, task_manager_ref = actor_pools
    global_slot_ref = await mo.actor_ref(GlobalSlotManagerActor.default_uid(),
                                         address=sv_pool.external_address)

    scheduling_api = await SchedulingAPI.create(session_id, sv_pool.external_address)
    storage_api = await StorageAPI.create(session_id, worker_pool.external_address)

    a = mt.ones((10, 10), chunk_size=10)
    b = a + 1

    subtask = _gen_subtask(b, session_id)
    subtask.expect_bands = [(worker_pool.external_address, 'numa-0')]
    await scheduling_api.add_subtasks([subtask], [(0,)])
    await task_manager_ref.wait_subtask_result(subtask.subtask_id)

    result_key = next(subtask.chunk_graph.iter_indep(reverse=True)).key
    result = await storage_api.get(result_key)
    np.testing.assert_array_equal(np.ones((10, 10)) + 1, result)

    assert (await global_slot_ref.get_used_slots())['numa-0'] == 0


@pytest.mark.asyncio
async def test_schedule_queue(actor_pools):
    sv_pool, worker_pool, session_id, task_manager_ref = actor_pools
    global_slot_ref = await mo.actor_ref(GlobalSlotManagerActor.default_uid(),
                                         address=sv_pool.external_address)
    scheduling_api = await SchedulingAPI.create(session_id, sv_pool.external_address)

    finish_ids, finish_time = [], []

    def _remote_fun(secs):
        time.sleep(secs)
        return secs

    async def _waiter_fun(subtask_id):
        await task_manager_ref.wait_subtask_result(subtask_id)
        await scheduling_api.finish_subtasks([subtask_id])
        finish_ids.append(subtask_id)
        finish_time.append(time.time())

    subtasks = []
    wait_tasks = []
    for task_id in range(6):
        a = mr.spawn(_remote_fun, args=(0.5 + 0.01 * task_id,))
        subtask = _gen_subtask(a, session_id)
        subtask.subtask_id = f'test_schedule_queue_subtask_{task_id}'
        subtask.expect_bands = [(worker_pool.external_address, 'numa-0')]
        subtask.priority = (4 - task_id,)
        wait_tasks.append(asyncio.create_task(_waiter_fun(subtask.subtask_id)))
        subtasks.append(subtask)

    await scheduling_api.add_subtasks(subtasks)
    await scheduling_api.update_subtask_priority(subtasks[-1].subtask_id, (6,))
    await asyncio.gather(*wait_tasks)

    assert (await global_slot_ref.get_used_slots())['numa-0'] == 0


@pytest.mark.asyncio
async def test_schedule_error(actor_pools):
    sv_pool, worker_pool, session_id, task_manager_ref = actor_pools
    global_slot_ref = await mo.actor_ref(GlobalSlotManagerActor.default_uid(),
                                         address=sv_pool.external_address)
    scheduling_api = await SchedulingAPI.create(session_id, sv_pool.external_address)

    def _remote_fun():
        raise ValueError

    a = mr.spawn(_remote_fun)
    subtask = _gen_subtask(a, session_id)
    subtask.expect_bands = [(worker_pool.external_address, 'numa-0')]

    await scheduling_api.add_subtasks([subtask])
    with pytest.raises(ValueError):
        await task_manager_ref.wait_subtask_result(subtask.subtask_id)

    assert (await global_slot_ref.get_used_slots())['numa-0'] == 0


@pytest.mark.asyncio
async def test_schedule_cancel(actor_pools):
    sv_pool, worker_pool, session_id, task_manager_ref = actor_pools
    global_slot_ref = await mo.actor_ref(GlobalSlotManagerActor.default_uid(),
                                         address=sv_pool.external_address)
    scheduling_api = await SchedulingAPI.create(session_id, sv_pool.external_address)

    def _remote_fun(secs):
        time.sleep(secs)
        return secs

    async def _waiter_fun(subtask_id):
        await task_manager_ref.wait_subtask_result(subtask_id)
        await scheduling_api.finish_subtasks([subtask_id])

    subtasks = []
    wait_tasks = []
    for task_id in range(6):
        a = mr.spawn(_remote_fun, args=(1 - 0.01 * task_id,))
        subtask = _gen_subtask(a, session_id)
        subtask.subtask_id = f'test_schedule_queue_subtask_{task_id}'
        subtask.expect_bands = [(worker_pool.external_address, 'numa-0')]
        subtask.priority = (4 - task_id,)
        wait_tasks.append(asyncio.create_task(_waiter_fun(subtask.subtask_id)))
        subtasks.append(subtask)

    await scheduling_api.add_subtasks(subtasks)
    await asyncio.gather(*wait_tasks[:2])

    await scheduling_api.cancel_subtasks([subtask.subtask_id for subtask in subtasks],
                                         kill_timeout=0.1)

    for wait_task in wait_tasks[2:]:
        with pytest.raises(asyncio.CancelledError):
            await wait_task

    assert (await global_slot_ref.get_used_slots())['numa-0'] == 0
