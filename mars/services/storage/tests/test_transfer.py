# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import sys

import numpy as np
import pandas as pd
import pytest

import mars.oscar as mo
from mars.oscar.backends.allocate_strategy import IdleLabel
from mars.services.storage.errors import DataNotExist
from mars.services.storage.core import StorageManagerActor, StorageHandlerActor
from mars.services.storage.transfer import SenderManagerActor
from mars.storage import StorageLevel


@pytest.fixture
async def actor_pools():
    async def start_pool():
        start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
            if sys.platform != 'win32' else None

        pool = await mo.create_actor_pool('127.0.0.1', n_process=2,
                                          labels=['main', 'sub', 'io'],
                                          subprocess_start_method=start_method)
        await pool.start()
        return pool

    worker_pool_1 = await start_pool()
    worker_pool_2 = await start_pool()
    yield worker_pool_1, worker_pool_2
    await worker_pool_1.stop()
    await worker_pool_2.stop()


@pytest.fixture
async def create_actors(actor_pools):
    worker_pool_1, worker_pool_2 = actor_pools

    if sys.platform == 'darwin':
        plasma_dir = '/tmp'
    else:
        plasma_dir = '/dev/shm'
    plasma_setup_params = dict(
        store_memory=5 * 1024 * 1024,
        plasma_directory=plasma_dir,
        check_dir_size=False)
    storage_configs = {
        "plasma": plasma_setup_params,
    }

    manager_ref1 = await mo.create_actor(
        StorageManagerActor, storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_1.external_address)

    manager_ref2 = await mo.create_actor(
        StorageManagerActor, storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=worker_pool_2.external_address)
    yield worker_pool_1.external_address, worker_pool_2.external_address
    await mo.destroy_actor(manager_ref1)
    await mo.destroy_actor(manager_ref2)


@pytest.mark.asyncio
async def test_simple_transfer(create_actors):
    worker_address_1, worker_address_2 = create_actors

    session_id = 'mock_session'
    data1 = np.random.rand(100, 100)
    data2 = pd.DataFrame(np.random.randint(0, 100, (500, 10)))

    storage_handler1 = await mo.actor_ref(uid=StorageHandlerActor.default_uid(),
                                          address=worker_address_1)
    storage_handler2 = await mo.actor_ref(uid=StorageHandlerActor.default_uid(),
                                          address=worker_address_2)

    await storage_handler1.put(session_id, 'data_key1', data1, StorageLevel.MEMORY)
    await storage_handler1.put(session_id, 'data_key2', data2, StorageLevel.MEMORY)
    await storage_handler2.put(session_id, 'data_key3', data2, StorageLevel.MEMORY)

    sender_actor = await mo.actor_ref(address=worker_address_1,
                                      uid=SenderManagerActor.default_uid())

    # send data to worker2 from worker1
    await sender_actor.send_data(session_id, 'data_key1',
                                 worker_address_2,
                                 StorageLevel.MEMORY, block_size=1000)

    await sender_actor.send_data(session_id, 'data_key2',
                                 worker_address_2,
                                 StorageLevel.MEMORY, block_size=1000)

    get_data1 = await storage_handler2.get(session_id, 'data_key1')
    np.testing.assert_array_equal(data1, get_data1)

    get_data2 = await storage_handler2.get(session_id, 'data_key2')
    pd.testing.assert_frame_equal(data2, get_data2)

    # send data to worker1 from worker2
    sender_actor = await mo.actor_ref(address=worker_address_2,
                                      uid=SenderManagerActor.default_uid())
    await sender_actor.send_data(session_id, 'data_key3', worker_address_1,
                                 StorageLevel.MEMORY)
    get_data3 = await storage_handler1.get(session_id, 'data_key3')
    pd.testing.assert_frame_equal(data2, get_data3)


class MockSenderManagerActor(SenderManagerActor):
    @staticmethod
    async def send_part(receiver_ref, message):
        send_task = None
        try:
            await asyncio.sleep(3)
            send_task = asyncio.create_task(
                receiver_ref.receive_part_data(message))
            await send_task
        except asyncio.CancelledError:  # pragma: no cover
            if send_task:
                send_task.cancel()
                await send_task


@pytest.mark.asyncio
async def test_cancel_transfer(create_actors):
    worker_address_1, worker_address_2 = create_actors

    strategy = IdleLabel('io', 'mock_sender')
    await mo.create_actor(
        MockSenderManagerActor, uid=MockSenderManagerActor.default_uid(),
        address=worker_address_1, allocate_strategy=strategy)

    data1 = np.random.rand(10, 10)
    storage_handler1 = await mo.actor_ref(
        uid=StorageHandlerActor.default_uid(),
        address=worker_address_1)
    storage_handler2 = await mo.actor_ref(
        uid=StorageHandlerActor.default_uid(),
        address=worker_address_2)
    await storage_handler1.put('mock', 'data_key1',
                               data1, StorageLevel.MEMORY)

    sender_actor = await mo.actor_ref(address=worker_address_1,
                                      uid=MockSenderManagerActor.default_uid())
    send_task = asyncio.create_task(sender_actor.send_data(
        'mock', 'data_key1', worker_address_2, StorageLevel.MEMORY))

    await asyncio.sleep(0.2)
    send_task.cancel()
    await send_task

    with pytest.raises(DataNotExist):
        await storage_handler2.get('mock', 'data_key1')
