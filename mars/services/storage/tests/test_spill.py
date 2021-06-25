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

import os
import sys
import tempfile

import numpy as np
import pytest

import mars.oscar as mo
from mars.services.storage.core import StorageManagerActor, StorageHandlerActor, \
    StorageQuotaActor
from mars.storage import StorageLevel, PlasmaStorage


MEMORY_SIZE = 100 * 1024


@pytest.fixture
async def actor_pool():
    async def start_pool():
        start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
            if sys.platform != 'win32' else None

        pool = await mo.create_actor_pool('127.0.0.1', n_process=2,
                                          labels=['main', 'sub', 'io'],
                                          subprocess_start_method=start_method)
        await pool.start()
        return pool

    worker_pool = await start_pool()
    yield worker_pool
    await worker_pool.stop()


@pytest.fixture
async def create_actors(actor_pool):
    if sys.platform == 'darwin':
        plasma_dir = '/tmp'
    else:
        plasma_dir = '/dev/shm'
    plasma_setup_params = dict(
        store_memory=MEMORY_SIZE,
        plasma_directory=plasma_dir,
        check_dir_size=False
    )
    tempdir = tempfile.mkdtemp()
    disk_setup_params = dict(
        root_dirs=tempdir,
        level='disk'
    )
    storage_configs = {
        "plasma": plasma_setup_params,
        "filesystem": disk_setup_params
    }

    manager_ref = await mo.create_actor(
        StorageManagerActor, storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=actor_pool.external_address)

    yield actor_pool.external_address
    await mo.destroy_actor(manager_ref)


@pytest.mark.asyncio
async def test_spill(create_actors):
    worker_address = create_actors
    storage_handler = await mo.actor_ref(uid=StorageHandlerActor.default_uid(),
                                         address=worker_address)

    storage_manager = await mo.actor_ref(uid=StorageManagerActor.default_uid(),
                                         address=worker_address)

    init_params = await storage_manager.get_client_params()
    plasma_init_params = init_params['plasma']
    plasma_handler = PlasmaStorage(**plasma_init_params)
    memory_quota = await mo.actor_ref(
        StorageQuotaActor, StorageLevel.MEMORY, MEMORY_SIZE,
        address=worker_address, uid=StorageQuotaActor.gen_uid(StorageLevel.MEMORY))

    # fill to trigger spill
    session_id = 'mock_session'
    data_list = []
    key_list = []
    for i in range(10):
        data = np.random.randint(0, 10000, (8000,), np.int16)
        key = f'mock_key_{i}'
        await storage_handler.put(
            session_id, key, data, StorageLevel.MEMORY)
        used = (await memory_quota.get_quota())[1]
        assert used < MEMORY_SIZE
        data_list.append(data)
        key_list.append(key)

    memory_object_list = await storage_handler.list(StorageLevel.MEMORY)
    disk_object_list = await storage_handler.list(StorageLevel.DISK)
    assert len(memory_object_list) == 2
    assert len(disk_object_list) == 8

    for key, data in zip(key_list, data_list):
        get_data = await storage_handler.get(session_id, key)
        np.testing.assert_array_equal(data, get_data)

    plasma_list = await plasma_handler.list()
    assert len(plasma_list) == len(memory_object_list)
