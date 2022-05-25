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
import sys
import tempfile

import numpy as np
import pytest

from .... import oscar as mo
from ....storage import StorageLevel, PlasmaStorage
from ....utils import calc_data_size
from ..core import StorageManagerActor, StorageQuotaActor, build_data_info
from ..handler import StorageHandlerActor

# todo enable this test module when spill support added
#  on storage quotas
if sys.platform.lower().startswith("win"):
    pytestmark = pytest.mark.skip

MEMORY_SIZE = 100 * 1024


@pytest.fixture
async def actor_pool():
    async def start_pool():
        start_method = (
            os.environ.get("POOL_START_METHOD", "forkserver")
            if sys.platform != "win32"
            else None
        )

        pool = await mo.create_actor_pool(
            "127.0.0.1",
            n_process=2,
            labels=["main", "numa-0", "io"],
            subprocess_start_method=start_method,
        )
        await pool.start()
        return pool

    worker_pool = await start_pool()
    try:
        yield worker_pool
    finally:
        await worker_pool.stop()


def _build_storage_config():
    if sys.platform == "darwin":
        plasma_dir = "/tmp"
    else:
        plasma_dir = "/dev/shm"
    plasma_setup_params = dict(
        store_memory=MEMORY_SIZE, plasma_directory=plasma_dir, check_dir_size=False
    )
    tempdir = tempfile.mkdtemp()
    disk_setup_params = dict(root_dirs=tempdir, level="disk")
    storage_configs = {"plasma": plasma_setup_params, "filesystem": disk_setup_params}
    return storage_configs


@pytest.fixture
async def create_actors(actor_pool):
    storage_configs = _build_storage_config()
    manager_ref = await mo.create_actor(
        StorageManagerActor,
        storage_configs,
        uid=StorageManagerActor.default_uid(),
        address=actor_pool.external_address,
    )

    sub_processes = list(actor_pool.sub_processes)
    yield actor_pool.external_address, sub_processes[0], sub_processes[1]
    await mo.destroy_actor(manager_ref)


@pytest.mark.asyncio
async def test_spill(create_actors):
    worker_address, _, _ = create_actors
    storage_handler = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=worker_address
    )

    storage_manager = await mo.actor_ref(
        uid=StorageManagerActor.default_uid(), address=worker_address
    )

    init_params = (await storage_manager.get_client_params())["numa-0"]
    plasma_init_params = init_params["plasma"]
    plasma_handler = PlasmaStorage(**plasma_init_params)
    memory_quota = await mo.actor_ref(
        StorageQuotaActor,
        StorageLevel.MEMORY,
        MEMORY_SIZE,
        address=worker_address,
        uid=StorageQuotaActor.gen_uid("numa-0", StorageLevel.MEMORY),
    )

    # fill to trigger spill
    session_id = "mock_session"
    data_list = []
    key_list = []
    for i in range(10):
        data = np.random.randint(0, 10000, (8000,), np.int16)
        key = f"mock_key_{i}"
        await storage_handler.put(session_id, key, data, StorageLevel.MEMORY)
        used = (await memory_quota.get_quota())[1]
        assert used < MEMORY_SIZE
        data_list.append(data)
        key_list.append(key)

    memory_object_list = await storage_handler.list(StorageLevel.MEMORY)
    disk_object_list = await storage_handler.list(StorageLevel.DISK)
    assert len(memory_object_list) == 3
    assert len(disk_object_list) == 7

    for key, data in zip(key_list, data_list):
        get_data = await storage_handler.get(session_id, key)
        np.testing.assert_array_equal(data, get_data)

    plasma_list = await plasma_handler.list()
    assert len(plasma_list) == len(memory_object_list)


class DelayPutStorageHandler(StorageHandlerActor):
    async def put(
        self, session_id: str, data_key: str, obj: object, level: StorageLevel
    ):
        size = calc_data_size(obj)
        await self.request_quota_with_spill(level, size)
        # sleep to trigger `NoDataToSpill`
        await asyncio.sleep(0.5)
        object_info = await self._clients[level].put(obj)
        data_info = build_data_info(object_info, level, size)
        await self._data_manager_ref.put_data_info(
            session_id, data_key, data_info, object_info
        )
        if object_info.size is not None and data_info.memory_size != object_info.size:
            await self._quota_refs[level].update_quota(
                object_info.size - data_info.memory_size
            )
        await self.notify_spillable_space(level)
        return data_info


@pytest.fixture
async def create_actors_with_delay(actor_pool):
    storage_configs = _build_storage_config()
    manager_ref = await mo.create_actor(
        StorageManagerActor,
        storage_configs,
        storage_handler_cls=DelayPutStorageHandler,
        uid=StorageManagerActor.default_uid(),
        address=actor_pool.external_address,
    )

    sub_processes = list(actor_pool.sub_processes)
    yield actor_pool.external_address, sub_processes[0], sub_processes[1]
    await mo.destroy_actor(manager_ref)


@pytest.mark.asyncio
async def test_spill_event(create_actors_with_delay):
    worker_address, sub_pool_address1, sub_pool_address2 = create_actors_with_delay
    storage_handler1 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=sub_pool_address1
    )
    storage_handler2 = await mo.actor_ref(
        uid=StorageHandlerActor.gen_uid("numa-0"), address=sub_pool_address2
    )
    # total store size is 65536, single data size is around 40000
    # we put two data simultaneously
    data = np.random.randint(0, 10000, (5000,))
    session_id = "mock_session"
    key1 = "mock_key1"
    key2 = "mock_key2"
    put1 = asyncio.create_task(
        storage_handler1.put(session_id, key1, data, StorageLevel.MEMORY)
    )
    put2 = asyncio.create_task(
        storage_handler2.put(session_id, key2, data, StorageLevel.MEMORY)
    )
    await asyncio.gather(put1, put2)

    get_data = await storage_handler2.get(session_id, key1)
    np.testing.assert_array_equal(data, get_data)
    get_data = await storage_handler1.get(session_id, key2)
    np.testing.assert_array_equal(data, get_data)

    memory_object_list = await storage_handler1.list(StorageLevel.MEMORY)
    disk_object_list = await storage_handler1.list(StorageLevel.DISK)
    assert len(memory_object_list) == 1
    assert len(disk_object_list) == 1
