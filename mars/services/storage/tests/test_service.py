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

import os
import sys

import numpy as np
import pandas as pd
import pytest

import mars.oscar as mo
from mars.services import start_services, NodeRole
from mars.services.storage import StorageAPI


@pytest.fixture
async def actor_pools():
    async def start_pool():
        start_method = os.environ.get('POOL_START_METHOD', 'forkserver') \
            if sys.platform != 'win32' else None
        pool = await mo.create_actor_pool('127.0.0.1', n_process=1,
                                          subprocess_start_method=start_method)
        await pool.start()
        return pool

    worker_pool = await start_pool()
    yield worker_pool
    await worker_pool.stop()


@pytest.mark.asyncio
async def test_storage_service(actor_pools):
    worker_pool = actor_pools

    if sys.platform == 'darwin':
        plasma_dir = '/tmp'
    else:
        plasma_dir = '/dev/shm'
    plasma_setup_params = dict(
        store_memory=10 * 1024 * 1024,
        plasma_directory=plasma_dir,
        check_dir_size=False)

    config = {
        "services": ["storage"],
        "storage": {
            "backends": ["plasma"],
            "plasma": plasma_setup_params,
        }
    }

    await start_services(
        NodeRole.WORKER, config, address=worker_pool.external_address)

    api = await StorageAPI.create('mock_session', worker_pool.external_address)
    value1 = np.random.rand(10, 10)
    await api.put('data1', value1)
    get_value1 = await api.get('data1')
    np.testing.assert_array_equal(value1, get_value1)

    # test api in subpool
    subpool_address = list(worker_pool._sub_processes.keys())[0]
    api2 = await StorageAPI.create('mock_session', subpool_address)
    assert api2._storage_handler_ref.address == subpool_address

    get_value1 = await api2.get('data1')
    np.testing.assert_array_equal(value1, get_value1)

    sliced_value = await api2.get('data1', conditions=[slice(None, None), slice(0, 4)])
    np.testing.assert_array_equal(value1[:, :4], sliced_value)

    value2 = pd.DataFrame(value1)
    await api2.put('data2', value2)

    get_value2 = await api.get('data2')
    pd.testing.assert_frame_equal(value2, get_value2)
