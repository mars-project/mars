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

import sys

import numpy as np
import pandas as pd
import pytest

import mars.oscar as mo
from mars.serialize import dataserializer
from mars.services.storage.api import MockStorageApi
from mars.storage import StorageLevel
from mars.tests.core import require_ray

try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import ray
except ImportError:
    ray = None

require_lib = lambda x: x
storage_configs = []

# plasma backend
plasma_storage_size = 10 * 1024 * 1024
if sys.platform == 'darwin':
    plasma_dir = '/tmp'
else:
    plasma_dir = '/dev/shm'
plasma_setup_params = dict(
    store_memory=plasma_storage_size,
    plasma_directory=plasma_dir,
    check_dir_size=False)
storage_configs.append({'plasma': plasma_setup_params})

# ray backend
if ray is not None:
    require_lib = require_ray
    storage_configs.append({'ray': dict()})

# vineyard
if vineyard is not None:
    storage_configs.append({'vineyard': dict(
        vineyard_size='256M',
        vineyard_socket='/tmp/vineyard.sock'
    )})

# shared_memory
if sys.version_info[:2] >= (3, 8):
    storage_configs.append({'shared_memory': dict()})


@pytest.mark.asyncio
@pytest.mark.parametrize('storage_configs', storage_configs)
@require_lib
async def test_storage_mock_api(storage_configs):
    start_method = 'fork' if sys.platform != 'win32' else None
    pool = await mo.create_actor_pool('127.0.0.1', 1,
                                      subprocess_start_method=start_method)
    async with pool:
        session_id = 'mock_session_id'
        storage_api = await MockStorageApi.create(
            address=pool.external_address,
            session_id=session_id,
            storage_configs=storage_configs)

        # test put and get
        value1 = np.random.rand(10, 10)
        await storage_api.put('data1', value1)
        get_value1 = await storage_api.get('data1')
        np.testing.assert_array_equal(value1, get_value1)

        value2 = pd.DataFrame({'col1': [str(i) for i in range(10)],
                               'col2': np.random.randint(0, 100, (10,))})
        await storage_api.put('data2', value2)
        await storage_api.prefetch('data2')
        get_value2 = await storage_api.get('data2')
        pd.testing.assert_frame_equal(value2, get_value2)

        sliced_value = await storage_api.get('data2', conditions=[slice(3, 5), slice(None, None)])
        pd.testing.assert_frame_equal(value2.iloc[3:5, :], sliced_value)

        infos = await storage_api.get_infos('data2')
        assert infos[0].store_size > 0

        await storage_api.delete('data2')

        await storage_api.prefetch('data1')

        write_data = dataserializer.dumps(value2)
        # test open_reader and open_writer
        writer = await storage_api.open_writer('write_key', len(write_data),
                                               StorageLevel.MEMORY)
        async with writer:
            await writer.write(write_data)

        reader = await storage_api.open_reader('write_key')
        async with reader:
            read_bytes = await reader.read()
            read_value = dataserializer.loads(read_bytes)

        pd.testing.assert_frame_equal(value2, read_value)
