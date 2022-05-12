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

import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from .... import oscar as mo
from .... import tensor as mt
from ....core import tile
from ....serialization import AioDeserializer, AioSerializer
from ....storage import StorageLevel
from ....tests.core import require_ray
from ....utils import get_next_port, lazy_import
from ...cluster import MockClusterAPI
from ...meta import MockMetaAPI
from ...session import MockSessionAPI
from ...web import WebActor
from ..api import MockStorageAPI, WebStorageAPI

ray = lazy_import("ray")
vineyard = lazy_import("vineyard")

require_lib = lambda x: x
storage_configs = []

# plasma backend
plasma_storage_size = 10 * 1024 * 1024
if sys.platform == "darwin":
    plasma_dir = "/tmp"
else:
    plasma_dir = "/dev/shm"
plasma_setup_params = dict(
    store_memory=plasma_storage_size, plasma_directory=plasma_dir, check_dir_size=False
)
if not sys.platform.lower().startswith("win"):
    storage_configs.append({"plasma": plasma_setup_params})

# ray backend
if ray is not None:
    require_lib = require_ray
    storage_configs.append({"ray": dict()})

# vineyard
if vineyard is not None:
    storage_configs.append({"vineyard": dict(vineyard_size="256M")})

# shared_memory
storage_configs.append({"shared_memory": dict()})


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_configs", storage_configs)
@pytest.mark.parametrize(
    "ray_start_regular", [{"enable": ray is not None}], indirect=True
)
@require_lib
async def test_storage_mock_api(ray_start_regular, storage_configs):
    start_method = "fork" if sys.platform != "win32" else None
    pool = await mo.create_actor_pool(
        "127.0.0.1",
        2,
        labels=["main", "numa-0", "io"],
        subprocess_start_method=start_method,
    )
    async with pool:
        session_id = "mock_session_id"
        storage_api = await MockStorageAPI.create(
            address=pool.external_address,
            session_id=session_id,
            storage_configs=storage_configs,
        )

        # test put and get
        value1 = np.random.rand(10, 10)
        await storage_api.put("data1", value1)
        get_value1 = await storage_api.get("data1")
        np.testing.assert_array_equal(value1, get_value1)

        value2 = pd.DataFrame(
            {
                "col1": [str(i) for i in range(10)],
                "col2": np.random.randint(0, 100, (10,)),
            }
        )
        await storage_api.put("data2", value2)
        get_value2 = await storage_api.get("data2")
        pd.testing.assert_frame_equal(value2, get_value2)

        sliced_value = await storage_api.get(
            "data2", conditions=[slice(3, 5), slice(None, None)]
        )
        pd.testing.assert_frame_equal(value2.iloc[3:5, :], sliced_value)

        infos = await storage_api.get_infos("data2")
        assert infos[0].store_size > 0

        await storage_api.delete("data2")
        buffers = await AioSerializer(value2).run()
        size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
        # test open_reader and open_writer
        writer = await storage_api.open_writer("write_key", size, StorageLevel.MEMORY)
        async with writer:
            for buf in buffers:
                await writer.write(buf)

        reader = await storage_api.open_reader("write_key")
        async with reader:
            read_value = await AioDeserializer(reader).run()

        pd.testing.assert_frame_equal(value2, read_value)

        await MockStorageAPI.cleanup(pool.external_address)


@pytest.mark.asyncio
async def test_web_storage_api():
    from ..api.web import StorageWebAPIHandler

    tempdir = tempfile.mkdtemp()
    start_method = "fork" if sys.platform != "win32" else None
    pool = await mo.create_actor_pool(
        "127.0.0.1", 1, subprocess_start_method=start_method
    )
    async with pool:
        session_id = "mock_session_id"
        await MockClusterAPI.create(address=pool.external_address)
        await MockSessionAPI.create(
            session_id=session_id, address=pool.external_address
        )
        meta_api = await MockMetaAPI.create(
            session_id=session_id, address=pool.external_address
        )
        await MockStorageAPI.create(
            address=pool.external_address,
            session_id=session_id,
            storage_configs={
                "shared_memory": dict(),
                "disk": dict(root_dirs=[tempdir]),
            },
        )

        web_config = {
            "port": get_next_port(),
            "web_handlers": {
                StorageWebAPIHandler.get_root_pattern(): StorageWebAPIHandler
            },
        }
        await mo.create_actor(WebActor, web_config, address=pool.external_address)

        web_storage_api = WebStorageAPI(
            session_id, f'http://127.0.0.1:{web_config["port"]}', "numa-0"
        )

        value = np.random.rand(10, 10)
        t = mt.random.rand(10, 10)
        t = tile(t)
        await meta_api.set_chunk_meta(
            t.chunks[0], bands=[(pool.external_address, "numa-0")]
        )
        await web_storage_api.put(t.chunks[0].key, value)

        ret_value = await web_storage_api.get(t.chunks[0].key)
        np.testing.assert_array_equal(value, ret_value)

        sliced_value = await web_storage_api.get(
            t.chunks[0].key, conditions=[slice(3, 5), slice(None, None)]
        )
        np.testing.assert_array_equal(value[3:5, :], sliced_value)

        infos = await web_storage_api.get_infos(t.chunks[0].key)
        assert len(infos) == 1
        assert infos[0].level == StorageLevel.MEMORY
        assert infos[0].memory_size == t.chunks[0].nbytes

        await MockStorageAPI.cleanup(pool.external_address)
        await MockClusterAPI.cleanup(pool.external_address)
