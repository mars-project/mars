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

import numpy as np
import pandas as pd
import pytest

from .... import oscar as mo
from ....resource import Resource
from ....serialization import AioDeserializer, AioSerializer
from ....storage import StorageLevel
from ....tests.core import require_cudf, require_cupy
from ... import start_services, stop_services, NodeRole
from ...cluster import MockClusterAPI
from .. import StorageAPI

_is_windows = sys.platform.lower().startswith("win")


@pytest.fixture
async def actor_pools():
    async def start_pool():
        start_method = (
            os.environ.get("POOL_START_METHOD", "forkserver")
            if sys.platform != "win32"
            else None
        )
        pool = await mo.create_actor_pool(
            "127.0.0.1",
            n_process=2,
            subprocess_start_method=start_method,
            labels=["main", "numa-0", "io"],
        )
        await pool.start()
        return pool

    worker_pool = await start_pool()
    try:
        yield worker_pool
    finally:
        await worker_pool.stop()


@pytest.mark.asyncio
async def test_storage_service(actor_pools):
    worker_pool = actor_pools

    if sys.platform == "darwin":
        plasma_dir = "/tmp"
    else:
        plasma_dir = "/dev/shm"
    plasma_setup_params = dict(
        store_memory=10 * 1024 * 1024, plasma_directory=plasma_dir, check_dir_size=False
    )

    config = {
        "services": ["storage"],
        "storage": {
            "backends": ["plasma" if not _is_windows else "shared_memory"],
            "plasma": plasma_setup_params,
        },
    }

    await start_services(NodeRole.WORKER, config, address=worker_pool.external_address)

    api = await StorageAPI.create("mock_session", worker_pool.external_address)
    value1 = np.random.rand(10, 10)
    await api.put("data1", value1)
    get_value1 = await api.get("data1")
    np.testing.assert_array_equal(value1, get_value1)

    # test api in subpool
    subpool_address = list(worker_pool._sub_processes.keys())[0]
    api2 = await StorageAPI.create("mock_session", subpool_address)
    assert api2._storage_handler_ref.address == subpool_address

    get_value1 = await api2.get("data1")
    np.testing.assert_array_equal(value1, get_value1)

    sliced_value = await api2.get("data1", conditions=[slice(None, None), slice(0, 4)])
    np.testing.assert_array_equal(value1[:, :4], sliced_value)

    await api.unpin("data1")

    value2 = pd.DataFrame(value1)
    await api2.put("data2", value2)

    get_value2 = await api.get("data2")
    pd.testing.assert_frame_equal(value2, get_value2)

    # test writer and read
    buffers = await AioSerializer(value2).run()
    size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
    # test open_reader and open_writer
    writer = await api.open_writer("write_key", size, StorageLevel.MEMORY)
    async with writer:
        for buf in buffers:
            await writer.write(buf)

    reader = await api.open_reader("write_key")
    async with reader:
        read_value = await AioDeserializer(reader).run()

    pd.testing.assert_frame_equal(value2, read_value)

    await stop_services(
        NodeRole.WORKER, address=worker_pool.external_address, config=config
    )


@pytest.fixture
async def actor_pools_with_gpu():
    async def start_pool():
        start_method = (
            os.environ.get("POOL_START_METHOD", "forkserver")
            if sys.platform != "win32"
            else None
        )
        pool = await mo.create_actor_pool(
            "127.0.0.1",
            n_process=3,
            subprocess_start_method=start_method,
            labels=["main", "numa-0", "gpu-0", "io"],
        )
        await pool.start()
        return pool

    worker_pool = await start_pool()
    try:
        yield worker_pool
    finally:
        await worker_pool.stop()


@require_cupy
@require_cudf
@pytest.mark.asyncio
async def test_storage_service_with_cuda(actor_pools_with_gpu):
    import cudf
    import cupy

    worker_pool = actor_pools_with_gpu

    if sys.platform == "darwin":
        plasma_dir = "/tmp"
    else:
        plasma_dir = "/dev/shm"
    plasma_setup_params = dict(
        store_memory=10 * 1024 * 1024, plasma_directory=plasma_dir, check_dir_size=False
    )

    config = {
        "services": ["storage"],
        "storage": {
            "backends": ["plasma" if not _is_windows else "shared_memory", "cuda"],
            "plasma": plasma_setup_params,
            "cuda": dict(),
        },
    }

    await MockClusterAPI.create(
        worker_pool.external_address,
        band_to_resource={
            "numa-0": Resource(num_cpus=1),
            "gpu-0": Resource(num_gpus=1),
        },
        use_gpu=True,
    )
    await start_services(NodeRole.WORKER, config, address=worker_pool.external_address)

    storage_api = await StorageAPI.create(
        "mock_session", worker_pool.external_address, band_name="gpu-0"
    )
    data1 = cupy.asarray(np.random.rand(10, 10))
    await storage_api.put("mock_cupy_key", data1, level=StorageLevel.GPU)
    get_data1 = await storage_api.get("mock_cupy_key")
    assert isinstance(get_data1, cupy.ndarray)
    cupy.testing.assert_array_equal(data1, get_data1)

    data2 = cudf.DataFrame(
        pd.DataFrame(
            {
                "col1": np.arange(10),
                "col2": [f"str{i}" for i in range(10)],
                "col3": np.random.rand(10),
            },
        )
    )
    await storage_api.put("mock_cudf_key", data2, level=StorageLevel.GPU)
    get_data2 = await storage_api.get("mock_cudf_key")
    assert isinstance(get_data2, cudf.DataFrame)
    cudf.testing.assert_frame_equal(data2, get_data2)

    await MockClusterAPI.cleanup(worker_pool.external_address)
    await stop_services(NodeRole.WORKER, config, address=worker_pool.external_address)
