#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import pkgutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from ...lib.filesystem import LocalFileSystem
from ...lib.sparse import SparseNDArray, SparseMatrix
from ...serialization import AioSerializer, AioDeserializer
from ...tests.core import require_ray, require_cudf, require_cupy
from ..base import StorageLevel
from ..cuda import CudaStorage
from ..filesystem import DiskStorage
from ..plasma import PlasmaStorage
from ..shared_memory import SharedMemoryStorage
from ..vineyard import VineyardStorage
from ..ray import RayStorage

try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import ray
except ImportError:
    ray = None

require_lib = lambda x: x
params = [
    "filesystem",
    "shared_memory",
]
if (
    not sys.platform.startswith("win")
    and pkgutil.find_loader("pyarrow.plasma") is not None
):
    params.append("plasma")
if vineyard is not None:
    params.append("vineyard")
if ray is not None:
    params.append("ray")
    require_lib = require_ray


@pytest.mark.parametrize(
    "ray_start_regular", [{"enable": ray is not None}], indirect=True
)
@pytest.fixture(params=params)
async def storage_context(ray_start_regular, request):
    if request.param == "filesystem":
        tempdir = tempfile.mkdtemp()
        params, teardown_params = await DiskStorage.setup(
            fs=LocalFileSystem(), root_dirs=[tempdir]
        )
        storage = DiskStorage(**params)
        assert storage.level == StorageLevel.DISK

        yield storage

        await storage.teardown(**teardown_params)
    elif request.param == "plasma":
        plasma_storage_size = 10 * 1024 * 1024
        if sys.platform == "darwin":
            plasma_dir = "/tmp"
        else:
            plasma_dir = "/dev/shm"
        params, teardown_params = await PlasmaStorage.setup(
            store_memory=plasma_storage_size,
            plasma_directory=plasma_dir,
            check_dir_size=False,
        )
        storage = PlasmaStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await PlasmaStorage.teardown(**teardown_params)
    elif request.param == "vineyard":
        vineyard_size = "256M"
        params, teardown_params = await VineyardStorage.setup(
            vineyard_size=vineyard_size
        )
        storage = VineyardStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await VineyardStorage.teardown(**teardown_params)
    elif request.param == "shared_memory":
        params, teardown_params = await SharedMemoryStorage.setup()
        storage = SharedMemoryStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        teardown_params["object_ids"] = storage._object_ids
        await SharedMemoryStorage.teardown(**teardown_params)
    elif request.param == "ray":
        params, teardown_params = await RayStorage.setup()
        storage = RayStorage(**params)
        assert storage.level == StorageLevel.MEMORY | StorageLevel.REMOTE

        yield storage

        await RayStorage.teardown(**teardown_params)


def test_storage_level():
    level = StorageLevel.DISK | StorageLevel.MEMORY
    assert level == StorageLevel.DISK.value | StorageLevel.MEMORY.value

    assert (StorageLevel.DISK | StorageLevel.MEMORY) & StorageLevel.DISK
    assert not (StorageLevel.DISK | StorageLevel.MEMORY) & StorageLevel.GPU

    assert StorageLevel.GPU < StorageLevel.MEMORY < StorageLevel.DISK
    assert StorageLevel.DISK > StorageLevel.MEMORY > StorageLevel.GPU


@pytest.mark.asyncio
@require_lib
@pytest.mark.parametrize(
    "ray_start_regular", [{"enable": ray is not None}], indirect=True
)
async def test_base_operations(ray_start_regular, storage_context):
    storage = storage_context

    data1 = np.random.rand(10, 10)
    put_info1 = await storage.put(data1)
    get_data1 = await storage.get(put_info1.object_id)
    np.testing.assert_array_equal(data1, get_data1)

    info1 = await storage.object_info(put_info1.object_id)
    # FIXME: remove os check when size issue fixed
    assert info1.size == put_info1.size

    data2 = pd.DataFrame(
        {
            "col1": np.arange(10),
            "col2": [f"str{i}" for i in range(10)],
            "col3": np.random.rand(10),
        },
    )
    put_info2 = await storage.put(data2)
    get_data2 = await storage.get(put_info2.object_id)
    pd.testing.assert_frame_equal(data2, get_data2)

    info2 = await storage.object_info(put_info2.object_id)
    # FIXME: remove os check when size issue fixed
    assert info2.size == put_info2.size

    # FIXME: remove when list functionality is ready for vineyard.
    if not isinstance(storage, (VineyardStorage, SharedMemoryStorage, RayStorage)):
        num = len(await storage.list())
        assert num == 2
        await storage.delete(info2.object_id)

    # test SparseMatrix
    s1 = sps.csr_matrix([[1, 0, 1], [0, 0, 1]])
    s = SparseNDArray(s1)
    put_info3 = await storage.put(s)
    get_data3 = await storage.get(put_info3.object_id)
    assert isinstance(get_data3, SparseMatrix)
    np.testing.assert_array_equal(get_data3.toarray(), s1.A)
    np.testing.assert_array_equal(get_data3.todense(), s1.A)


@pytest.mark.asyncio
@require_lib
@pytest.mark.parametrize(
    "ray_start_regular", [{"enable": ray is not None}], indirect=True
)
async def test_reader_and_writer(ray_start_regular, storage_context):
    storage = storage_context

    if isinstance(storage, VineyardStorage):
        pytest.skip(
            "open_{reader,writer} in vineyard doesn't use the DEFAULT_SERIALIZATION"
        )

    # test writer and reader
    t = np.random.random(10)
    buffers = await AioSerializer(t).run()
    size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
    async with await storage.open_writer(size=size) as writer:
        for buf in buffers:
            await writer.write(buf)

    async with await storage.open_reader(writer.object_id) as reader:
        r = await AioDeserializer(reader).run()

    np.testing.assert_array_equal(t, r)

    # test writer and reader with seek offset
    t = np.random.random(10)
    buffers = await AioSerializer(t).run()
    size = sum(getattr(buf, "nbytes", len(buf)) for buf in buffers)
    async with await storage.open_writer(size=20 + size) as writer:
        await writer.write(b" " * 10)
        for buf in buffers:
            await writer.write(buf)
        await writer.write(b" " * 10)

    async with await storage.open_reader(writer.object_id) as reader:
        with pytest.raises((OSError, ValueError)):
            await reader.seek(-1)

        assert 5 == await reader.seek(5)
        assert 10 == await reader.seek(5, os.SEEK_CUR)
        assert 10 == await reader.seek(-10 - size, os.SEEK_END)
        assert 10 == await reader.tell()
        r = await AioDeserializer(reader).run()

    np.testing.assert_array_equal(t, r)


@pytest.mark.asyncio
@require_lib
@pytest.mark.parametrize(
    "ray_start_regular", [{"enable": ray is not None}], indirect=True
)
async def test_reader_and_writer_vineyard(ray_start_regular, storage_context):
    storage = storage_context

    if not isinstance(storage, VineyardStorage):
        pytest.skip(
            "open_{reader,writer} in vineyard doesn't use the DEFAULT_SERIALIZATION"
        )

    # test writer and reader
    t = np.random.random(10)
    tinfo = await storage.put(t)

    # testing the roundtrip of `open_{reader,writer}`.

    buffers = []
    async with await storage.open_reader(tinfo.object_id) as reader:
        while True:
            buf = await reader.read()
            if buf:
                buffers.append(buf)
            else:
                break

    writer_object_id = None
    async with await storage.open_writer() as writer:
        for buf in buffers:
            await writer.write(buf)

        # The `object_id` of `StorageFileObject` returned by `open_writer` in vineyard
        # storage only available after `close` and before `__exit__` of `AioFileObject`.
        #
        # As `StorageFileObject.object_id` is only used for testing here, I think its
        # fine to have such a hack.
        await writer.close()
        writer_object_id = writer._file._object_id

    t2 = await storage.get(writer_object_id)
    np.testing.assert_array_equal(t, t2)


@require_cupy
@require_cudf
@pytest.mark.asyncio
async def test_cuda_backend():
    import cupy
    import cudf

    params, teardown_params = await CudaStorage.setup()
    storage = CudaStorage(**params)
    assert storage.level == StorageLevel.GPU

    data1 = cupy.asarray(np.random.rand(10, 10))
    put_info1 = await storage.put(data1)
    get_data1 = await storage.get(put_info1.object_id)
    cupy.testing.assert_array_equal(data1, get_data1)

    info1 = await storage.object_info(put_info1.object_id)
    assert info1.size == put_info1.size

    data2 = cudf.DataFrame(
        pd.DataFrame(
            {
                "col1": np.arange(10),
                "col2": [f"str{i}" for i in range(10)],
                "col3": np.random.rand(10),
            },
        )
    )
    put_info2 = await storage.put(data2)
    get_data2 = await storage.get(put_info2.object_id)
    cudf.testing.assert_frame_equal(data2, get_data2)

    info2 = await storage.object_info(put_info2.object_id)
    assert info2.size == put_info2.size

    await CudaStorage.teardown(**teardown_params)

    # test writer and reader
    read_chunk = 100
    writer = await storage.open_writer(put_info1.size)
    async with await storage.open_reader(put_info1.object_id) as reader:
        while True:
            content = await reader.read(read_chunk)
            if content:
                await writer.write(content)
            else:
                break
    writer._file._write_close()
    write_data = await storage.get(writer._file._object_id)
    cupy.testing.assert_array_equal(write_data, get_data1)

    await storage.delete(put_info1.object_id)
