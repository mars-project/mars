#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import tempfile
import pytest

import numpy as np
import pandas as pd
import scipy.sparse as sps

from mars.lib.filesystem import LocalFileSystem
from mars.lib.sparse import SparseNDArray, SparseMatrix
from mars.serialization import serialize, deserialize
from mars.serialize import dataserializer
from mars.storage.base import StorageLevel
from mars.storage.cuda import CudaStorage
from mars.storage.filesystem import FileSystemStorage
from mars.storage.plasma import PlasmaStorage
from mars.storage.shared_memory import SharedMemoryStorage
from mars.storage.vineyard import VineyardStorage
from mars.storage.ray import RayStorage
from mars.tests.core import require_ray, require_cudf, require_cupy
try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import ray
except ImportError:
    ray = None


require_lib = lambda x: x
params = ['filesystem', 'plasma']
if vineyard:
    params.append('vineyard')
if ray:
    params.append('ray')
    require_lib = require_ray
if sys.version_info[:2] >= (3, 8):
    params.append('shared_memory')


@pytest.fixture(params=params)
async def storage_context(request):
    if request.param == 'filesystem':
        tempdir = tempfile.mkdtemp()
        params, teardown_params = await FileSystemStorage.setup(
            fs=LocalFileSystem(),
            root_dirs=[tempdir],
            level=StorageLevel.DISK)
        storage = FileSystemStorage(**params)
        assert storage.level == StorageLevel.DISK

        yield storage

        await storage.teardown(**teardown_params)
    elif request.param == 'plasma':
        plasma_storage_size = 10 * 1024 * 1024
        if sys.platform == 'darwin':
            plasma_dir = '/tmp'
        else:
            plasma_dir = '/dev/shm'
        params, teardown_params = await PlasmaStorage.setup(
            store_memory=plasma_storage_size,
            plasma_directory=plasma_dir,
            check_dir_size=False)
        storage = PlasmaStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await PlasmaStorage.teardown(**teardown_params)
    elif request.param == 'vineyard':
        vineyard_size = '256M'
        vineyard_socket = '/tmp/vineyard.sock'
        params, teardown_params = await VineyardStorage.setup(
            vineyard_size=vineyard_size,
            vineyard_socket=vineyard_socket)
        storage = VineyardStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        await VineyardStorage.teardown(**teardown_params)
    elif request.param == 'shared_memory':
        params, teardown_params = await SharedMemoryStorage.setup()
        storage = SharedMemoryStorage(**params)
        assert storage.level == StorageLevel.MEMORY

        yield storage

        teardown_params['object_ids'] = storage._object_ids
        await SharedMemoryStorage.teardown(**teardown_params)
    elif request.param == 'ray':
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
async def test_base_operations(storage_context):
    storage = storage_context

    data1 = np.random.rand(10, 10)
    put_info1 = await storage.put(data1)
    get_data1 = await storage.get(put_info1.object_id)
    np.testing.assert_array_equal(data1, get_data1)

    info1 = await storage.object_info(put_info1.object_id)
    assert info1.size == put_info1.size

    data2 = pd.DataFrame({'col1': np.arange(10),
                          'col2': [f'str{i}' for i in range(10)],
                          'col3': np.random.rand(10)},)
    put_info2 = await storage.put(data2)
    get_data2 = await storage.get(put_info2.object_id)
    pd.testing.assert_frame_equal(data2, get_data2)

    info2 = await storage.object_info(put_info2.object_id)
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

    # test writer and reader
    t = np.random.random(10)
    b = dataserializer.dumps(t)
    async with await storage.open_writer(size=len(b)) as writer:
        split = len(b) // 2
        await writer.write(b[:split])
        await writer.write(b[split:])

    async with await storage.open_reader(writer.object_id) as reader:
        content = await reader.read()
        t2 = dataserializer.loads(content)

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

    await storage.delete(put_info1.object_id)

    data2 = cudf.DataFrame(pd.DataFrame({'col1': np.arange(10),
                                         'col2': [f'str{i}' for i in range(10)],
                                         'col3': np.random.rand(10)},))
    put_info2 = await storage.put(data2)
    get_data2 = await storage.get(put_info2.object_id)
    cudf.testing.assert_frame_equal(data2, get_data2)

    info2 = await storage.object_info(put_info2.object_id)
    assert info2.size == put_info2.size

    await CudaStorage.teardown(**teardown_params)

    # test writer and reader
    t = np.random.random(10)
    b = dataserializer.dumps(t)
    async with await storage.open_writer(size=len(b)) as writer:
        split = len(b) // 2
        await writer.write(b[:split])
        await writer.write(b[split:])

    async with await storage.open_reader(writer.object_id) as reader:
        content = await reader.read()
        b = content.to_host_array().tobytes()
        t2 = dataserializer.loads(b)
    np.testing.assert_array_equal(t, t2)

    # write cupy array
    t = cupy.random.random((10,))
    headers, buffers = serialize(t)
    async with await storage.open_writer(size=len(b)) as writer:
        for buffer in buffers:
            await writer.write(buffer.data)

    async with await storage.open_reader(writer.object_id) as reader:
        b2 = await reader.read()
        t2 = deserialize(headers, [b2])

    cupy.testing.assert_array_equal(t, t2)

    await CudaStorage.teardown(**teardown_params)
