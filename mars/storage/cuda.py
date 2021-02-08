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

import os
from typing import Tuple, Dict, List

from ..serialization import serialize, deserialize
from ..utils import lazy_import
from .base import StorageBackend, StorageLevel, ObjectInfo
from .core import StorageFileObject

import numpy as np
import pandas as pd

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class CudaObjectId:
    def __init__(self, headers: Dict, ptrs: List[int]):
        from rmm import DeviceBuffer

        self._headers = headers
        self._ptrs = ptrs
        self._owners = [DeviceBuffer(ptr=ptr, size=length)
                        for ptr, length in zip(ptrs, headers['lengths'])]

    @property
    def headers(self):
        return self._headers

    @property
    def ptrs(self):
        return self._ptrs

    @property
    def owners(self):
        return self._owners


class CudaStorage(StorageBackend):
    def __init__(self, **kw):
        if kw:  # pragma: no cover
            raise TypeError(f'CudaStorage got unexpected arguments: {",".join(kw)}')

    @classmethod
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        if kwargs:  # pragma: no cover
            raise TypeError(f'CudaStorage got unexpected config: {",".join(kwargs)}')

        return dict(), dict()

    @staticmethod
    async def teardown(**kwargs):
        pass

    @property
    def level(self):
        return StorageLevel.GPU

    @staticmethod
    def _to_cuda(obj):
        if isinstance(obj, np.ndarray):
            return cupy.asarray(obj)
        elif isinstance(obj, pd.DataFrame):
            return cudf.DataFrame.from_pandas(obj)
        elif isinstance(obj, pd.Series):
            return cudf.Series.from_pandas(obj)
        return obj

    @staticmethod
    def _from_cuda(obj):
        if isinstance(obj, cupy.ndarray):
            obj = cupy.asnumpy(obj)
        elif cudf and isinstance(obj, (cudf.DataFrame, cudf.Series)):
            obj = obj.to_pandas()
        return obj

    async def get(self, object_id: CudaObjectId, **kwargs) -> object:
        from cupy.cuda.memory import MemoryPointer, UnownedMemory
        from cudf.core.buffer import Buffer

        headers = object_id.headers
        ptrs = object_id.ptrs
        data_type = headers.pop('data_type')
        if data_type == 'cupy':
            ptr = ptrs[0]
            size = headers['lengths'][0]
            device = headers.get('device', None)
            cupy_mem_pointer = MemoryPointer(
                UnownedMemory(ptr, size, object_id.owners[0], device_id=device), 0)
            return cupy.ndarray(
                shape=headers["shape"],
                dtype=headers["typestr"],
                memptr=cupy_mem_pointer,
                strides=headers["strides"],
            )
        elif data_type == 'cudf':
            buffers = [Buffer(ptr, owner.size, owner)
                       for ptr, owner in zip(ptrs, object_id.owners)]
            return deserialize(headers, buffers)
        else:
            raise TypeError(f'Unknown data type {data_type}')

    async def put(self, obj, importance=0) -> ObjectInfo:
        obj = self._to_cuda(obj)
        headers, buffers = serialize(obj)
        if isinstance(obj, cupy.ndarray):
            device = obj.device.id
            headers['data_type'] = 'cupy'
            ptrs = [b.data.ptr for b in buffers]
        elif isinstance(obj, (cudf.DataFrame, cudf.Series, cudf.Index)):
            device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',', 1)[0])
            headers['data_type'] = 'cudf'
            ptrs = [b.ptr for b in buffers]
        else:  # pragma: no cover
            raise TypeError(f'Unsupported data type: {type(obj)}')

        headers['device'] = device
        object_id = CudaObjectId(headers, ptrs)
        size = sum(getattr(buf, 'nbytes', len(buf)) for buf in buffers)
        return ObjectInfo(size=size, object_id=object_id, device=device)

    async def delete(self, object_id):
        pass

    async def object_info(self, object_id: CudaObjectId) -> ObjectInfo:
        size = sum(object_id.headers['lengths'])
        return ObjectInfo(size=size, object_id=object_id, device=object_id.headers['device'])

    async def open_writer(self, size=None) -> StorageFileObject:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `open_writer` method.")

    async def open_reader(self, object_id) -> StorageFileObject:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `open_reader` method.")

    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `list` method.")
