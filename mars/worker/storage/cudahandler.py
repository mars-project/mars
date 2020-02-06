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

import itertools

import numpy as np
import pandas as pd

from ...errors import StorageFull
from ...serialize import dataserializer
from ...utils import calc_data_size, lazy_import
from .core import StorageHandler, DataStorageDevice, ObjectStorageMixin, \
    SpillableStorageMixin, wrap_promised, register_storage_handler_cls

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())


class CudaHandler(StorageHandler, ObjectStorageMixin, SpillableStorageMixin):
    storage_type = DataStorageDevice.CUDA

    def __init__(self, storage_ctx, proc_id=None):
        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._cuda_store_ref_attr = None
        self._cuda_size_limit_attr = None

    async def _get_cuda_store_ref(self):
        if self._cuda_store_ref_attr is None:
            self._cuda_store_ref_attr = self._storage_ctx.promise_ref(
                await self._storage_ctx.manager_ref.get_process_holder(
                    self._proc_id, DataStorageDevice.CUDA))
        return self._cuda_store_ref_attr

    async def _get_cuda_size_limit(self):
        if self._cuda_size_limit_attr is None:
            ref = await self._get_cuda_store_ref()
            self._cuda_size_limit_attr = await ref.get_size_limit()
        return self._cuda_size_limit_attr

    @staticmethod
    def _obj_to_cuda(o):
        from cupy.cuda.memory import OutOfMemoryError
        from numba.cuda.cudadrv.driver import CudaAPIError
        try:
            if isinstance(o, np.ndarray):
                return cp.asarray(o)
            elif isinstance(o, pd.DataFrame):
                return cudf.DataFrame.from_pandas(o)
            elif isinstance(o, pd.Series):
                return cudf.Series.from_pandas(o)
            return o
        except OutOfMemoryError:
            raise StorageFull
        except CudaAPIError as ex:
            if ex.code == 1:  # CUDA_ERROR_INVALID_VALUE
                raise StorageFull
            raise

    @staticmethod
    def _obj_to_mem(o):
        if cp and isinstance(o, cp.ndarray):
            o = cp.asnumpy(o)
        elif cudf and isinstance(o, (cudf.DataFrame, cudf.Series)):
            o = o.to_pandas()
        return o

    @wrap_promised
    async def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        ref = await self._get_cuda_store_ref()
        objs = await ref.get_objects(session_id, data_keys)
        if serialize:
            objs = [dataserializer.serialize(self._obj_to_mem(o)) for o in objs]
        return objs

    @wrap_promised
    async def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False,
                          pin_token=None, _promise=False):
        objs = [self._deserial(obj) if serialize else obj for obj in objs]
        sizes = sizes or [calc_data_size(obj) for obj in objs]

        obj = None
        succ_keys, succ_objs, succ_sizes, succ_shapes = [], [], [], []
        affected_keys = []
        request_size, capacity = 0, 0

        ref = await self._get_cuda_store_ref()
        try:
            for idx, key, obj, size in zip(itertools.count(0), data_keys, objs, sizes):
                try:
                    obj = objs[idx] = self._obj_to_cuda(obj)
                    succ_objs.append(obj)
                    succ_keys.append(key)
                    succ_shapes.append(getattr(obj, 'shape', None))
                    succ_sizes.append(size)
                except StorageFull:
                    affected_keys.append(key)
                    request_size += size
                    capacity = await self._get_cuda_size_limit()

            await ref.put_objects(
                session_id, succ_keys, succ_objs, succ_sizes, pin_token=pin_token)
            await self.register_data(session_id, succ_keys, succ_sizes, succ_shapes)

            if affected_keys:
                raise StorageFull(request_size=request_size, capacity=capacity,
                                  affected_keys=affected_keys)
        finally:
            del obj
            objs[:] = []
            succ_objs[:] = []

    async def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        async def keys_loader(keys):
            return await src_handler.get_objects(session_id, keys, _promise=True)

        return self._batch_load_objects(
            session_id, data_keys, keys_loader, pin_token=pin_token, batch_get=True)

    async def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        async def _read_serialized(reader):
            async with reader:
                return await reader.execute_in_pool(reader.read)

        return await self._batch_load_objects(
            session_id, data_keys,
            lambda k: src_handler.create_bytes_reader(session_id, k, _promise=True).then(_read_serialized),
            serialize=True, pin_token=pin_token,
        )

    async def delete(self, session_id, data_keys, _tell=False):
        ref = await self._get_cuda_store_ref()
        await ref.delete_objects(session_id, data_keys, _tell=_tell)
        await self.unregister_data(session_id, data_keys, _tell=_tell)

    async def spill_size(self, size, multiplier=1):
        ref = await self._get_cuda_store_ref()
        return ref.spill_size(size, multiplier, _promise=True)

    async def lift_data_keys(self, session_id, data_keys):
        ref = await self._get_cuda_store_ref()
        await ref.lift_data_keys(session_id, data_keys, _tell=True)

    async def pin_data_keys(self, session_id, data_keys, token):
        ref = await self._get_cuda_store_ref()
        return await ref.pin_data_keys(session_id, data_keys, token)

    async def unpin_data_keys(self, session_id, data_keys, token, _tell=False):
        ref = await self._get_cuda_store_ref()
        return await ref.unpin_data_keys(session_id, data_keys, token, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.CUDA, CudaHandler)
