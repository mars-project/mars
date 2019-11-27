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

import numpy as np
import pandas as pd

from ...serialize import dataserializer
from ...utils import calc_data_size, lazy_import
from .core import StorageHandler, DataStorageDevice, ObjectStorageMixin, \
    wrap_promised, register_storage_handler_cls

cp = lazy_import('cupy', globals=globals(), rename='cp')
cudf = lazy_import('cudf', globals=globals())


class CudaHandler(StorageHandler, ObjectStorageMixin):
    storage_type = DataStorageDevice.CUDA

    def __init__(self, storage_ctx, proc_id=None):
        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._cuda_store_ref_attr = None

    @property
    def _cuda_store_ref(self):
        if self._cuda_store_ref_attr is None:
            self._cuda_store_ref_attr = self._storage_ctx.actor_ctx.actor_ref(
                self._storage_ctx.manager_ref.get_process_holder(
                    self._proc_id, DataStorageDevice.CUDA))
        return self._cuda_store_ref_attr

    @staticmethod
    def _obj_to_cuda(o):
        if isinstance(o, np.ndarray):
            return cp.asarray(o)
        elif isinstance(o, pd.DataFrame):
            return cudf.DataFrame.from_pandas(o)
        elif isinstance(o, pd.Series):
            return cudf.Series.from_pandas(o)
        return o

    @staticmethod
    def _obj_to_mem(o):
        if cp and isinstance(o, cp.ndarray):
            o = cp.asnumpy(o)
        elif cudf and isinstance(o, (cudf.DataFrame, cudf.Series)):
            o = o.to_pandas()
        return o

    @wrap_promised
    def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        objs = self._cuda_store_ref.get_objects(session_id, data_keys)
        if serialize:
            objs = [dataserializer.serialize(self._obj_to_mem(o)) for o in objs]
        return objs

    @wrap_promised
    def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False,
                    pin_token=None, _promise=False):
        objs = [self._deserial(obj) if serialize else obj for obj in objs]
        sizes = sizes or [calc_data_size(obj) for obj in objs]

        objs = [self._obj_to_cuda(obj) for obj in objs]
        shapes = [getattr(obj, 'shape', None) for obj in objs]

        self._cuda_store_ref.put_objects(session_id, data_keys, objs, sizes, pin_token=pin_token)
        self.register_data(session_id, data_keys, sizes, shapes)

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        return self._batch_load_objects(
            session_id, data_keys,
            lambda k: src_handler.get_objects(session_id, k, _promise=True), pin_token=pin_token, batch_get=True)

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _read_serialized(reader):
            with reader:
                return reader.get_io_pool().submit(reader.read).result()

        return self._batch_load_objects(
            session_id, data_keys,
            lambda k: src_handler.create_bytes_reader(session_id, k, _promise=True).then(_read_serialized),
            serialize=True, pin_token=pin_token,
        )

    def delete(self, session_id, data_keys, _tell=False):
        self._cuda_store_ref.delete_objects(session_id, data_keys, _tell=_tell)
        self.unregister_data(session_id, data_keys, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.CUDA, CudaHandler)
