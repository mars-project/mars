# Copyright 1999-2019 Alibaba Group Holding Ltd.
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

    @wrap_promised
    def get_object(self, session_id, data_key, serialized=False, _promise=False):
        obj = self._cuda_store_ref.get_object(session_id, data_key)
        if serialized:
            if cp and isinstance(obj, cp.ndarray):
                obj = cp.asnumpy(obj)
            elif cudf and isinstance(obj, (cudf.DataFrame, cudf.Series)):
                obj = obj.to_pandas()
            obj = dataserializer.serialize(obj)
        return obj

    @wrap_promised
    def put_object(self, session_id, data_key, obj, serialized=False, _promise=False):
        o = self._deserial(obj) if serialized else obj
        data_size = calc_data_size(o)

        if isinstance(o, np.ndarray):
            o = cp.asarray(o)
        elif isinstance(o, pd.DataFrame):
            o = cudf.DataFrame.from_pandas(o)
        elif isinstance(o, pd.Series):
            o = cudf.Series.from_pandas(o)

        self._cuda_store_ref.put_object(session_id, data_key, o)
        self.register_data(session_id, data_key, data_size, shape=getattr(o, 'shape', None))

    def load_from_object_io(self, session_id, data_key, src_handler):
        return src_handler.get_object(session_id, data_key, _promise=True) \
            .then(lambda obj: self.put_object(session_id, data_key, obj))

    def load_from_bytes_io(self, session_id, data_key, src_handler):
        def _read_and_put(reader):
            with reader:
                result = reader.get_io_pool().submit(reader.read).result()
            self.put_object(session_id, data_key, result, serialized=True)

        return src_handler.create_bytes_reader(session_id, data_key, _promise=True) \
            .then(_read_and_put)

    def delete(self, session_id, data_key, _tell=False):
        self._cuda_store_ref.delete_object(session_id, data_key, _tell=_tell)
        self.unregister_data(session_id, data_key, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.CUDA, CudaHandler)
