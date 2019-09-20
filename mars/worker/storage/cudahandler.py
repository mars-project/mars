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


from .core import StorageHandler, DataStorageDevice, ObjectStorageMixin, \
    register_storage_handler_cls


class CudaHandler(StorageHandler, ObjectStorageMixin):
    storage_type = DataStorageDevice.CUDA

    def __init__(self, storage_ctx):
        StorageHandler.__init__(self, storage_ctx)

        self._proc_id = storage_ctx.proc_id
        self._cuda_store_ref_attr = None

    @property
    def _inproc_store_ref(self):
        if self._cuda_store_ref_attr is None:
            self._cuda_store_ref_attr = self._storage_ctx.actor_ctx.actor_ref(
                self._storage_ctx.manager_ref.get_process_holder(
                    self._proc_id, DataStorageDevice.CUDA))
        return self._cuda_store_ref_attr

    def load_from_object_io(self, session_id, data_key, src_handler):
        pass


register_storage_handler_cls(DataStorageDevice.CUDA, CudaHandler)
