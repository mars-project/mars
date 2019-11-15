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

from ...serialize import dataserializer
from ...utils import calc_data_size
from .core import DataStorageDevice, StorageHandler, ObjectStorageMixin, \
    wrap_promised, register_storage_handler_cls


class ProcMemHandler(StorageHandler, ObjectStorageMixin):
    storage_type = DataStorageDevice.PROC_MEMORY

    def __init__(self, storage_ctx, proc_id=None):
        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._inproc_store_ref_attr = None

    @property
    def _inproc_store_ref(self):
        if self._inproc_store_ref_attr is None:
            self._inproc_store_ref_attr = self._storage_ctx.actor_ctx.actor_ref(
                self._storage_ctx.manager_ref.get_process_holder(
                    self._proc_id, DataStorageDevice.PROC_MEMORY))
        return self._inproc_store_ref_attr

    @wrap_promised
    def get_object(self, session_id, data_key, serialized=False, _promise=False):
        obj = self._inproc_store_ref.get_object(session_id, data_key)
        if serialized:
            obj = dataserializer.serialize(obj)
        return obj

    @wrap_promised
    def put_objects(self, session_id, data_keys, objs, sizes=None, serialized=False, _promise=False):
        objs = [self._deserial(obj) if serialized else obj for obj in objs]
        sizes = sizes or [calc_data_size(obj) for obj in objs]
        shapes = [getattr(obj, 'shape', None) for obj in objs]
        self._inproc_store_ref.put_objects(session_id, data_keys, objs, sizes)
        self.register_data(session_id, data_keys, sizes, shapes)

    def load_from_bytes_io(self, session_id, data_keys, src_handler):
        def _read_serialized(reader):
            with reader:
                return reader.get_io_pool().submit(reader.read).result()

        def _fallback(*_):
            return self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.create_bytes_reader(session_id, k, _promise=True).then(_read_serialized),
                True
            )

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def load_from_object_io(self, session_id, data_keys, src_handler):
        def _fallback(*_):
            return self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.get_object(session_id, k, _promise=True))

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def delete(self, session_id, data_keys, _tell=False):
        self._inproc_store_ref.delete_objects(session_id, data_keys, _tell=_tell)
        self.unregister_data(session_id, data_keys, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.PROC_MEMORY, ProcMemHandler)
