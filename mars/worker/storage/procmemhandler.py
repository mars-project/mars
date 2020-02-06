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

from ...lib.asyncinit import asyncinit
from ...serialize import dataserializer
from ...utils import calc_data_size
from .core import DataStorageDevice, StorageHandler, ObjectStorageMixin, \
    wrap_promised, register_storage_handler_cls


@asyncinit
class ProcMemHandler(StorageHandler, ObjectStorageMixin):
    storage_type = DataStorageDevice.PROC_MEMORY

    def __init__(self, storage_ctx, proc_id=None):
        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._inproc_store_ref_attr = None

    async def _get_inproc_store_ref(self):
        if self._inproc_store_ref_attr is None:
            self._inproc_store_ref_attr = self._storage_ctx.actor_ctx.actor_ref(
                await self._storage_ctx.manager_ref.get_process_holder(
                    self._proc_id, DataStorageDevice.PROC_MEMORY))
        return self._inproc_store_ref_attr

    @wrap_promised
    async def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        objs = await (await self._get_inproc_store_ref()).get_objects(session_id, data_keys)
        if serialize:
            objs = [dataserializer.serialize(o) for o in objs]
        return objs

    @wrap_promised
    async def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False,
                          pin_token=None, _promise=False):
        objs = [self._deserial(obj) if serialize else obj for obj in objs]
        obj = None
        try:
            sizes = sizes or [calc_data_size(obj) for obj in objs]
            shapes = [getattr(obj, 'shape', None) for obj in objs]
            await (await self._get_inproc_store_ref()).put_objects(
                session_id, data_keys, objs, sizes, pin_token=pin_token)
            await self.register_data(session_id, data_keys, sizes, shapes)
        finally:
            objs[:] = []
            del obj

    async def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        async def _read_serialized(reader):
            async with reader:
                return await reader.execute_in_pool(reader.read)

        async def _read_single(k):
            return (await src_handler.create_bytes_reader(session_id, k, _promise=True)) \
                .then(_read_serialized)

        async def _fallback(*_):
            return await self._batch_load_objects(session_id, data_keys, _read_single, serialize=True)

        return await self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    async def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        async def _fallback(*_):
            return await self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.get_objects(session_id, k, _promise=True), batch_get=True)

        return await self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    async def delete(self, session_id, data_keys, _tell=False):
        await (await self._get_inproc_store_ref()).delete_objects(session_id, data_keys, _tell=_tell)
        await self.unregister_data(session_id, data_keys, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.PROC_MEMORY, ProcMemHandler)
