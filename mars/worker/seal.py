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

from ..utils import log_unhandled
from .storage import DataStorageDevice
from .utils import WorkerActor


class SealActor(WorkerActor):
    """
    Actor sealing a chunk from a serials of record chunks.
    """
    @staticmethod
    def gen_uid(session_id, chunk_key):
        return 's:0:seal$%s$%s' % (session_id, chunk_key)

    def __init__(self):
        super().__init__()
        self._mem_quota_ref = None

    def post_create(self):
        from .quota import MemQuotaActor
        super().post_create()
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

    @log_unhandled
    def seal_chunk(self, session_id, graph_key, chunk_key, keys, shape, record_type, dtype, fill_value):
        chunk_bytes_size = np.prod(shape) * dtype.itemsize
        self._mem_quota_ref.request_batch_quota({chunk_key: chunk_bytes_size})
        if fill_value is None:
            ndarr = np.zeros(shape, dtype=dtype)
        else:
            ndarr = np.full(shape, fill_value, dtype=dtype)
        ndarr_ts = np.zeros(shape, dtype=np.dtype('datetime64[ns]'))

        # consolidate
        for key in keys:
            buffer = None
            try:
                # todo potential memory quota issue must be dealt with
                obj = self.storage_client.get_object(
                    session_id, key, [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK], _promise=False)
                record_view = obj.view(dtype=record_type, type=np.recarray)

                for record in record_view:
                    idx = np.unravel_index(record.index, shape)
                    if record.ts > ndarr_ts[idx]:
                        ndarr[idx] = record.value
            finally:
                del buffer

            # clean up
            self.storage_client.delete(session_id, [key])
            self.get_meta_client().delete_meta(session_id, key, False)

        self._mem_quota_ref.release_quotas(keys)

        self.storage_client.put_objects(
            session_id, [chunk_key], [ndarr], [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK])
        self.get_meta_client().set_chunk_meta(session_id, chunk_key, size=chunk_bytes_size,
                                              shape=shape, workers=(self.address,))
