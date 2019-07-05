# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
import pyarrow

from ..config import options
from ..errors import SpillNotConfigured
from ..serialize import dataserializer
from ..utils import log_unhandled
from .spill import build_spill_file_name
from .utils import WorkerActor


class SealActor(WorkerActor):
    """
    Actor sealing a chunk from a serials of record chunks.
    """
    @staticmethod
    def gen_uid(session_id, chunk_key):
        return 's:0:seal$%s$%s' % (session_id, chunk_key)

    def __init__(self):
        super(SealActor, self).__init__()
        self._chunk_holder_ref = None
        self._mem_quota_ref = None

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .quota import MemQuotaActor
        super(SealActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

    @log_unhandled
    def seal_chunk(self, session_id, graph_key, chunk_key, keys, shape, record_type, dtype):
        from ..serialize.dataserializer import decompressors, mars_serialize_context
        chunk_bytes_size = np.prod(shape) * dtype.itemsize
        self._mem_quota_ref.request_batch_quota({chunk_key: chunk_bytes_size})
        ndarr = np.zeros(shape, dtype=dtype)
        ndarr_ts = np.zeros(shape, dtype=np.dtype('datetime64[ns]'))

        # consolidate
        for key in keys:
            try:
                if self._chunk_store.contains(session_id, key):
                    buf = self._chunk_store.get_buffer(session_id, key)
                else:
                    file_name = build_spill_file_name(key)
                    # The `disk_compression` is used in `_create_writer`
                    disk_compression = dataserializer.CompressType(options.worker.disk_compression)
                    if not file_name:
                        raise SpillNotConfigured('Spill not configured')
                    with open(file_name, 'rb') as inf:
                        buf = decompressors[disk_compression](inf.read())
                buffer = pyarrow.deserialize(memoryview(buf), mars_serialize_context())
                record_view = np.asarray(memoryview(buffer)).view(dtype=record_type, type=np.recarray)

                for record in record_view:
                    idx = np.unravel_index(record.index, shape)
                    if record.ts > ndarr_ts[idx]:
                        ndarr[idx] = record.value
            finally:
                del buf

            # clean up
            self._chunk_holder_ref.unregister_chunk(session_id, key)
            self.get_meta_client().delete_meta(session_id, key, False)
            self._mem_quota_ref.release_quota(key)

        # Hold the reference of the chunk before register_chunk
        chunk_ref = self._chunk_store.put(session_id, chunk_key, ndarr)
        self.get_meta_client().set_chunk_meta(session_id, chunk_key, size=chunk_bytes_size,
                                              shape=shape, workers=(self.address,))
        self._chunk_holder_ref.register_chunk(session_id, chunk_key)
        del chunk_ref
