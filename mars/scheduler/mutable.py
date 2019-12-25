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

from collections import defaultdict
import uuid

import numpy as np

from ..worker.transfer import ReceiverManagerActor
from ..utils import log_unhandled, tokenize
from .utils import SchedulerActor


class MutableTensorActor(SchedulerActor):
    """
    Actor handling processing of a Mars mutable tensor.
    """
    @staticmethod
    def gen_uid(session_id, name):
        return 's:0:mutable-tensor$%s$%s' % (session_id, name)

    def __init__(self, session_id, name, shape, dtype, graph_key, fill_value=None, chunk_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_id = session_id
        self._name = name
        self._shape = shape
        if isinstance(dtype, np.dtype):
            self._dtype = dtype
        else:
            self._dtype = np.dtype(dtype)
        self._graph_key = graph_key
        self._chunk_size = chunk_size
        self._fill_value = fill_value
        self._tensor = None
        self._sealed = False
        self._chunk_map = defaultdict(lambda: [])
        self._chunk_to_endpoint = dict()
        self._record_type = np.dtype([("index", np.uint32),
                                      ("ts", np.dtype('datetime64[ns]')),
                                      ("value", self._dtype)])
        self._resource_ref = None

    @log_unhandled
    def post_create(self):
        from .resource import ResourceActor
        from ..tensor.core import MutableTensor, MutableTensorData
        from ..tensor.utils import create_fetch_tensor

        super().post_create()
        self.set_cluster_info_ref()
        self._resource_ref = self.ctx.actor_ref(ResourceActor.default_uid())

        tensor = create_fetch_tensor(self._chunk_size, self._shape, self._dtype)
        endpoints = self._resource_ref.select_worker(size=len(tensor.chunks))

        for chunk, endpoint in zip(tensor.chunks, endpoints):
            self._chunk_to_endpoint[chunk.key] = endpoint

        self._tensor = MutableTensor(data=MutableTensorData(
            name=self._name, op=None, shape=self._shape, dtype=tensor.dtype,
            nsplits=tensor.nsplits, key=tensor.key, chunks=tensor.chunks,
            chunk_eps=endpoints))

    def tensor_meta(self):
        # avoid built-in scalar dtypes are made into one-field record type.
        if self._dtype.fields:
            dtype_descr = self._dtype.descr
        else:
            dtype_descr = str(self._dtype)
        chunk_keys = [c.key for c in self._tensor.chunks]
        chunk_eps = [self._chunk_to_endpoint[key] for key in chunk_keys]
        return self._shape, dtype_descr, self._chunk_size, chunk_keys, chunk_eps

    def tensor_key(self):
        return self._tensor.key

    def sealed(self):
        return self._sealed

    @log_unhandled
    def read(self, tensor_index):
        raise NotImplementedError

    @log_unhandled
    def write(self, index, value):
        chunk_records_to_send = self._tensor._do_write(index, value)
        self._send_chunk_records(chunk_records_to_send)

    @log_unhandled
    def append_chunk_records(self, chunk_records):
        for chunk_key, record_chunk_key in chunk_records:
            self._chunk_map[chunk_key].append(record_chunk_key)

    @log_unhandled
    def seal(self):
        from ..worker.seal import SealActor
        self._sealed = True

        # dump current buffers to worker
        chunk_records_to_send = self._tensor._do_flush()
        self._send_chunk_records(chunk_records_to_send)

        endpoint_to_sealer = dict()

        # consolidate chunks
        for chunk in self._tensor.chunks:
            ep = self._chunk_to_endpoint[chunk.key]
            if ep in endpoint_to_sealer:
                sealer_ref = endpoint_to_sealer[ep]
            else:
                sealer_uid = SealActor.gen_uid(self._session_id, chunk.key)
                sealer_ref = endpoint_to_sealer[ep] = self.ctx.create_actor(SealActor, uid=sealer_uid, address=ep)
            sealer_ref.seal_chunk(self._session_id, self._graph_key,
                                  chunk.key, self._chunk_map[chunk.key],
                                  chunk.shape, self._record_type, self._dtype, self._fill_value)

        for sealer_ref in endpoint_to_sealer.values():
            sealer_ref.destroy()
        # return the hex of self._graph_key since UUID is not json serializable.
        return self._graph_key.hex, self._tensor.key, self._tensor.id, self.tensor_meta()

    @log_unhandled
    def _send_chunk_records(self, chunk_records_to_send):
        from ..worker.transfer import put_remote_chunk

        chunk_records = []

        for chunk_key, ep, records in chunk_records_to_send:
            record_chunk_key = tokenize(chunk_key, uuid.uuid4().hex)
            # send record chunk
            receiver_manager_ref = self.ctx.actor_ref(ReceiverManagerActor.default_uid(), address=ep)
            put_remote_chunk(self._session_id, record_chunk_key, records, receiver_manager_ref)
            chunk_records.append((chunk_key, record_chunk_key))

        # register the record chunks
        self.append_chunk_records(chunk_records)
