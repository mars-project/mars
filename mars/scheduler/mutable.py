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

from collections import defaultdict

import numpy as np

from ..utils import log_unhandled
from .utils import SchedulerActor


class MutableTensorActor(SchedulerActor):
    """
    Actor handling processing of a Mars mutable tensor.
    """
    @staticmethod
    def gen_uid(session_id, name):
        return 's:0:mutable-tensor$%s$%s' % (session_id, name)

    def __init__(self, session_id, name, shape, dtype, graph_key, chunk_size=None, *args, **kwargs):
        super(MutableTensorActor, self).__init__(*args, **kwargs)
        self._session_id = session_id
        self._name = name
        self._shape = shape
        if isinstance(dtype, np.dtype):
            self._dtype = dtype
        else:
            self._dtype = np.dtype(dtype)
        self._graph_key = graph_key
        self._chunk_size = chunk_size
        self._tensor = None
        self._sealed = False
        self._chunk_map = defaultdict(lambda: [])
        self._record_type = np.dtype([("index", np.uint32), ("ts", np.dtype('datetime64[ns]')), ("value", self._dtype)])

    @log_unhandled
    def post_create(self):
        from ..tensor.expressions.utils import create_fetch_tensor

        super(MutableTensorActor, self).post_create()
        self.set_cluster_info_ref()
        self._tensor = create_fetch_tensor(self._chunk_size, self._shape, self._dtype)

    def tensor_meta(self):
        return self._shape, self._dtype, self._chunk_size, [c.key for c in self._tensor.chunks]

    def tensor_key(self):
        return self._tensor.key

    def sealed(self):
        return self._sealed

    @log_unhandled
    def read(self, tensor_index):
        raise NotImplementedError

    @log_unhandled
    def append_chunk_records(self, chunk_records):
        for chunk_key, record_chunk_key in chunk_records:
            self._chunk_map[chunk_key].append(record_chunk_key)

    @log_unhandled
    def seal(self):
        from ..worker.seal import SealActor
        self._sealed = True
        for chunk in self._tensor.chunks:
            ep = self.get_scheduler(chunk.key)
            sealer_uid = SealActor.gen_uid(self._session_id, chunk.key)
            sealer_ref = self.ctx.create_actor(SealActor, uid=sealer_uid, address=ep)

            sealer_ref.seal_chunk(self._session_id, self._graph_key,
                                  chunk.key, self._chunk_map[chunk.key],
                                  chunk.shape, self._record_type, self._dtype)
        return self._graph_key, self._tensor.key, self._tensor.id, self.tensor_meta()
