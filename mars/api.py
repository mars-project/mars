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

import logging
import uuid
import zlib
import itertools

import pyarrow

from .actors import new_client
from .errors import GraphNotExists
from .scheduler import SessionActor, GraphActor, GraphMetaActor, ResourceActor, \
    SessionManagerActor, ChunkMetaClient
from .scheduler.node_info import NodeInfoActor
from .scheduler.utils import SchedulerClusterInfoActor
from .worker.transfer import ResultSenderActor
from .tensor.expressions.utils import slice_split
from .config import options
from .serialize import dataserializer
from .utils import tokenize, calc_nsplits, merge_chunks

logger = logging.getLogger(__name__)


class MarsAPI(object):
    def __init__(self, scheduler_ip):
        self.actor_client = new_client()
        self.cluster_info = self.actor_client.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_ip)
        self.session_manager = self.get_actor_ref(SessionManagerActor.default_uid())
        self.chunk_meta_client = ChunkMetaClient(self.actor_client, self.cluster_info)

    def get_actor_ref(self, uid):
        actor_address = self.cluster_info.get_scheduler(uid)
        return self.actor_client.actor_ref(uid, address=actor_address)

    def get_graph_meta_ref(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_meta_uid = GraphMetaActor.gen_uid(session_id, graph_key)
        graph_addr = self.cluster_info.get_scheduler(graph_uid)
        return self.actor_client.actor_ref(graph_meta_uid, address=graph_addr)

    def get_schedulers_info(self):
        schedulers = self.cluster_info.get_schedulers()
        infos = dict()
        for scheduler in schedulers:
            info_ref = self.actor_client.actor_ref(NodeInfoActor.default_uid(), address=scheduler)
            infos[scheduler] = info_ref.get_info()
        return infos

    def _get_receiver_ref(self, chunk_key):
        from .worker.dispatcher import DispatchActor
        ep = self.cluster_info.get_scheduler(chunk_key)
        dispatch_ref = self.actor_client.actor_ref(DispatchActor.default_uid(), address=ep)
        uid = dispatch_ref.get_hash_slot('receiver', chunk_key)
        return self.actor_client.actor_ref(uid, address=ep)

    def count_workers(self):
        try:
            uid = ResourceActor.default_uid()
            return self.get_actor_ref(uid).get_worker_count()
        except KeyError:
            return 0

    def create_session(self, session_id, **kw):
        self.session_manager.create_session(session_id, **kw)

    def delete_session(self, session_id):
        self.session_manager.delete_session(session_id)

    def submit_graph(self, session_id, serialized_graph, graph_key, target,
                     compose=True, wait=True):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        session_ref.submit_tileable_graph(
            serialized_graph, graph_key, target, compose=compose, _tell=not wait)

    def create_mutable_tensor(self, session_id, name, shape, dtype, *args, **kwargs):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.create_mutable_tensor(name, shape, dtype, *args, **kwargs)

    def get_mutable_tensor(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.get_mutable_tensor(name)

    def send_chunk_records(self, session_id, name, chunk_records_to_send, directly=True):
        from .worker.dataio import ArrowBufferIO
        from .worker.quota import MemQuotaActor
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)

        chunk_records = []
        for chunk_key, records in chunk_records_to_send.items():
            record_chunk_key = tokenize(chunk_key, uuid.uuid4().hex)
            ep = self.cluster_info.get_scheduler(chunk_key)
            # register quota
            quota_ref = self.actor_client.actor_ref(MemQuotaActor.default_uid(), address=ep)
            quota_ref.request_batch_quota({record_chunk_key: records.nbytes})
            # send record chunk
            buf = pyarrow.serialize(records).to_buffer()
            receiver_ref = self._get_receiver_ref(chunk_key)
            receiver_ref.create_data_writer(session_id, record_chunk_key, buf.size, None,
                                            ensure_cached=False, use_promise=False)

            block_size = options.worker.transfer_block_size

            try:
                reader = ArrowBufferIO(buf, 'r', block_size=block_size)
                checksum = 0
                while True:
                    next_chunk = reader.read(block_size)
                    if not next_chunk:
                        reader.close()
                        receiver_ref.finish_receive(session_id, record_chunk_key, checksum)
                        break
                    checksum = zlib.crc32(next_chunk, checksum)
                    receiver_ref.receive_data_part(session_id, record_chunk_key, next_chunk, checksum)
            except:
                receiver_ref.cancel_receive(session_id, chunk_key)
                raise
            finally:
                if reader:
                    reader.close()
                del reader

            chunk_records.append((chunk_key, record_chunk_key))

        # register the record chunk to MutableTensorActor
        session_ref.append_chunk_records(name, chunk_records)

    def seal(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.seal(name)

    def delete_graph(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.destroy()

    def stop_graph(self, session_id, graph_key):
        from .scheduler import GraphState
        graph_meta_ref = self.get_graph_meta_ref(session_id, graph_key)
        graph_meta_ref.set_state(GraphState.CANCELLING)

        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.stop_graph()

    def get_graph_state(self, session_id, graph_key):
        from .scheduler import GraphState

        graph_meta_ref = self.get_graph_meta_ref(session_id, graph_key)
        if self.actor_client.has_actor(graph_meta_ref):
            state_obj = graph_meta_ref.get_state()
            state = state_obj.value if state_obj else 'preparing'
        else:
            raise GraphNotExists
        state = GraphState(state.lower())
        return state

    def fetch_data(self, session_id, graph_key, tileable_key, index_obj=None, compressions=None):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        chunk_indexes = graph_ref.get_tileable_chunk_indexes(tileable_key)
        chunk_shapes = self.chunk_meta_client.batch_get_chunk_shape(
            session_id, list(chunk_indexes.values()))

        if index_obj is None:
            chunk_results = dict((idx, self.fetch_chunk_data(session_id, k)) for
                                 idx, k in chunk_indexes.items())
        else:
            chunk_idx_to_shape = dict(zip(chunk_indexes.keys(), chunk_shapes))
            nsplits = calc_nsplits(chunk_idx_to_shape)
            chunk_results = dict()
            indexes = dict()
            for i, s in enumerate(index_obj):
                idx_to_slices = slice_split(s, nsplits[i])
                indexes[i] = idx_to_slices
            for chunk_index in itertools.product(*[v.keys() for v in indexes.values()]):
                slice_obj = [indexes[i][j] for i, j in enumerate(chunk_index)]
                chunk_key = chunk_indexes[chunk_index]
                chunk_results[chunk_index] = self.fetch_chunk_data(session_id, chunk_key, slice_obj)

        chunk_results = [(idx, dataserializer.loads(f.result())) for
                         idx, f in chunk_results.items()]
        if len(chunk_results) == 1:
            ret = chunk_results[0][1]
        else:
            ret = merge_chunks(chunk_results)
        return dataserializer.dumps(ret, compress=max(compressions))

    def fetch_chunk_data(self, session_id, chunk_key, index_obj=None):
        endpoints = self.chunk_meta_client.get_workers(session_id, chunk_key)
        sender_ref = self.actor_client.actor_ref(ResultSenderActor.default_uid(), address=endpoints[-1])
        future = sender_ref.fetch_data(session_id, chunk_key, index_obj, _wait=False)
        return future

    def delete_data(self, session_id, graph_key, tileable_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.free_tileable_data(tileable_key, _tell=True)

    def get_tileable_nsplits(self, session_id, graph_key, tileable_key):
        # nsplits is essential for operator like `reshape` and shape can be calculated by nsplits
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        chunk_indexes = graph_ref.get_tileable_chunk_indexes(tileable_key)

        chunk_shapes = self.chunk_meta_client.batch_get_chunk_shape(
            session_id, list(chunk_indexes.values()))

        return calc_nsplits(dict(zip(chunk_indexes.keys(), chunk_shapes)))
