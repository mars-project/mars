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

import itertools
import logging
import random
import uuid

from .actors import new_client, ActorNotExist
from .errors import GraphNotExists
from .scheduler import SessionActor, GraphActor, GraphMetaActor, ResourceActor, \
    SessionManagerActor, ChunkMetaClient
from .scheduler.node_info import NodeInfoActor
from .scheduler.utils import SchedulerClusterInfoActor
from .worker.transfer import ResultSenderActor, ReceiverManagerActor
from .tensor.utils import slice_split
from .serialize import dataserializer
from .utils import tokenize, merge_chunks

logger = logging.getLogger(__name__)


class MarsAPI(object):
    def __init__(self, scheduler_ip):
        self.__schedulers_cache = None
        self._session_manager = None
        self.actor_client = new_client()
        self.cluster_info = self.actor_client.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_ip)
        self.chunk_meta_client = ChunkMetaClient(self.actor_client, self.cluster_info)

    @property
    def session_manager(self):
        if self._session_manager is None:
            self._session_manager = self.get_actor_ref(SessionManagerActor.default_uid())
        return self._session_manager

    def get_schedulers(self):
        if not self.__schedulers_cache:
            self.__schedulers_cache = self.cluster_info.get_schedulers()
        return self.__schedulers_cache

    def get_scheduler(self, uid):
        schedulers = self.get_schedulers()
        if len(schedulers) == 1:
            return schedulers[0]
        else:
            return self.cluster_info.get_scheduler(uid)

    def get_actor_ref(self, uid):
        return self.actor_client.actor_ref(uid, address=self.get_scheduler(uid))

    def get_graph_meta_ref(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_meta_uid = GraphMetaActor.gen_uid(session_id, graph_key)
        graph_addr = self.get_scheduler(graph_uid)
        return self.actor_client.actor_ref(graph_meta_uid, address=graph_addr)

    def get_schedulers_info(self):
        schedulers = self.get_schedulers()
        infos = dict()
        for scheduler in schedulers:
            info_ref = self.actor_client.actor_ref(NodeInfoActor.default_uid(), address=scheduler)
            infos[scheduler] = info_ref.get_info()
        return infos

    def count_workers(self):
        try:
            uid = ResourceActor.default_uid()
            return self.get_actor_ref(uid).get_worker_count()
        except KeyError:
            return 0

    def get_workers_meta(self):
        resource_uid = ResourceActor.default_uid()
        resource_ref = self.get_actor_ref(resource_uid)
        return resource_ref.get_workers_meta()

    def create_session(self, session_id, **kw):
        self.session_manager.create_session(session_id, **kw)

    def delete_session(self, session_id):
        self.session_manager.delete_session(session_id)

    def has_session(self, session_id):
        """
        Check if the session with given session_id exists.
        """
        return self.session_manager.has_session(session_id)

    def submit_graph(self, session_id, serialized_graph, graph_key, target,
                     names=None, compose=True, wait=True):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        session_ref.submit_tileable_graph(
            serialized_graph, graph_key, target, names=names, compose=compose, _tell=not wait)

    def create_mutable_tensor(self, session_id, name, shape, dtype, *args, **kwargs):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.create_mutable_tensor(name, shape, dtype, *args, **kwargs)

    def get_mutable_tensor(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.get_mutable_tensor(name)

    def send_chunk_records(self, session_id, name, chunk_records_to_send, directly=True):
        from .worker.quota import MemQuotaActor
        from .worker.transfer import put_remote_chunk
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)

        chunk_records = []
        for chunk_key, endpoint, records in chunk_records_to_send:
            record_chunk_key = tokenize(chunk_key, uuid.uuid4().hex)
            # register quota
            quota_ref = self.actor_client.actor_ref(MemQuotaActor.default_uid(), address=endpoint)
            quota_ref.request_batch_quota({record_chunk_key: records.nbytes})
            # send record chunk
            receiver_manager_ref = self.actor_client.actor_ref(
                ReceiverManagerActor.default_uid(), address=endpoint)
            put_remote_chunk(session_id, record_chunk_key, records, receiver_manager_ref)
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
        try:
            state_obj = graph_meta_ref.get_state()
            state = state_obj.value if state_obj else 'preparing'
        except ActorNotExist:
            raise GraphNotExists
        return GraphState(state.lower())

    def get_graph_exc_info(self, session_id, graph_key):
        graph_meta_ref = self.get_graph_meta_ref(session_id, graph_key)
        try:
            return graph_meta_ref.get_exc_info()
        except ActorNotExist:
            raise GraphNotExists

    def wait_graph_finish(self, session_id, graph_key, timeout=None):
        graph_meta_ref = self.get_graph_meta_ref(session_id, graph_key)
        self.actor_client.actor_ref(graph_meta_ref.get_wait_ref()).wait(timeout)

    def fetch_chunks_data(self, session_id, chunk_indexes, chunk_keys, nsplits,
                          index_obj=None, serial=True, serial_type=None,
                          compressions=None, pickle_protocol=None):
        chunk_index_to_key = dict((index, key) for index, key in zip(chunk_indexes, chunk_keys))
        if not index_obj:
            chunk_results = dict((idx, self.fetch_chunk_data(session_id, k)) for
                                 idx, k in zip(chunk_indexes, chunk_keys))
        else:
            chunk_results = dict()
            indexes = dict()
            for axis, s in enumerate(index_obj):
                idx_to_slices = slice_split(s, nsplits[axis])
                indexes[axis] = idx_to_slices
            for chunk_index in itertools.product(*[v.keys() for v in indexes.values()]):
                # slice_obj: use tuple, since numpy complains
                #
                # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use
                # `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array
                # index, `arr[np.array(seq)]`, which will result either in an error or a different result.
                slice_obj = tuple(indexes[axis][chunk_idx] for axis, chunk_idx in enumerate(chunk_index))
                chunk_key = chunk_index_to_key[chunk_index]
                chunk_results[chunk_index] = self.fetch_chunk_data(session_id, chunk_key, slice_obj)

        chunk_results = [(idx, dataserializer.loads(f.result())) for
                         idx, f in chunk_results.items()]
        if len(chunk_results) == 1:
            ret = chunk_results[0][1]
        else:
            ret = merge_chunks(chunk_results)
        if not serial:
            return ret
        compressions = max(compressions) if compressions else dataserializer.CompressType.NONE
        return dataserializer.dumps(ret, serial_type=serial_type, compress=compressions,
                                    pickle_protocol=pickle_protocol)

    def fetch_data(self, session_id, graph_key, tileable_key, index_obj=None,
                   serial=True, serial_type=None, compressions=None, pickle_protocol=None):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        nsplits, chunk_keys, chunk_indexes = graph_ref.get_tileable_metas([tileable_key])[0]
        return self.fetch_chunks_data(session_id, chunk_indexes, chunk_keys, nsplits,
                                      index_obj=index_obj, serial=serial, serial_type=serial_type,
                                      compressions=compressions, pickle_protocol=pickle_protocol)

    def fetch_chunk_data(self, session_id, chunk_key, index_obj=None):
        endpoints = self.chunk_meta_client.get_workers(session_id, chunk_key)
        if endpoints is None:
            raise KeyError('Chunk key %s not exist in cluster' % chunk_key)
        sender_ref = self.actor_client.actor_ref(ResultSenderActor.default_uid(),
                                                 address=random.choice(endpoints))
        return sender_ref.fetch_data(session_id, chunk_key, index_obj, _wait=False)

    def get_chunk_metas(self, session_id, chunk_keys):
        return self.chunk_meta_client.batch_get_chunk_meta(session_id, chunk_keys)

    def delete_data(self, session_id, graph_key, tileable_key, wait=False):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.free_tileable_data(tileable_key, wait=wait, _tell=not wait)

    def get_tileable_nsplits(self, session_id, graph_key, tileable_key):
        # nsplits is essential for operator like `reshape` and shape can be calculated by nsplits
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)

        return graph_ref.get_tileable_metas([tileable_key], filter_fields=['nsplits'])[0][0]

    def get_tileable_key_by_name(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        return session_ref.get_tileable_key(name)
