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

from .actors import new_client
from .errors import GraphNotExists
from .scheduler import SessionActor, GraphActor, GraphMetaActor, ResourceActor, \
    SessionManagerActor, ChunkMetaClient
from .scheduler.graph import ResultReceiverActor
from .scheduler.node_info import NodeInfoActor
from .scheduler.utils import SchedulerClusterInfoActor
from .serialize import dataserializer

logger = logging.getLogger(__name__)


class MarsAPI(object):
    def __init__(self, scheduler_ip):
        self.actor_client = new_client()
        self.cluster_info = self.actor_client.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_ip)
        self.session_manager = self.get_actor_ref(SessionManagerActor.default_uid())

    def get_actor_ref(self, uid):
        actor_address = self.cluster_info.get_scheduler(uid)
        return self.actor_client.actor_ref(uid, address=actor_address)

    def get_schedulers_info(self):
        schedulers = self.cluster_info.get_schedulers()
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

    def create_session(self, session_id, **kw):
        self.session_manager.create_session(session_id, **kw)

    def delete_session(self, session_id):
        self.session_manager.delete_session(session_id)

    def submit_graph(self, session_id, serialized_graph, graph_key, target,
                     compose=True, wait=True):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)
        session_ref.submit_tensor_graph(
            serialized_graph, graph_key, target, compose=compose, _tell=not wait)

    def delete_graph(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.destroy()

    def stop_graph(self, session_id, graph_key):
        from .scheduler import GraphState
        graph_meta_uid = GraphMetaActor.gen_uid(session_id, graph_key)
        self.get_actor_ref(graph_meta_uid).set_state(GraphState.CANCELLING)

        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.stop_graph()

    def get_graph_state(self, session_id, graph_key):
        from .scheduler import GraphState

        graph_meta_uid = GraphMetaActor.gen_uid(session_id, graph_key)
        graph_meta_ref = self.get_actor_ref(graph_meta_uid)
        if self.actor_client.has_actor(graph_meta_ref):
            state_obj = graph_meta_ref.get_state()
            state = state_obj.value if state_obj else 'preparing'
        else:
            raise GraphNotExists
        state = GraphState(state.lower())
        return state

    def fetch_data(self, session_id, graph_key, tensor_key, compressions=None, wait=True):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_address = self.cluster_info.get_scheduler(graph_uid)
        result_ref = self.actor_client.create_actor(ResultReceiverActor, address=graph_address)

        compressions = set(compressions or []) | {dataserializer.CompressType.NONE}
        return result_ref.fetch_tensor(session_id, graph_key, tensor_key, compressions, _wait=wait)

    def delete_data(self, session_id, graph_key, tensor_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.free_tensor_data(tensor_key, _tell=True)

    def get_tensor_nsplits(self, session_id, graph_key, tensor_key):
        # nsplits is essential for operator like `reshape` and shape can be calculated by nsplits
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        chunk_indexes = graph_ref.get_tensor_chunk_indexes(tensor_key)

        chunk_meta_client = ChunkMetaClient(self.actor_client, self.cluster_info)
        chunk_shapes = chunk_meta_client.batch_get_chunk_shape(
            session_id, list(chunk_indexes.keys()))

        # for each dimension, record chunk shape whose index is zero on other dimensions
        ndim = len(chunk_shapes[0])
        tensor_nsplits = []
        for i in range(ndim):
            splits = []
            for index, shape in zip(chunk_indexes.values(), chunk_shapes):
                if all(idx == 0 for j, idx in enumerate(index) if j != i):
                    splits.append(shape[i])
            tensor_nsplits.append(tuple(splits))

        return tuple(tensor_nsplits)
