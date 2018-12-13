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
from .cluster_info import ClusterInfoActor
from .node_info import NodeInfoActor
from .scheduler import SessionActor, GraphActor, KVStoreActor
from .scheduler.session import SessionManagerActor
from .scheduler.graph import ResultReceiverActor

logger = logging.getLogger(__name__)


class MarsAPI(object):
    def __init__(self, scheduler_ip):
        self.actor_client = new_client()
        self.cluster_info = self.actor_client.actor_ref(
            ClusterInfoActor.default_name(), address=scheduler_ip)
        self.kv_store = self.get_actor_ref(KVStoreActor.default_name())
        self.session_manager = self.get_actor_ref(SessionManagerActor.default_name())

    def get_actor_ref(self, uid):
        actor_address = self.cluster_info.get_scheduler(uid)
        return self.actor_client.actor_ref(uid, address=actor_address)

    def get_schedulers_info(self):
        schedulers = self.cluster_info.get_schedulers()
        infos = []
        for scheduler in schedulers:
            info_ref = self.actor_client.actor_ref(NodeInfoActor.default_name(),
                                                   address=scheduler)
            infos.append(info_ref.get_info())
        return infos

    def count_workers(self):
        try:
            worker_info = self.kv_store.read('/workers/meta')
            workers_num = len(worker_info.children)
            return workers_num
        except KeyError:
            return 0

    def create_session(self, session_id, **kw):
        self.session_manager.create_session(session_id, **kw)

    def delete_session(self, session_id):
        self.session_manager.delete_session(session_id)

    def submit_graph(self, session_id, serialized_graph, graph_key, target):
        session_uid = SessionActor.gen_name(session_id)
        session_ref = self.get_actor_ref(session_uid)
        session_ref.submit_tensor_graph(serialized_graph, graph_key, target, _tell=True)

    def delete_graph(self, session_id, graph_key):
        graph_uid = GraphActor.gen_name(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.destroy()

    def stop_graph(self, session_id, graph_key):
        graph_uid = GraphActor.gen_name(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.stop_graph()

    def get_graph_state(self, session_id, graph_key):
        from .scheduler.utils import GraphState

        state_obj = self.kv_store.read(
            '/sessions/%s/graph/%s/state' % (session_id, graph_key), silent=True)
        state = state_obj.value if state_obj else 'preparing'
        state = GraphState(state.lower())
        return state

    def fetch_data(self, session_id, graph_key, tensor_key):
        graph_uid = GraphActor.gen_name(session_id, graph_key)
        graph_address = self.cluster_info.get_scheduler(graph_uid)
        result_ref = self.actor_client.create_actor(ResultReceiverActor, address=graph_address)
        return result_ref.fetch_tensor(session_id, graph_key, tensor_key)

    def delete_data(self, session_id, graph_key, tensor_key):
        graph_uid = GraphActor.gen_name(session_id, graph_key)
        graph_ref = self.get_actor_ref(graph_uid)
        graph_ref.free_tensor_data(tensor_key, _tell=True)
