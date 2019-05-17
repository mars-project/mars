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
import os

from .utils import SchedulerActor
from .graph import GraphActor

logger = logging.getLogger(__name__)


class SessionActor(SchedulerActor):
    def __init__(self, session_id, **kwargs):
        super(SessionActor, self).__init__()
        self._session_id = session_id
        self._args = kwargs

        self._cluster_info_ref = None
        self._assigner_ref = None
        self._graph_refs = dict()

    @staticmethod
    def gen_uid(session_id):
        return 's:h1:session$%s' % session_id

    def get_argument(self, key):
        return self._args[key]

    def get_graph_refs(self):
        return self._graph_refs

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        self.set_cluster_info_ref()

        from .assigner import AssignerActor
        assigner_uid = AssignerActor.gen_uid(self._session_id)
        assigner_addr = self.get_scheduler(assigner_uid)
        self._assigner_ref = self.ctx.create_actor(
            AssignerActor, uid=assigner_uid, address=assigner_addr)

    def pre_destroy(self):
        self.ctx.destroy_actor(self._assigner_ref)
        for graph_ref in self._graph_refs.values():
            self.ctx.destroy_actor(graph_ref)

    def submit_tensor_graph(self, serialized_graph, graph_key, target_tensors=None, compose=True):
        graph_uid = GraphActor.gen_uid(self._session_id, graph_key)
        graph_ref = self.ctx.create_actor(GraphActor, self._session_id, graph_key,
                                          serialized_graph, target_tensors=target_tensors,
                                          uid=graph_uid, address=self.get_scheduler(graph_uid))
        graph_ref.execute_graph(_tell=True, compose=compose)
        self._graph_refs[graph_key] = graph_ref

    def graph_state(self, graph_key):
        return self._graph_refs[graph_key].get_state()

    def fetch_result(self, graph_key, tensor_key):
        # TODO just for test, should move to web handler
        graph_ref = self._graph_refs[graph_key]
        return graph_ref.fetch_tensor_result(tensor_key)


class SessionManagerActor(SchedulerActor):
    def __init__(self):
        super(SessionManagerActor, self).__init__()
        self._sessions = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()

    def get_sessions(self):
        return self._sessions

    def create_session(self, session_id, **kwargs):
        uid = SessionActor.gen_uid(session_id)
        scheduler_address = self.get_scheduler(uid)
        session_ref = self.ctx.create_actor(SessionActor, uid=uid, address=scheduler_address,
                                            session_id=session_id, **kwargs)
        self._sessions[session_id] = session_ref
        return session_ref

    def delete_session(self, session_id):
        if session_id in self._sessions:
            session_ref = self._sessions[session_id]
            session_ref.destroy()
            del self._sessions[session_id]
