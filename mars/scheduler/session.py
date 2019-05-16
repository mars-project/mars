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
from ..utils import log_unhandled

logger = logging.getLogger(__name__)


class SessionActor(SchedulerActor):
    def __init__(self, session_id, **kwargs):
        super(SessionActor, self).__init__()
        self._session_id = session_id
        self._args = kwargs

        self._cluster_info_ref = None
        self._manager_ref = None
        self._graph_refs = dict()
        self._graph_meta_refs = dict()
        self._tensor_to_graph = dict()

    @staticmethod
    def gen_uid(session_id):
        return 's:h1:session$%s' % session_id

    def get_argument(self, key):
        return self._args[key]

    def get_graph_refs(self):
        return self._graph_refs

    def get_graph_meta_refs(self):
        return self._graph_meta_refs

    def get_graph_ref_by_tensor_key(self, tensor_key):
        return self._tensor_to_graph[tensor_key]

    def post_create(self):
        super(SessionActor, self).post_create()
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        self.set_cluster_info_ref()
        self._manager_ref = self.ctx.actor_ref(SessionManagerActor.default_name())

    def pre_destroy(self):
        super(SessionActor, self).pre_destroy()
        self._manager_ref.delete_session(self._session_id, _tell=True)
        for graph_ref in self._graph_refs.values():
            self.ctx.destroy_actor(graph_ref)

    @log_unhandled
    def submit_tensor_graph(self, serialized_graph, graph_key, target_tensors=None, compose=True):
        from .graph import GraphActor, GraphMetaActor

        graph_uid = GraphActor.gen_uid(self._session_id, graph_key)
        graph_addr = self.get_scheduler(graph_uid)
        graph_ref = self.ctx.create_actor(GraphActor, self._session_id, graph_key,
                                          serialized_graph, target_tensors=target_tensors,
                                          uid=graph_uid, address=graph_addr)
        self._graph_refs[graph_key] = graph_ref
        self._graph_meta_refs[graph_key] = self.ctx.actor_ref(
            GraphMetaActor.gen_uid(self._session_id, graph_key), address=graph_addr)

        graph_ref.execute_graph(_tell=True, compose=compose)
        for tensor_key in target_tensors or ():
            if tensor_key not in self._tensor_to_graph:
                self._tensor_to_graph[tensor_key] = graph_ref
        return graph_ref

    def graph_state(self, graph_key):
        return self._graph_refs[graph_key].get_state()

    def fetch_result(self, graph_key, tensor_key):
        # TODO just for test, should move to web handler
        graph_ref = self._graph_refs[graph_key]
        return graph_ref.fetch_tensor_result(tensor_key)

    @log_unhandled
    def handle_worker_change(self, adds, removes):
        """
        Receive changes in worker list, collect relevant data losses
        and notify graphs to handle these changes.

        :param adds: endpoints of workers newly added to the cluster
        :param removes: endpoints of workers removed to the cluster
        """
        logger.debug('Worker change detected. adds: %r, removes: %r', adds, removes)

        from .chunkmeta import LocalChunkMetaActor

        # collect affected chunks
        futures = []
        if removes:
            lost_chunks = set()
            for scheduler in self.get_schedulers():
                ref = self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=scheduler)
                futures.append(ref.remove_workers_in_session(self._session_id, removes, _wait=False))

            for f in futures:
                lost_chunks.update(f.result())
            lost_chunks = list(lost_chunks)
            logger.debug('Meta collection done, %d chunks lost.', len(lost_chunks))
        else:
            lost_chunks = []

        # notify every graph to handle failures
        futures = []
        for ref in self._graph_refs.values():
            futures.append(ref.handle_worker_change(adds, removes, lost_chunks, _wait=False, _tell=True))
        [f.result() for f in futures]


class SessionManagerActor(SchedulerActor):
    def __init__(self):
        super(SessionManagerActor, self).__init__()
        self._session_refs = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()

    def get_sessions(self):
        return self._session_refs

    @log_unhandled
    def create_session(self, session_id, **kwargs):
        uid = SessionActor.gen_uid(session_id)
        scheduler_address = self.get_scheduler(uid)
        session_ref = self.ctx.create_actor(SessionActor, uid=uid, address=scheduler_address,
                                            session_id=session_id, **kwargs)
        self._session_refs[session_id] = session_ref
        return session_ref

    @log_unhandled
    def delete_session(self, session_id):
        if session_id in self._session_refs:
            session_ref = self._session_refs[session_id]
            session_ref.destroy()
            del self._session_refs[session_id]

    @log_unhandled
    def broadcast_sessions(self, handler, *args, **kwargs):
        futures = []
        for ref in self._session_refs.values():
            kwargs.update(dict(_wait=False, _tell=True))
            futures.append(getattr(ref, handler)(*args, **kwargs))
        [f.result() for f in futures]
