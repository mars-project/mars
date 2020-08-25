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

import logging
import os
import uuid

from .utils import SchedulerActor
from ..utils import log_unhandled

logger = logging.getLogger(__name__)


class SessionActor(SchedulerActor):
    def __init__(self, session_id, **kwargs):
        super().__init__()
        self._session_id = session_id
        self._args = kwargs

        self._tileable_names = dict()
        self._graph_infos = dict()

        self._cluster_info_ref = None
        self._assigner_ref = None
        self._manager_ref = None
        self._graph_refs = dict()
        self._graph_meta_refs = dict()
        self._tileable_to_graph = dict()
        self._mut_tensor_refs = dict()

    @staticmethod
    def gen_uid(session_id):
        return f's:h1:session${session_id}'

    def get_argument(self, key):
        return self._args[key]

    def get_tileable_key(self, name):
        return self._tileable_names[name]

    def get_graph_refs(self):
        return self._graph_refs

    def get_graph_meta_refs(self):
        return self._graph_meta_refs

    def get_graph_ref_by_tileable_key(self, tileable_key):
        return self._tileable_to_graph[tileable_key]

    def post_create(self):
        super().post_create()
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        self.set_cluster_info_ref()
        self._manager_ref = self.ctx.actor_ref(SessionManagerActor.default_uid())

        from .assigner import AssignerActor
        assigner_uid = AssignerActor.gen_uid(self._session_id)
        address = self.get_scheduler(assigner_uid)
        self._assigner_ref = self.ctx.create_actor(AssignerActor, uid=assigner_uid, address=address)

    def pre_destroy(self):
        super().pre_destroy()
        self.unset_cluster_info_ref()

        self._manager_ref.delete_session(self._session_id, _tell=True)
        self.ctx.destroy_actor(self._assigner_ref)
        for graph_ref in self._graph_refs.values():
            self.ctx.destroy_actor(graph_ref)
        for mut_tensor_ref in self._mut_tensor_refs.values():
            self.ctx.destroy_actor(mut_tensor_ref)

    def set_graph_info(self, graph_key, **kwargs):
        try:
            info_dict = self._graph_infos[graph_key]
        except KeyError:
            info_dict = self._graph_infos[graph_key] = dict()
        info_dict.update(kwargs)

    def get_graph_infos(self):
        return self._graph_infos

    @log_unhandled
    def submit_tileable_graph(self, serialized_graph, graph_key, target_tileables=None, names=None, compose=True):
        from .graph import GraphActor, GraphMetaActor

        graph_uid = GraphActor.gen_uid(self._session_id, graph_key)

        graph_addr = self.get_scheduler(graph_uid)
        graph_ref = self.ctx.create_actor(GraphActor, self._session_id, graph_key,
                                          serialized_graph, target_tileables=target_tileables,
                                          uid=graph_uid, address=graph_addr)
        self._graph_refs[graph_key] = graph_ref
        self._graph_meta_refs[graph_key] = self.ctx.actor_ref(
            GraphMetaActor.gen_uid(self._session_id, graph_key), address=graph_addr)

        graph_ref.execute_graph(_tell=True, compose=compose)

        for tileable_key in target_tileables or ():
            if tileable_key not in self._tileable_to_graph:
                self._tileable_to_graph[tileable_key] = graph_ref

        if names:
            self._tileable_names.update(zip(names, target_tileables))
        return graph_ref

    @log_unhandled
    def create_mutable_tensor(self, name, shape, dtype, *args, **kwargs):
        from .mutable import MutableTensorActor
        if name in self._mut_tensor_refs:
            raise ValueError(f"The mutable tensor named '{name}' already exists.")
        graph_key = uuid.uuid4()
        mut_tensor_uid = MutableTensorActor.gen_uid(self._session_id, name)
        mut_tensor_addr = self.get_scheduler(mut_tensor_uid)
        mut_tensor_ref = self.ctx.create_actor(MutableTensorActor, self._session_id, name,
                                               shape, dtype, graph_key, uid=mut_tensor_uid,
                                               address=mut_tensor_addr, *args, **kwargs)
        self._mut_tensor_refs[name] = mut_tensor_ref
        return mut_tensor_ref.tensor_meta()

    @log_unhandled
    def get_mutable_tensor(self, name):
        tensor_ref = self._mut_tensor_refs.get(name)
        if tensor_ref is None or tensor_ref.sealed():
            raise ValueError(f"The mutable tensor named '{name}' doesn't exist, or has already been sealed.")
        return tensor_ref.tensor_meta()

    @log_unhandled
    def write_mutable_tensor(self, name, index, value):
        tensor_ref = self._mut_tensor_refs.get(name)
        if tensor_ref is None or tensor_ref.sealed():
            raise ValueError(f"The mutable tensor named '{name}' doesn't exist, or has already been sealed.")
        return tensor_ref.write(index, value)

    @log_unhandled
    def append_chunk_records(self, name, chunk_records):
        tensor_ref = self._mut_tensor_refs.get(name)
        if tensor_ref is None or tensor_ref.sealed():
            raise ValueError(f"The mutable tensor named '{name}' doesn't exist, or has already been sealed.")
        return tensor_ref.append_chunk_records(chunk_records)

    @log_unhandled
    def seal(self, name):
        from .graph import GraphActor, GraphMetaActor
        from .utils import GraphState

        tensor_ref = self._mut_tensor_refs.get(name)
        if tensor_ref is None or tensor_ref.sealed():
            raise ValueError(f"The mutable tensor named '{name}' doesn't exist, or has already been sealed.")

        graph_key_hex, tensor_key, tensor_id, tensor_meta = tensor_ref.seal()
        shape, dtype, chunk_size, chunk_keys, _ = tensor_meta
        graph_key = uuid.UUID(graph_key_hex)

        # Create a GraphActor
        graph_uid = GraphActor.gen_uid(self._session_id, graph_key)
        graph_addr = self.get_scheduler(graph_uid)

        graph_ref = self.ctx.create_actor(GraphActor, self._session_id, graph_key,
                                          serialized_tileable_graph=None,
                                          state=GraphState.SUCCEEDED, final_state=GraphState.SUCCEEDED,
                                          uid=graph_uid, address=graph_addr)
        self._graph_refs[graph_key] = graph_ref
        self._graph_meta_refs[graph_key] = self.ctx.actor_ref(
            GraphMetaActor.gen_uid(self._session_id, graph_key), address=tensor_ref.__getstate__()[0])

        # Add the tensor to the GraphActor
        graph_ref.add_fetch_tileable(tensor_key, tensor_id, shape, dtype, chunk_size, chunk_keys)
        self._tileable_to_graph[tensor_key] = graph_ref

        # Clean up mutable tensor refs.
        actor_ref = self._mut_tensor_refs[name]
        actor_ref.destroy()
        del self._mut_tensor_refs[name]
        return graph_key_hex, tensor_key, tensor_id, tensor_meta

    def graph_state(self, graph_key):
        return self._graph_refs[graph_key].get_state()

    def fetch_result(self, graph_key, tileable_key, check=False):
        # TODO just for test, should move to web handler
        graph_ref = self._graph_refs[graph_key]
        return graph_ref.fetch_tileable_result(tileable_key, _check=check)

    @log_unhandled
    def handle_worker_change(self, adds, removes):
        """
        Receive changes in worker list, collect relevant data losses
        and notify graphs to handle these changes.

        :param adds: endpoints of workers newly added to the cluster
        :param removes: endpoints of workers removed to the cluster
        """
        logger.debug('Worker change detected. adds: %r, removes: %r', adds, removes)

        from .chunkmeta import ChunkMetaActor

        # collect affected chunks
        if removes:
            lost_chunks = set()
            futures = []
            for scheduler in self.get_schedulers():
                ref = self.ctx.actor_ref(ChunkMetaActor.default_uid(), address=scheduler)
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
        super().__init__()
        self._session_refs = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()

    def pre_destroy(self):
        self.unset_cluster_info_ref()

    def get_sessions(self):
        return list(self._session_refs.keys())

    def get_session_refs(self):
        return self._session_refs

    @log_unhandled
    def create_session(self, session_id, **kwargs):
        uid = SessionActor.gen_uid(session_id)
        scheduler_address = self.get_scheduler(uid)
        if session_id in self._session_refs:
            return self._session_refs[session_id]
        else:
            session_ref = self.ctx.create_actor(SessionActor, uid=uid, address=scheduler_address,
                                                session_id=session_id, **kwargs)
            self._session_refs[session_id] = session_ref
            return session_ref

    @log_unhandled
    def has_session(self, session_id):
        return session_id in self._session_refs

    @log_unhandled
    def delete_session(self, session_id):
        if session_id in self._session_refs:
            session_ref = self._session_refs[session_id]
            session_ref.destroy()
            del self._session_refs[session_id]

    @log_unhandled
    def broadcast_sessions(self, handler, *args, **kwargs):
        from .assigner import AssignerActor
        futures = []
        for key in self._session_refs.keys():
            ref = self.get_actor_ref(AssignerActor.gen_uid(key))
            futures.append(ref.mark_metrics_expired(_tell=True, _wait=False))
        [f.result() for f in futures]
        for f in futures:
            f.result()

        futures = []
        for ref in self._session_refs.values():
            kwargs.update(dict(_wait=False, _tell=True))
            futures.append(getattr(ref, handler)(*args, **kwargs))
        [f.result() for f in futures]
