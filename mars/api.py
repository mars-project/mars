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


class MarsAPI:
    def __init__(self, scheduler_ip):
        self.__schedulers_cache = None
        self._session_manager = None
        self.actor_client = new_client()
        self.cluster_info = self.actor_client.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_ip)
        self.chunk_meta_client = ChunkMetaClient(self.actor_client, self.cluster_info)

    async def get_session_manager(self):
        if self._session_manager is None:
            self._session_manager = await self.get_actor_ref(SessionManagerActor.default_uid())
        return self._session_manager

    async def get_schedulers(self):
        if not self.__schedulers_cache:
            self.__schedulers_cache = await self.cluster_info.get_schedulers()
        return self.__schedulers_cache

    async def get_scheduler(self, uid):
        schedulers = await self.get_schedulers()
        if len(schedulers) == 1:
            return schedulers[0]
        else:
            return await self.cluster_info.get_scheduler(uid)

    async def get_actor_ref(self, uid):
        return self.actor_client.actor_ref(uid, address=await self.get_scheduler(uid))

    async def get_graph_meta_ref(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_meta_uid = GraphMetaActor.gen_uid(session_id, graph_key)
        graph_addr = await self.get_scheduler(graph_uid)
        return self.actor_client.actor_ref(graph_meta_uid, address=graph_addr)

    async def get_schedulers_info(self):
        schedulers = await self.get_schedulers()
        infos = dict()
        for scheduler in schedulers:
            info_ref = self.actor_client.actor_ref(NodeInfoActor.default_uid(), address=scheduler)
            infos[scheduler] = await info_ref.get_info()
        return infos

    async def count_workers(self):
        try:
            uid = ResourceActor.default_uid()
            return await (await self.get_actor_ref(uid)).get_worker_count()
        except KeyError:
            return 0

    async def create_session(self, session_id, **kw):
        session_manager = await self.get_session_manager()
        await session_manager.create_session(session_id, **kw)

    async def delete_session(self, session_id):
        session_manager = await self.get_session_manager()
        await session_manager.delete_session(session_id)

    async def has_session(self, session_id):
        """
        Check if the session with given session_id exists.
        """
        session_manager = await self.get_session_manager()
        return session_manager.has_session(session_id)

    async def submit_graph(self, session_id, serialized_graph, graph_key, target,
                           names=None, compose=True, wait=True):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = await self.get_actor_ref(session_uid)
        await session_ref.submit_tileable_graph(
            serialized_graph, graph_key, target, names=names, compose=compose, _tell=not wait)

    async def create_mutable_tensor(self, session_id, name, shape, dtype, *args, **kwargs):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = await self.get_actor_ref(session_uid)
        return await session_ref.create_mutable_tensor(name, shape, dtype, *args, **kwargs)

    async def get_mutable_tensor(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = await self.get_actor_ref(session_uid)
        return await session_ref.get_mutable_tensor(name)

    async def send_chunk_records(self, session_id, name, chunk_records_to_send, directly=True):
        from .worker.quota import MemQuotaActor
        from .worker.transfer import put_remote_chunk
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = await self.get_actor_ref(session_uid)

        chunk_records = []
        for chunk_key, endpoint, records in chunk_records_to_send:
            record_chunk_key = tokenize(chunk_key, uuid.uuid4().hex)
            # register quota
            quota_ref = self.actor_client.actor_ref(MemQuotaActor.default_uid(), address=endpoint)
            await quota_ref.request_batch_quota({record_chunk_key: records.nbytes})
            # send record chunk
            receiver_manager_ref = self.actor_client.actor_ref(
                ReceiverManagerActor.default_uid(), address=endpoint)
            await put_remote_chunk(session_id, record_chunk_key, records, receiver_manager_ref)
            chunk_records.append((chunk_key, record_chunk_key))

        # register the record chunk to MutableTensorActor
        await session_ref.append_chunk_records(name, chunk_records)

    async def seal(self, session_id, name):
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = await self.get_actor_ref(session_uid)
        return await session_ref.seal(name)

    async def delete_graph(self, session_id, graph_key):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = await self.get_actor_ref(graph_uid)
        await graph_ref.destroy()

    async def stop_graph(self, session_id, graph_key):
        from .scheduler import GraphState
        graph_meta_ref = await self.get_graph_meta_ref(session_id, graph_key)
        await graph_meta_ref.set_state(GraphState.CANCELLING)

        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = await self.get_actor_ref(graph_uid)
        await graph_ref.stop_graph()

    async def get_graph_state(self, session_id, graph_key):
        from .scheduler import GraphState

        graph_meta_ref = await self.get_graph_meta_ref(session_id, graph_key)
        try:
            state_obj = await graph_meta_ref.get_state()
            state = state_obj.value if state_obj else 'preparing'
        except ActorNotExist:
            raise GraphNotExists
        return GraphState(state.lower())

    async def get_graph_exc_info(self, session_id, graph_key):
        graph_meta_ref = await self.get_graph_meta_ref(session_id, graph_key)
        try:
            return await graph_meta_ref.get_exc_info()
        except ActorNotExist:
            raise GraphNotExists

    async def wait_graph_finish(self, session_id, graph_key, timeout=None):
        graph_meta_ref = await self.get_graph_meta_ref(session_id, graph_key)
        await self.actor_client.actor_ref(await graph_meta_ref.get_wait_ref()).wait(timeout)

    async def fetch_data(self, session_id, graph_key, tileable_key, index_obj=None, compressions=None):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = await self.get_actor_ref(graph_uid)
        nsplits, chunk_keys, chunk_indexes = (await graph_ref.get_tileable_metas([tileable_key]))[0]
        chunk_index_to_key = dict((index, key) for index, key in zip(chunk_indexes, chunk_keys))
        if not index_obj:
            chunk_results = dict([(idx, await self.fetch_chunk_data(session_id, k, wait=False)) for
                                  idx, k in zip(chunk_indexes, chunk_keys)])
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
                chunk_results[chunk_index] = await self.fetch_chunk_data(
                    session_id, chunk_key, slice_obj, wait=False)

        chunk_results = [(idx, dataserializer.loads(await f)) for
                         idx, f in chunk_results.items()]
        if len(chunk_results) == 1:
            ret = chunk_results[0][1]
        else:
            ret = merge_chunks(chunk_results)
        compressions = max(compressions) if compressions else dataserializer.CompressType.NONE
        return dataserializer.dumps(ret, compress=compressions)

    async def fetch_chunk_data(self, session_id, chunk_key, index_obj=None, wait=True):
        endpoints = await self.chunk_meta_client.get_workers(session_id, chunk_key)
        sender_ref = self.actor_client.actor_ref(ResultSenderActor.default_uid(),
                                                 address=random.choice(endpoints))
        if wait:
            return await sender_ref.fetch_data(session_id, chunk_key, index_obj)
        else:
            return sender_ref.fetch_data(session_id, chunk_key, index_obj, _wait=False)

    async def delete_data(self, session_id, graph_key, tileable_key, wait=False):
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = await self.get_actor_ref(graph_uid)
        await graph_ref.free_tileable_data(tileable_key, wait=wait, _tell=not wait)

    async def get_tileable_nsplits(self, session_id, graph_key, tileable_key):
        # nsplits is essential for operator like `reshape` and shape can be calculated by nsplits
        graph_uid = GraphActor.gen_uid(session_id, graph_key)
        graph_ref = await self.get_actor_ref(graph_uid)

        return (await graph_ref.get_tileable_metas([tileable_key], filter_fields=['nsplits']))[0][0]
