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
from collections import namedtuple, defaultdict

from .kvstore import KVStoreActor
from .utils import SchedulerActor
from ..compat import OrderedDict3

logger = logging.getLogger(__name__)

WorkerMeta = namedtuple('WorkerMeta', 'chunk_size workers')
_META_CACHE_SIZE = 1000


class LocalChunkMetaActor(SchedulerActor):
    @classmethod
    def default_name(cls):
        return 's:' + cls.__name__

    def __init__(self, chunk_info_uid=None):
        super(LocalChunkMetaActor, self).__init__()
        self._meta_store = dict()
        self._meta_broadcasts = dict()
        self._meta_cache = OrderedDict3()
        self._chunk_info_uid = chunk_info_uid

        self._kv_store_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d at %s', self.uid, os.getpid(), self.address)
        super(LocalChunkMetaActor, self).post_create()

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    def set_chunk_broadcasts(self, session_id, chunk_key, broadcast_dests):
        self._meta_broadcasts[(session_id, chunk_key)] = broadcast_dests

    def set_chunk_meta(self, session_id, chunk_key, size=None, workers=None):
        query_key = (session_id, chunk_key)
        if query_key in self._meta_store:
            stored_size, stored_workers = self._meta_store[query_key]
            size = size if size is not None else stored_size
            if workers is not None:
                workers = set(tuple(workers) + stored_workers)
        if self._kv_store_ref is not None:
            path = '/sessions/%s/chunks/%s' % (session_id, chunk_key)
            if size is not None:
                self._kv_store_ref.write(path + '/data_size', size, _tell=True, _wait=False)
            if workers is not None:
                for w in workers:
                    self._kv_store_ref.write(path + '/workers/%s' % w, '', _tell=True, _wait=False)
        meta = self._meta_store[query_key] = WorkerMeta(size, tuple(workers or ()))
        futures = []
        if query_key in self._meta_broadcasts:
            for dest in self._meta_broadcasts[query_key]:
                if dest == self.address:
                    continue
                futures.append(self.ctx.actor_ref(self.default_name(), address=dest) \
                    .cache_chunk_meta(session_id, chunk_key, meta, _wait=False, _tell=True))
            [f.result() for f in futures]

    def cache_chunk_meta(self, session_id, chunk_key, meta):
        query_key = (session_id, chunk_key)
        if query_key in self._meta_cache:
            self._meta_cache[query_key] = meta
            self._meta_cache.move_to_end(query_key)
        else:
            self._meta_cache[query_key] = meta
            while len(self._meta_cache) > _META_CACHE_SIZE:
                self._meta_cache.popitem(False)

    def get_chunk_meta(self, session_id, chunk_key):
        query_key = (session_id, chunk_key)
        try:
            return self._meta_store[query_key]
        except KeyError:
            return self._meta_cache.get(query_key)

    def get_workers(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.workers if meta is not None else None

    def get_chunk_size(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.chunk_size if meta is not None else None

    def add_worker(self, session_id, chunk_key, worker_addr):
        self.set_chunk_meta(session_id, chunk_key, workers=(worker_addr,))

    def batch_get_chunk_meta(self, session_id, chunk_keys):
        return [self.get_chunk_meta(session_id, k) for k in chunk_keys]

    def delete_meta(self, session_id, chunk_key):
        query_key = (session_id, chunk_key)
        try:
            del self._meta_store[query_key]
        except KeyError:
            pass
        try:
            del self._meta_cache[query_key]
        except KeyError:
            pass
        if self._kv_store_ref is not None:
            self._kv_store_ref.delete('/sessions/%s/chunks/%s' % (session_id, chunk_key), recursive=True,
                                      _tell=True, _wait=False)

    def batch_delete_meta(self, session_id, chunk_keys):
        for chunk_key in chunk_keys:
            self.delete_meta(session_id, chunk_key)


class ChunkMetaActor(SchedulerActor):
    @classmethod
    def default_name(cls):
        return 's:' + cls.__name__

    def __init__(self):
        super(ChunkMetaActor, self).__init__()
        self._local_meta_store_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d at %s', self.uid, os.getpid(), self.address)
        super(ChunkMetaActor, self).post_create()

        self.set_cluster_info_ref()
        self._local_meta_store_ref = self.ctx.create_actor(
            LocalChunkMetaActor, uid=LocalChunkMetaActor.default_name())

    def set_chunk_broadcasts(self, session_id, chunk_key, broadcast_dests):
        addr = self.get_scheduler((session_id, chunk_key))
        self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .set_chunk_broadcasts(session_id, chunk_key, broadcast_dests, _tell=True, _wait=False)

    def set_chunk_meta(self, session_id, chunk_key, size=None, workers=None):
        addr = self.get_scheduler((session_id, chunk_key))
        self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .set_chunk_meta(session_id, chunk_key, size=size, workers=workers)

    def set_chunk_size(self, session_id, chunk_key, size):
        self.set_chunk_meta(session_id, chunk_key, size=size)

    def get_chunk_size(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.chunk_size if meta is not None else None

    def add_worker(self, session_id, chunk_key, worker_addr):
        self.set_chunk_meta(session_id, chunk_key, workers=(worker_addr,))

    def get_workers(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.workers if meta is not None else None

    def batch_get_workers(self, session_id, chunk_keys):
        metas = self.batch_get_chunk_meta(session_id, chunk_keys)
        return [meta.workers if meta is not None else None for meta in metas]

    def batch_get_chunk_size(self, session_id, chunk_keys):
        metas = self.batch_get_chunk_meta(session_id, chunk_keys)
        return [meta.chunk_size if meta is not None else None for meta in metas]

    def get_chunk_meta(self, session_id, chunk_key):
        local_result = self._local_meta_store_ref.get_chunk_meta(session_id, chunk_key)
        if local_result is not None:
            return local_result

        addr = self.get_scheduler((session_id, chunk_key))
        meta = self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .get_chunk_meta(session_id, chunk_key)
        return meta

    def batch_get_chunk_meta(self, session_id, chunk_keys):
        chunk_keys = tuple(chunk_keys)
        query_dict = defaultdict(set)
        meta_dict = dict()

        local_results = self._local_meta_store_ref.batch_get_chunk_meta(session_id, chunk_keys)

        for chunk_key, local_result in zip(chunk_keys, local_results):
            if local_result is not None:
                meta_dict[chunk_key] = local_result
            else:
                k = (session_id, chunk_key)
                query_dict[self.get_scheduler(k)].add(chunk_key)

        query_dict = dict((k, list(v)) for k, v in query_dict.items())

        futures = []
        for addr, keys in query_dict.items():
            futures.append(self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
                           .batch_get_chunk_meta(session_id, keys, _wait=False))
        for keys, future in zip(query_dict.values(), futures):
            results = future.result()
            meta_dict.update(zip(keys, results))
        return [meta_dict.get(k) for k in chunk_keys]

    def delete_meta(self, session_id, chunk_key):
        query_key = (session_id, chunk_key)
        addr = self.get_scheduler(query_key)
        self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .delete_meta(session_id, chunk_key, _tell=True)

    def batch_delete_meta(self, session_id, chunk_keys):
        chunk_keys = tuple(chunk_keys)
        query_dict = defaultdict(set)
        for chunk_key in chunk_keys:
            k = (session_id, chunk_key)
            query_dict[self.get_scheduler(k)].add(chunk_key)
        futures = []
        for addr, keys in query_dict.items():
            futures.append(self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
                           .batch_delete_meta(session_id, list(keys), _wait=False, _tell=True))
        [f.result() for f in futures]
