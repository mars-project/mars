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
from collections import defaultdict

from .kvstore import KVStoreActor
from .utils import SchedulerActor
from ..compat import PY27, OrderedDict3

logger = logging.getLogger(__name__)

_META_CACHE_SIZE = 1000


class WorkerMeta(object):
    __slots__ = 'chunk_size', 'chunk_shape', 'workers'

    def __init__(self, chunk_size=0, chunk_shape=None, workers=None):
        self.chunk_size = chunk_size
        self.chunk_shape = chunk_shape
        self.workers = workers or ()

    def __eq__(self, other):
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in self.__slots__)

    def __iter__(self):
        for slot in self.__slots__:
            yield getattr(self, slot)

    def __repr__(self):
        return 'WorkerMeta(%r, %r ,%r)' % (self.chunk_size, self.chunk_shape, self.workers)

    if PY27:
        def __getstate__(self):
            return tuple(getattr(self, s) for s in self.__slots__)

        def __setstate__(self, state):
            self.__init__(**dict(zip(self.__slots__, state)))


class ChunkMetaStore(object):
    """
    Storage of chunk meta, holding worker -> chunk relation as well
    """
    def __init__(self):
        self._chunk_metas = dict()
        self._worker_to_chunk_keys = defaultdict(set)

    def __contains__(self, chunk_key):
        return chunk_key in self._chunk_metas

    def __getitem__(self, chunk_key):
        return self._chunk_metas[chunk_key]

    def __setitem__(self, chunk_key, worker_meta):
        if chunk_key in self._chunk_metas:
            self._del_chunk_key_from_workers(chunk_key, self._chunk_metas[chunk_key].workers)

        self._chunk_metas[chunk_key] = worker_meta

        worker_chunks = self._worker_to_chunk_keys
        for w in worker_meta.workers:
            worker_chunks[w].add(chunk_key)

    def __delitem__(self, chunk_key):
        self._del_chunk_key_from_workers(chunk_key, self._chunk_metas[chunk_key].workers)
        del self._chunk_metas[chunk_key]

    def _del_chunk_key_from_workers(self, chunk_key, workers):
        """
        Delete chunk keys from worker
        """
        worker_to_chunks = self._worker_to_chunk_keys
        for w in workers:
            worker_to_chunks[w].remove(chunk_key)

    def get(self, chunk_key, default=None):
        return self._chunk_metas.get(chunk_key, default)

    def get_worker_chunk_keys(self, worker, default=None):
        """
        Get chunk keys held in a worker
        :param worker: worker endpoint
        :param default: default value
        """
        return self._worker_to_chunk_keys.get(worker, default)

    def remove_worker_keys(self, worker, filter_fun=None):
        """
        Remove a worker from storage and return keys of lost chunks
        :param worker: worker endpoint
        :param filter_fun: key filter
        :return: keys of lost chunks
        """
        if worker not in self._worker_to_chunk_keys:
            return []

        filter_fun = filter_fun or (lambda k: True)
        store = self._chunk_metas
        affected = []
        for ckey in tuple(self._worker_to_chunk_keys[worker]):
            if not filter_fun(ckey):
                continue
            self._del_chunk_key_from_workers(ckey, (worker,))
            meta = store[ckey]
            new_workers = meta.workers = tuple(w for w in meta.workers if w != worker)
            if not new_workers:
                affected.append(ckey)
        for ckey in affected:
            del self[ckey]
        if not self._worker_to_chunk_keys[worker]:
            del self._worker_to_chunk_keys[worker]
        return affected


class ChunkMetaCache(ChunkMetaStore):
    """
    Cache of chunk meta with an LRU
    """
    def __init__(self, limit=_META_CACHE_SIZE):
        super(ChunkMetaCache, self).__init__()
        self._chunk_metas = OrderedDict3()
        self._limit = limit

    def __getitem__(self, item):
        self._chunk_metas.move_to_end(item)
        return super(ChunkMetaCache, self).__getitem__(item)

    def get(self, chunk_key, default=None):
        try:
            self._chunk_metas.move_to_end(chunk_key)
        except KeyError:
            pass
        return super(ChunkMetaCache, self).get(chunk_key, default)

    def __setitem__(self, key, value):
        limit = self._limit
        store = self._chunk_metas
        if key in store:
            super(ChunkMetaCache, self).__setitem__(key, value)
            store.move_to_end(key)
        else:
            super(ChunkMetaCache, self).__setitem__(key, value)
            while len(store) > limit:
                dkey, ditem = store.popitem(False)
                self._del_chunk_key_from_workers(dkey, ditem.workers)


class LocalChunkMetaActor(SchedulerActor):
    """
    Actor storing chunk metas and chunk cache
    """
    @classmethod
    def default_name(cls):
        return 's:' + cls.__name__

    def __init__(self, chunk_info_uid=None):
        super(LocalChunkMetaActor, self).__init__()
        self._meta_store = ChunkMetaStore()
        self._meta_broadcasts = dict()
        self._meta_cache = ChunkMetaCache()
        self._chunk_info_uid = chunk_info_uid

        self._kv_store_ref = None

    def post_create(self):
        logger.debug('Actor %s running in process %d at %s', self.uid, os.getpid(), self.address)
        super(LocalChunkMetaActor, self).post_create()

        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())
        if not self.ctx.has_actor(self._kv_store_ref):
            self._kv_store_ref = None

    def set_chunk_broadcasts(self, session_id, chunk_key, broadcast_dests):
        """
        Configure broadcast destination addresses. After configuration,
        when the meta of the chunk updates, the update will be broadcast
        into the configured destinations to reduce RPC cost in future.
        :param session_id: session id
        :param chunk_key: chunk key
        :param broadcast_dests: addresses of broadcast destinations, in tuple
        """
        self._meta_broadcasts[(session_id, chunk_key)] = \
            [d for d in broadcast_dests if d != self.address]

    def get_chunk_broadcasts(self, session_id, chunk_key):
        """
        Get chunk broadcast addresses, for test only
        :param session_id: session id
        :param chunk_key: chunk key
        """
        return self._meta_broadcasts.get((session_id, chunk_key))

    def set_chunk_meta(self, session_id, chunk_key, size=None, shape=None, workers=None):
        """
        Update chunk meta in current storage
        :param session_id: session id
        :param chunk_key: chunk key
        :param size: size of the chunk
        :param shape: shape of the chunk
        :param workers: workers holding the chunk
        """
        query_key = (session_id, chunk_key)
        # update input with existing value
        if query_key in self._meta_store:
            stored_size, stored_shape, stored_workers = self._meta_store[query_key]
            size = size if size is not None else stored_size
            shape = shape if shape is not None else stored_shape
            if workers is not None:
                workers = set(tuple(workers) + stored_workers)

        # sync to external kv store
        if self._kv_store_ref is not None:
            path = '/sessions/%s/chunks/%s' % (session_id, chunk_key)
            if size is not None:
                self._kv_store_ref.write(path + '/data_size', size, _tell=True, _wait=False)
            if shape is not None:
                self._kv_store_ref.write(path + '/data_shape', shape, _tell=True, _wait=False)
            if workers is not None:
                for w in workers:
                    self._kv_store_ref.write(path + '/workers/%s' % w, '', _tell=True, _wait=False)

        meta = self._meta_store[query_key] = WorkerMeta(size, shape, tuple(workers or ()))

        # broadcast into pre-determined destinations
        futures = []
        if query_key in self._meta_broadcasts:
            for dest in self._meta_broadcasts[query_key]:
                futures.append(
                    self.ctx.actor_ref(self.default_name(), address=dest)
                        .cache_chunk_meta(session_id, chunk_key, meta, _wait=False, _tell=True)
                )
            [f.result() for f in futures]

    def cache_chunk_meta(self, session_id, chunk_key, meta):
        """
        Receive updates for caching

        :param session_id: session id
        :param chunk_key: chunk key
        :param meta: meta data
        """
        query_key = (session_id, chunk_key)
        self._meta_cache[query_key] = meta

    def get_chunk_meta(self, session_id, chunk_key):
        """
        Obtain metadata. If not exists, None will be returned.
        """
        query_key = (session_id, chunk_key)
        try:
            return self._meta_store[query_key]
        except KeyError:
            return self._meta_cache.get(query_key)

    def batch_get_chunk_meta(self, session_id, chunk_keys):
        """
        Obtain metadata in batch
        """
        return [self.get_chunk_meta(session_id, k) for k in chunk_keys]

    def delete_meta(self, session_id, chunk_key):
        """
        Delete metadata from store and cache
        """
        query_key = (session_id, chunk_key)
        try:
            del self._meta_store[query_key]
            if self._kv_store_ref is not None:
                self._kv_store_ref.delete('/sessions/%s/chunks/%s' % (session_id, chunk_key),
                                          recursive=True, _tell=True, _wait=False)
        except KeyError:
            pass
        try:
            del self._meta_cache[query_key]
        except KeyError:
            pass

        # broadcast deletion into pre-determined destinations
        if query_key in self._meta_broadcasts:
            for dest in self._meta_broadcasts[query_key]:
                self.ctx.actor_ref(self.default_name(), address=dest) \
                    .delete_meta(session_id, chunk_key, _wait=False, _tell=True)
            del self._meta_broadcasts[query_key]

    def batch_delete_meta(self, session_id, chunk_keys):
        """
        Delete metadata in batch from store and cache
        """
        for chunk_key in chunk_keys:
            self.delete_meta(session_id, chunk_key)

    def remove_workers_in_session(self, session_id, workers):
        """
        Remove workers from storage given session id and return keys of lost chunks
        :param session_id: session id
        :param workers: worker endpoints
        :return: keys of lost chunks
        """
        logger.debug('Removing workers %r from store', workers)
        removed_chunks = set()
        for w in workers:
            self._meta_cache.remove_worker_keys(w, lambda k: k[0] == session_id)
            removed_chunks.update(self._meta_store.remove_worker_keys(w, lambda k: k[0] == session_id))
        for c in removed_chunks:
            try:
                del self._meta_broadcasts[c]
            except KeyError:
                pass
        return [k[1] for k in removed_chunks]


class ChunkMetaActor(SchedulerActor):
    """
    Actor dispatches chunk meta requests to different scheduler hosts
    """
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
        """
        Update metadata broadcast destinations for chunks. After configuration,
        when the meta of the chunk updates, the update will be broadcast
        into the configured destinations to reduce RPC cost in future.
        :param session_id: session id
        :param chunk_key: chunk key
        :param broadcast_dests: destination addresses for broadcast
        """
        addr = self.get_scheduler((session_id, chunk_key))
        self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .set_chunk_broadcasts(session_id, chunk_key, broadcast_dests, _tell=True, _wait=False)

    def set_chunk_meta(self, session_id, chunk_key, size=None, shape=None, workers=None):
        """
        Update chunk metadata
        :param session_id: session id
        :param chunk_key: chunk key
        :param size: size of the chunk
        :param shape: shape of the chunk
        :param workers: workers holding the chunk
        """
        addr = self.get_scheduler((session_id, chunk_key))
        self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .set_chunk_meta(session_id, chunk_key, size=size, shape=shape, workers=workers)

    def set_chunk_size(self, session_id, chunk_key, size):
        self.set_chunk_meta(session_id, chunk_key, size=size)

    def get_chunk_size(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.chunk_size if meta is not None else None

    def set_chunk_shape(self, session_id, chunk_key, shape):
        self.set_chunk_meta(session_id, chunk_key, shape=shape)

    def get_chunk_shape(self, session_id, chunk_key):
        meta = self.get_chunk_meta(session_id, chunk_key)
        return meta.chunk_shape if meta is not None else None

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

    def batch_get_chunk_shape(self, session_id, chunk_keys):
        metas = self.batch_get_chunk_meta(session_id, chunk_keys)
        return [meta.chunk_shape if meta is not None else None for meta in metas]

    def get_chunk_meta(self, session_id, chunk_key):
        """
        Obtain chunk metadata
        :param session_id: session id
        :param chunk_key: chunk key
        """
        local_result = self._local_meta_store_ref.get_chunk_meta(session_id, chunk_key)
        if local_result is not None:
            return local_result

        addr = self.get_scheduler((session_id, chunk_key))
        meta = self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr) \
            .get_chunk_meta(session_id, chunk_key)
        return meta

    def batch_get_chunk_meta(self, session_id, chunk_keys):
        """
        Obtain chunk metadata in batch
        :param session_id: session id
        :param chunk_keys: chunk keys
        """
        chunk_keys = tuple(chunk_keys)
        query_dict = defaultdict(set)
        meta_dict = dict()

        # try obtaining metadata from local cache
        local_results = self._local_meta_store_ref.batch_get_chunk_meta(session_id, chunk_keys)

        # collect dispatch destinations for non-local metadata
        for chunk_key, local_result in zip(chunk_keys, local_results):
            if local_result is not None:
                meta_dict[chunk_key] = local_result
            else:
                k = (session_id, chunk_key)
                query_dict[self.get_scheduler(k)].add(chunk_key)

        query_dict = dict((k, list(v)) for k, v in query_dict.items())

        # dispatch query
        futures = []
        for addr, keys in query_dict.items():
            futures.append(
                self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr)
                    .batch_get_chunk_meta(session_id, keys, _wait=False)
            )
        # accept results and merge
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
        """
        Delete chunk metadata in batch
        :param session_id: session id
        :param chunk_keys: chunk keys
        """
        # collect dispatch destinations
        chunk_keys = tuple(chunk_keys)
        query_dict = defaultdict(set)
        for chunk_key in chunk_keys:
            k = (session_id, chunk_key)
            query_dict[self.get_scheduler(k)].add(chunk_key)
        # dispatch delete requests and wait
        futures = []
        for addr, keys in query_dict.items():
            futures.append(
                self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr)
                    .batch_delete_meta(session_id, list(keys), _wait=False, _tell=True)
            )
        [f.result() for f in futures]
