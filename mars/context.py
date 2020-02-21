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

import sys
import threading
import random
from collections import namedtuple, defaultdict
from enum import Enum
from typing import List


_context_factory = threading.local()


def get_context():
    return getattr(_context_factory, 'current', None)


class RunningMode(Enum):
    local = 0
    local_cluster = 1
    distributed = 2


class ContextBase(object):
    """
    Context will be used as a global object to detect the environment,
    and fetch meta, etc, mostly used in the server side, not for user.
    """

    @property
    def running_mode(self):
        """
        Get the running mode, could be local, local_cluster or distributed.
        """
        raise NotImplementedError

    @property
    def session_id(self):
        """
        Get session id.
        """
        raise NotImplementedError

    def __enter__(self):
        _context_factory.prev = getattr(_context_factory, 'current', None)
        _context_factory.current = self

    def __exit__(self, *_):
        _context_factory.current = _context_factory.prev
        _context_factory.prev = None

    def wraps(self, func):
        def h(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return h

    # ---------------
    # Meta relative
    # ---------------

    def get_chunk_metas(self, chunk_keys, filter_fields=None):
        """
        Get chunk metas according to the given chunk keys.

        :param chunk_keys: chunk keys
        :param filter_fields: filter the fields in meta
        :return: List of chunk metas
        """
        raise NotImplementedError

    # -----------------
    # Cluster relative
    # -----------------

    def get_scheduler_addresses(self):
        """
        Get scheduler addresses

        :return: List of scheduler addresses
        """
        raise NotImplementedError

    def get_worker_addresses(self):
        """
        Get worker addreses

        :return: List of worker addresses
        """
        raise NotImplementedError

    def get_local_address(self):
        """
        Get local address

        :return: local address
        """
        raise NotImplementedError

    def get_ncores(self):
        """
        Get number of cores
        """
        raise NotImplementedError

    # ---------------
    # Graph relative
    # ---------------

    def submit_chunk_graph(self, graph, result_keys):
        """
        Submit fine-grained graph to execute.

        :param graph: fine-grained graph to execute
        :param result_keys: result chunk keys
        :return: Future
        """
        raise NotImplementedError

    def submit_tileable_graph(self, graph, result_keys):
        """
        Submit coarse-grained graph to execute.

        :param graph: coarse-grained graph to execute
        :param result_keys: result tileable keys
        :return: Future
        """
        raise NotImplementedError

    # ----------------
    # Result relative
    # ----------------

    def get_chunk_results(self, chunk_keys: List[str]) -> List:
        """
        Get results when given chunk keys

        :param chunk_keys: chunk keys
        :return: list of chunk results
        """
        raise NotImplementedError

    # ------
    # Others
    # ------

    def create_lock(self):
        raise NotImplementedError


ChunkMeta = namedtuple('ChunkMeta', ['chunk_size', 'chunk_shape', 'workers'])


class LocalContext(ContextBase, dict):
    def __init__(self, local_session, ncores=None):
        dict.__init__(self)
        self._local_session = local_session
        self._ncores = ncores

    def copy(self):
        new_d = LocalContext(self._local_session, ncores=self._ncores)
        new_d.update(self)
        return new_d

    def set_ncores(self, ncores):
        self._ncores = ncores

    @property
    def running_mode(self):
        return RunningMode.local

    @property
    def session_id(self):
        return

    def get_scheduler_addresses(self):
        return

    def get_worker_addresses(self):
        return

    def get_local_address(self):
        return

    def get_ncores(self):
        return self._ncores

    def get_chunk_metas(self, chunk_keys, filter_fields=None):
        if filter_fields is not None:  # pragma: no cover
            raise NotImplementedError("Local context doesn't support filter fields now")
        metas = []
        for chunk_key in chunk_keys:
            chunk_data = self.get(chunk_key)
            if chunk_data is None:
                metas.append(None)
                continue
            if hasattr(chunk_data, 'nbytes'):
                # ndarray
                size = chunk_data.nbytes
                shape = chunk_data.shape
            elif hasattr(chunk_data, 'memory_usage'):
                # DataFrame
                size = chunk_data.memory_usage(deep=True).sum()
                shape = chunk_data.shape
            else:
                # other
                size = sys.getsizeof(chunk_data)
                shape = ()

            metas.append(ChunkMeta(chunk_size=size, chunk_shape=shape, workers=None))

        return metas

    def get_chunk_results(self, chunk_keys: List[str]) -> List:
        # As the context is actually holding the data,
        # so for the local context, we just fetch data from itself
        return [self[chunk_key] for chunk_key in chunk_keys]

    def create_lock(self):
        return self._local_session.executor._sync_provider.lock()


class DistributedContext(ContextBase):
    def __init__(self, scheduler_address, session_id, actor_ctx=None, **kw):
        from .worker.api import WorkerAPI
        from .scheduler.api import MetaAPI
        from .scheduler.resource import ResourceActor
        from .scheduler.utils import SchedulerClusterInfoActor
        from .actors import new_client

        self._session_id = session_id
        self._scheduler_address = scheduler_address
        self._worker_api = WorkerAPI()
        self._meta_api = MetaAPI(actor_ctx=actor_ctx, scheduler_endpoint=scheduler_address)

        self._running_mode = None
        self._actor_ctx = actor_ctx or new_client()
        self._cluster_info = self._actor_ctx.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_address)
        is_distributed = self._cluster_info.is_distributed()
        self._running_mode = RunningMode.local_cluster \
            if not is_distributed else RunningMode.distributed
        self._address = kw.pop('address', None)
        self._resource_actor_ref = self._actor_ctx.actor_ref(
            ResourceActor.default_uid(), address=scheduler_address)

        self._extra_info = kw

    @property
    def running_mode(self):
        return self._running_mode

    @property
    def session_id(self):
        return self._session_id

    def get_scheduler_addresses(self):
        return self._cluster_info.get_schedulers()

    def get_worker_addresses(self):
        return self._resource_actor_ref.get_worker_endpoints()

    def get_local_address(self):
        return self._address

    def get_ncores(self):
        return self._extra_info.get('n_cpu')

    def get_chunk_results(self, chunk_keys: List[str]) -> List:
        from .serialize import dataserializer
        from .worker.transfer import ResultSenderActor

        all_workers = [m.workers for m in self.get_chunk_metas(chunk_keys)]
        results = []
        for chunk_key, endpoints in zip(chunk_keys, all_workers):
            sender_ref = self._actor_ctx.actor_ref(
                ResultSenderActor.default_uid(), address=endpoints[-1])
            results.append(
                dataserializer.loads(sender_ref.fetch_data(self._session_id, chunk_key)))
        return results

    # Meta API
    def get_tileable_metas(self, tileable_keys, filter_fields: List[str]=None) -> List:
        return self._meta_api.get_tileable_metas(self._session_id, tileable_keys, filter_fields)

    def get_chunk_metas(self, chunk_keys, filter_fields: List[str] = None) -> List:
        return self._meta_api.get_chunk_metas(self._session_id, chunk_keys, filter_fields)

    def get_tileable_key_by_name(self, name: str):
        return self._meta_api.get_tileable_key_by_name(self._session_id, name)

    # Worker API
    def get_chunks_data(self, worker: str, chunk_keys: List[str], indexes: List=None,
                        compression_types: List[str]=None):
        return self._worker_api.get_chunks_data(self._session_id, worker, chunk_keys, indexes=indexes,
                                                compression_types=compression_types)

    # Fetch tileable data by tileable keys and indexes.
    def get_tileable_data(self, tileable_key: str, indexes: List = None,
                          compression_types: List[str] = None):
        from .serialize import dataserializer
        from .utils import merge_chunks
        from .tensor.datasource import empty
        from .tensor.indexing.index_lib import NDArrayIndexesHandler

        nsplits, chunk_keys, chunk_indexes = self.get_tileable_metas([tileable_key])[0]
        chunk_idx_to_keys = dict(zip(chunk_indexes, chunk_keys))
        chunk_keys_to_idx = dict(zip(chunk_keys, chunk_indexes))
        endpoints = self.get_chunk_metas(chunk_keys, filter_fields=['workers'])
        chunk_keys_to_worker = dict((chunk_key, random.choice(es[0])) for es, chunk_key in zip(endpoints, chunk_keys))

        chunk_workers = defaultdict(list)
        [chunk_workers[e].append(chunk_key) for chunk_key, e in chunk_keys_to_worker.items()]

        chunk_results = dict()
        if indexes is None or len(indexes) == 0:
            datas = []
            for endpoint, chunks in chunk_workers.items():
                datas.append(self.get_chunks_data(endpoint, chunks, compression_types=compression_types))
            datas = [d.result() for d in datas]
            for (endpoint, chunks), d in zip(chunk_workers.items(), datas):
                d = [dataserializer.loads(db) for db in d]
                chunk_results.update(dict(zip([chunk_keys_to_idx[k] for k in chunks], d)))

            chunk_results = [(k, v) for k, v in chunk_results.items()]
            if len(chunk_results) == 1:
                return chunk_results[0][1]
            else:
                return merge_chunks(chunk_results)
        else:
            # Reuse the getitem logic to get each chunk's indexes
            tileable_shape = tuple(sum(s) for s in nsplits)
            empty_tileable = empty(tileable_shape, chunk_size=nsplits)._inplace_tile()
            indexed = empty_tileable[tuple(indexes)]
            indexes_handler = NDArrayIndexesHandler()
            try:
                context = indexes_handler.handle(indexed.op, return_context=True)
            except TypeError:
                raise TypeError("Doesn't support indexing by tensors")

            result_chunks = dict()
            for c in context.processed_chunks:
                result_chunks[chunk_idx_to_keys[c.inputs[0].index]] = [c.index, c.op.indexes]

            chunk_datas = dict()
            for endpoint, chunks in chunk_workers.items():
                to_fetch_keys = []
                to_fetch_indexes = []
                to_fetch_idx = []
                for r_chunk, (chunk_index, index_obj) in result_chunks.items():
                    if r_chunk in chunks:
                        to_fetch_keys.append(r_chunk)
                        to_fetch_indexes.append(index_obj)
                        to_fetch_idx.append(chunk_index)
                if to_fetch_keys:
                    datas = self.get_chunks_data(endpoint, to_fetch_keys, indexes=to_fetch_indexes,
                                                 compression_types=compression_types)
                    chunk_datas[tuple(to_fetch_idx)] = datas
            chunk_datas = dict((k, v.result()) for k, v in chunk_datas.items())
            for idx, d in chunk_datas.items():
                d = [dataserializer.loads(db) for db in d]
                chunk_results.update(dict(zip(idx, d)))

            chunk_results = [(k, v) for k, v in chunk_results.items()]
            return indexes_handler.aggregate_result(context, chunk_results)

    def create_lock(self):
        return self._actor_ctx.lock()


class DistributedDictContext(DistributedContext, dict):
    def __init__(self, *args, **kwargs):
        DistributedContext.__init__(self, *args, **kwargs)
        dict.__init__(self)
