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

import copy
import os
import random
import string
import sys
import threading
from collections import namedtuple, defaultdict
from enum import Enum
from typing import List, Dict, Optional

import numpy as np


_context_factory = threading.local()


def get_context() -> "ContextBase":
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

    def get_current_session(self):
        """
        Get current session.

        :return: Session
        """
        raise NotImplementedError

    # ------------
    # Meta related
    # ------------

    def get_tileable_metas(self, tileable_keys, filter_fields: List[str] = None) -> List:
        """
        get tileable metas. Tileable includes tensor, DataFrame, mutable tensor and mutable DataFrame.
        :param tileable_keys: tileable keys
        :param filter_fields: filter the fields in meta
        :return: List of tileable metas
        """
        raise NotImplementedError

    def get_chunk_metas(self, chunk_keys, filter_fields=None):
        """
        Get chunk metas according to the given chunk keys.

        :param chunk_keys: chunk keys
        :param filter_fields: filter the fields in meta
        :return: List of chunk metas
        """
        raise NotImplementedError

    # ---------------
    # Cluster related
    # ---------------

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

    def get_worker_metas(self):  # pragma: no cover
        """
        Get worker metas

        :return: List of worker metas
        """
        raise NotImplementedError

    def get_total_ncores(self):
        """
        Get all workers' cores
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

    # -------------
    # Graph related
    # -------------

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

    # -----------------------
    # Pool occupation related
    # -----------------------

    def yield_execution_pool(self):
        """
        Yields current execution pool to allow running other tasks
        """
        pass

    def acquire_execution_pool(self, yield_info):
        pass

    # --------------
    # Result related
    # --------------

    def get_chunk_results(self, chunk_keys: List[str]) -> List:
        """
        Get results when given chunk keys

        :param chunk_keys: chunk keys
        :return: list of chunk results
        """
        raise NotImplementedError

    # ----------------------------------
    # Vineyard backend: query key mapper
    # ----------------------------------
    def get_vineyard_object_id(self, chunk_key):
        """
        Get the vineyard object ID for given chunk.

        :param chunk_key: chunk keys to query
        :return: vineyard object ID for the queried chunk.
        """
        raise NotImplementedError

    # ------
    # Others
    # ------

    def create_lock(self):
        raise NotImplementedError

    def get_named_tileable_infos(self, name: str):
        raise NotImplementedError


ChunkMeta = namedtuple('ChunkMeta', ['chunk_size', 'chunk_shape', 'workers'])
TileableInfos = namedtuple('TileableInfos', ['tileable_key', 'tileable_shape',
                                             'tileable_extra_meta'])


class LocalContext(ContextBase, dict):
    def __init__(self, local_session, ncores=None):
        dict.__init__(self)
        self._local_session = local_session
        self._ncores = ncores

    def copy(self):
        new_d = LocalContext(self._local_session, ncores=self._ncores)
        new_d.update(self)
        return new_d

    def get_current_session(self):
        from .session import new_session

        sess = new_session()
        sess._sess = copy.copy(self._local_session)
        sess._sess.context = self
        return sess

    def set_ncores(self, ncores):
        self._ncores = ncores

    @property
    def running_mode(self):
        return RunningMode.local

    @property
    def session_id(self):
        return self._local_session.session_id

    def get_scheduler_addresses(self):
        return

    def get_worker_addresses(self):  # pragma: no cover
        return

    def get_worker_metas(self):  # pragma: no cover
        return

    def get_total_ncores(self):
        return self._ncores

    def get_local_address(self):
        return

    def get_ncores(self):
        return self._ncores

    def get_chunk_metas(self, chunk_keys, filter_fields=None):
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

        selected_metas = []
        for chunk_meta in metas:
            if filter_fields is not None:
                chunk_meta = [getattr(chunk_meta, field) for field in filter_fields]
            selected_metas.append(chunk_meta)
        return selected_metas

    def get_chunk_results(self, chunk_keys: List[str]) -> List:
        # As the context is actually holding the data,
        # so for the local context, we just fetch data from itself
        return [self[chunk_key] for chunk_key in chunk_keys]

    def get_named_tileable_infos(self, name: str):
        if name not in self._local_session.executor._tileable_names:
            raise ValueError(f"Name {name} doesn't exist.")
        tileable = self._local_session.executor._tileable_names[name]
        extra_meta = tileable.params.copy()
        extra_meta.pop('shape', None)
        extra_meta = dict((k, v.to_pandas() if hasattr(v, 'to_pandas') else v)
                          for k, v in extra_meta.items())
        return TileableInfos(tileable.key, tileable.shape, extra_meta)

    def create_lock(self):
        return self._local_session.executor._sync_provider.lock()


class DistributedContext(ContextBase):
    def __init__(self, scheduler_address, session_id, actor_ctx=None,
                 is_distributed=None, resource_ref=None, **kw):
        from .worker.api import WorkerAPI
        from .scheduler.custom_log import CustomLogMetaActor
        from .scheduler.resource import ResourceActor
        from .scheduler.utils import SchedulerClusterInfoActor
        from .actors import new_client

        self._session_id = session_id
        self._scheduler_address = scheduler_address
        self._worker_api = WorkerAPI()
        self._meta_api_thread_local = threading.local()

        self._running_mode = None
        self._actor_ctx = actor_ctx or new_client()
        self._cluster_info = self._actor_ctx.actor_ref(
            SchedulerClusterInfoActor.default_uid(), address=scheduler_address)

        if is_distributed is None:
            is_distributed = self._cluster_info.is_distributed()
        self._running_mode = RunningMode.local_cluster \
            if not is_distributed else RunningMode.distributed

        if resource_ref is not None:
            self._resource_actor_ref = self._actor_ctx.actor_ref(resource_ref)
        else:
            self._resource_actor_ref = self._actor_ctx.actor_ref(
                ResourceActor.default_uid(),
                address=self._cluster_info.get_scheduler(ResourceActor.default_uid()))

        self._custom_log_meta = self._actor_ctx.actor_ref(
            CustomLogMetaActor.default_uid(),
            address=self._cluster_info.get_scheduler(CustomLogMetaActor.default_uid()))

        self._address = kw.pop('address', None)
        self._extra_info = kw

    @property
    def meta_api(self):
        from .scheduler.api import MetaAPI
        try:
            return self._meta_api_thread_local._meta_api
        except AttributeError:
            meta_api = self._meta_api_thread_local._meta_api \
                = MetaAPI(scheduler_endpoint=self._scheduler_address)
            return meta_api

    @property
    def running_mode(self):
        return self._running_mode

    @property
    def session_id(self):
        return self._session_id

    @property
    def scheduler_address(self):
        return self._scheduler_address

    def get_current_session(self):
        from .session import new_session, ClusterSession

        sess = new_session()
        sess._sess = ClusterSession(self._scheduler_address,
                                    session_id=self._session_id)
        return sess

    def get_scheduler_addresses(self):
        return self._cluster_info.get_schedulers()

    def get_worker_addresses(self):
        return self._resource_actor_ref.get_worker_endpoints()

    def get_worker_metas(self):  # pragma: no cover
        return self._resource_actor_ref.get_worker_metas()

    def get_total_ncores(self):
        return self._resource_actor_ref.get_total_cpu_count()

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
    def get_tileable_metas(self, tileable_keys, filter_fields: List[str] = None) -> List:
        return self.meta_api.get_tileable_metas(self._session_id, tileable_keys, filter_fields)

    def get_chunk_metas(self, chunk_keys, filter_fields: List[str] = None) -> List:
        return self.meta_api.get_chunk_metas(self._session_id, chunk_keys, filter_fields)

    def get_named_tileable_infos(self, name: str):
        tileable_key = self.meta_api.get_tileable_key_by_name(self._session_id, name)
        nsplits, extra_meta = self.get_tileable_metas([tileable_key],
                                                      filter_fields=['nsplits', 'extra_meta'])[0]
        shape = tuple(sum(s) for s in nsplits)
        return TileableInfos(tileable_key, shape, extra_meta)

    # Worker API
    def get_chunks_data(self, worker: str, chunk_keys: List[str], indexes: List = None,
                        compression_types: List[str] = None):
        return self._worker_api.get_chunks_data(self._session_id, worker, chunk_keys, indexes=indexes,
                                                compression_types=compression_types)

    # Fetch tileable data by tileable keys and indexes.
    def get_tileable_data(self, tileable_key: str, indexes: List = None,
                          compression_types: List[str] = None):
        from .serialize import dataserializer
        from .utils import merge_chunks
        from .tensor.datasource import empty
        from .tensor.indexing.index_lib import NDArrayIndexesHandler

        nsplits, chunk_keys, chunk_indexes = self.get_tileable_metas(
            [tileable_key], filter_fields=['nsplits', 'chunk_keys', 'chunk_indexes'])[0]
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

    def get_custom_log_meta_ref(self):
        from .actors import new_client
        from .scheduler.custom_log import CustomLogMetaActor
        from .scheduler.utils import SchedulerClusterInfoActor

        actor_client = new_client()
        uid = CustomLogMetaActor.default_uid()
        cluster_info = actor_client.actor_ref(
            SchedulerClusterInfoActor.default_uid(),
            address=self._scheduler_address)
        return actor_client.actor_ref(
            uid, address=cluster_info.get_scheduler(uid))

    def fetch_tileable_op_logs(self, tileable_op_key,
                               chunk_op_key_to_offsets=None,
                               chunk_op_key_to_sizes=None) -> Optional[Dict]:
        from .worker.dispatcher import DispatchActor
        from .utils import parse_readable_size

        chunk_op_key_to_paths = self._custom_log_meta.get_tileable_op_log_paths(
            self._session_id, tileable_op_key)

        if chunk_op_key_to_paths is None:
            return

        worker_to_kwds = dict()
        for chunk_op_key, path in chunk_op_key_to_paths.items():
            worker_address, log_path = path

            if isinstance(chunk_op_key_to_offsets, dict):
                offset = chunk_op_key_to_offsets.get(chunk_op_key, 0)
            elif isinstance(chunk_op_key_to_offsets, str):
                offset = int(parse_readable_size(chunk_op_key_to_offsets)[0])
            elif isinstance(chunk_op_key_to_offsets, int):
                offset = chunk_op_key_to_offsets
            else:
                offset = 0
            if isinstance(chunk_op_key_to_sizes, dict):
                size = chunk_op_key_to_sizes.get(chunk_op_key, -1)
            elif isinstance(chunk_op_key_to_sizes, str):
                size = int(parse_readable_size(chunk_op_key_to_sizes)[0])
            elif isinstance(chunk_op_key_to_sizes, int):
                size = chunk_op_key_to_sizes
            else:
                size = -1

            if worker_address not in worker_to_kwds:
                worker_to_kwds[worker_address] = {
                    'chunk_op_keys': [],
                    'log_paths': [],
                    'offsets': [],
                    'sizes': []
                }

            kwds = worker_to_kwds[worker_address]
            kwds['chunk_op_keys'].append(chunk_op_key)
            kwds['log_paths'].append(log_path)
            kwds['offsets'].append(offset)
            kwds['sizes'].append(size)

        result = dict()
        for worker, kwds in worker_to_kwds.items():
            dispatch = self._actor_ctx.actor_ref(DispatchActor.default_uid(),
                                                 address=worker)
            custom_log_fetcher = self._actor_ctx.actor_ref(
                worker, dispatch.get_hash_slot('custom_log', np.random.bytes(8)))
            chunk_op_keys = kwds.pop('chunk_op_keys')
            logs = custom_log_fetcher.fetch_logs(**kwds)
            for chunk_op_key, log_result in zip(chunk_op_keys, logs):
                result[chunk_op_key] = log_result

        return result

    def get_vineyard_object_id(self, chunk_key):
        from .config import options
        from .worker.storage.vineyardhandler import VineyardKeyMapActor

        if options.vineyard.enabled and self.running_mode != RunningMode.local:
            addr = self._cluster_info.get_scheduler((self._session_id, chunk_key))
            obj_id = self._actor_ctx.actor_ref(VineyardKeyMapActor.default_uid(), address=addr) \
                .get(self._session_id, chunk_key)
            return obj_id
        else:
            return None

    def create_lock(self):
        return self._actor_ctx.lock()


class DistributedDictContext(DistributedContext, dict):
    def __init__(self, *args, **kwargs):
        DistributedContext.__init__(self, *args, **kwargs)
        dict.__init__(self)

    def yield_execution_pool(self):
        actor_cls = self.get('_actor_cls')
        actor_uid = self.get('_actor_uid')
        op_key = self.get('_op_key')
        if not actor_cls or not actor_uid:  # pragma: no cover
            return

        from .actors import new_client
        from .actors.errors import ActorAlreadyExist
        from .worker.daemon import WorkerDaemonActor
        client = new_client()

        worker_addr = self.get_local_address()
        if client.has_actor(client.actor_ref(WorkerDaemonActor.default_uid(), address=worker_addr)):
            holder = client.actor_ref(WorkerDaemonActor.default_uid(), address=worker_addr)
        else:
            holder = client

        while True:
            try:
                random_tail = ''.join(random.choice(string.ascii_letters + string.digits)
                                      for _ in range(5))
                uid = f'w:0:mars-cpu-calc-backup-{os.getpid()}-{op_key}-{random_tail}'
                uid = self._actor_ctx.distributor.make_same_process(uid, actor_uid)
                ref = holder.create_actor(actor_cls, uid=uid, address=worker_addr)
                break
            except ActorAlreadyExist:  # pragma: no cover
                pass
        return ref

    def acquire_execution_pool(self, yield_info):
        if yield_info is None:
            return

        from .actors import new_client
        client = new_client()
        calc_ref = client.actor_ref(yield_info, address=self.get_local_address())
        calc_ref.mark_destroy()
