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

import threading
from collections import namedtuple
import sys

from .compat import Enum


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

    def get_chunk_metas(self, chunk_keys):
        """
        Get chunk metas according to the given chunk keys.

        :param chunk_keys: chunk keys
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

    def get_chunk_results(self, chunk_keys):
        """
        Get results when given chunk keys

        :param chunk_keys: chunk keys
        :return: list of chunk results
        """
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

    def get_chunk_metas(self, chunk_keys):
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

    def get_chunk_results(self, chunk_keys):
        # As the context is actually holding the data,
        # so for the local context, we just fetch data from itself
        return [self[chunk_key] for chunk_key in chunk_keys]


class DistributedContext(ContextBase):
    def __init__(self, cluster_info, session_id, addr, chunk_meta_client,
                 resource_actor_ref, actor_ctx, **kw):
        self._cluster_info = cluster_info
        is_distributed = cluster_info.is_distributed()
        self._running_mode = RunningMode.local_cluster \
            if not is_distributed else RunningMode.distributed
        self._session_id = session_id
        self._address = addr
        self._chunk_meta_client = chunk_meta_client
        self._resource_actor_ref = resource_actor_ref
        self._actor_ctx = actor_ctx
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

    def get_chunk_metas(self, chunk_keys):
        return self._chunk_meta_client.batch_get_chunk_meta(
            self._session_id, chunk_keys)

    def get_chunk_results(self, chunk_keys):
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


class DistributedDictContext(DistributedContext, dict):
    def __init__(self, *args, **kwargs):
        DistributedContext.__init__(self, *args, **kwargs)
        dict.__init__(self)
