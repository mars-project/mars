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
import math
import os
import time
from collections import OrderedDict

from ..actors import ActorNotExist
from ..cluster_info import ClusterInfoActor, HasClusterInfoActor
from ..compat import OrderedDict3
from ..config import options
from ..errors import WorkerProcessStopped
from ..promise import PromiseActor
from ..utils import build_exc_info

logger = logging.getLogger(__name__)


class WorkerClusterInfoActor(ClusterInfoActor):
    @classmethod
    def default_uid(cls):
        return 'w:0:%s' % cls.__name__


class WorkerHasClusterInfoActor(HasClusterInfoActor):
    cluster_info_uid = WorkerClusterInfoActor.default_uid()


class WorkerActor(WorkerHasClusterInfoActor, PromiseActor):
    """
    Base class of all worker actors, providing necessary utils
    """
    @classmethod
    def default_uid(cls):
        return 'w:0:{0}'.format(cls.__name__)

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        try:
            self.set_cluster_info_ref()
        except ActorNotExist:
            pass
        self._init_chunk_store()

    def _init_chunk_store(self):
        import pyarrow.plasma as plasma
        from .chunkstore import PlasmaChunkStore, PlasmaKeyMapActor

        mapper_ref = self.ctx.actor_ref(uid=PlasmaKeyMapActor.default_uid())
        self._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        self._chunk_store = PlasmaChunkStore(self._plasma_client, mapper_ref)

    def get_meta_client(self):
        from ..scheduler.chunkmeta import ChunkMetaClient
        return ChunkMetaClient(self.ctx, self._cluster_info_ref)

    def handle_actors_down(self, halt_refs):
        """
        Handle process down event
        :param halt_refs: actor refs in halt processes
        """
        handled_refs = self.reject_promise_refs(halt_refs, *build_exc_info(WorkerProcessStopped))
        logger.debug('Process halt detected. Affected promises %r rejected.',
                     [ref.uid for ref in handled_refs])

    def register_actors_down_handler(self):
        from .daemon import WorkerDaemonActor

        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.register_callback(self.ref(), self.handle_actors_down.__name__, _tell=True)


class ExpMeanHolder(object):
    """
    Collector of statistics of a given series. The value decays by _factor as time elapses.
    """
    def __init__(self, factor=0.8):
        self._factor = factor
        self._count = 0
        self._v_divided = 0
        self._v_divisor = 0
        self._v2_divided = 0

    def put(self, value):
        self._count += 1
        self._v_divided = self._v_divided * self._factor + value
        self._v_divisor = self._v_divisor * self._factor + 1
        self._v2_divided = self._v2_divided * self._factor + value ** 2

    def count(self):
        return self._count

    def mean(self):
        if self._count == 0:
            return 0
        return self._v_divided * 1.0 / self._v_divisor

    def var(self):
        if self._count == 0:
            return 0
        return self._v2_divided * 1.0 / self._v_divisor - self.mean() ** 2

    def std(self):
        return math.sqrt(self.var())


class ExpiringCache(dict):
    def __init__(self, *args, **kwargs):
        expire_time = kwargs.pop('_expire_time', options.worker.callback_preserve_time)
        super(ExpiringCache, self).__init__(*args, **kwargs)

        self._expire_time = expire_time
        self._insert_times = OrderedDict3()

    def __setitem__(self, key, value):
        super(ExpiringCache, self).__setitem__(key, value)
        if key in self._insert_times:
            self._insert_times[key] = time.time()
            self._insert_times.move_to_end(key)
            return

        clean_keys = []
        self._insert_times[key] = time.time()
        last_finish_time = time.time() - self._expire_time
        for k, t in self._insert_times.items():
            if t < last_finish_time:
                clean_keys.append(k)
            else:
                break
        for k in clean_keys:
            del self[k]


def concat_operand_keys(graph, sep=','):
    from ..operands import Fetch
    graph_op_dict = OrderedDict()
    for c in graph:
        if isinstance(c.op, Fetch):
            continue
        graph_op_dict[c.op.key] = type(c.op).__name__
    keys = sep.join(graph_op_dict.keys())
    graph_ops = sep.join(graph_op_dict.values())
    return keys, graph_ops


def get_chunk_key(key):
    return key[0] if isinstance(key, tuple) else key


def build_load_key(graph_key, chunk_key):
    if isinstance(chunk_key, tuple):
        chunk_key = '@'.join(chunk_key)
    return '%s_load_memory_%s' % (graph_key, chunk_key)
