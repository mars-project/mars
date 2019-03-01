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
from ..compat import OrderedDict3
from ..config import options
from ..promise import PromiseActor
from ..cluster_info import HasClusterInfoActor


logger = logging.getLogger(__name__)


class WorkerActor(HasClusterInfoActor, PromiseActor):
    """
    Base class of all worker actors, providing necessary utils
    """
    @classmethod
    def default_name(cls):
        return 'w:{0}'.format(cls.__name__)

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

        mapper_ref = self.ctx.actor_ref(uid=PlasmaKeyMapActor.default_name())
        self._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        self._chunk_store = PlasmaChunkStore(self._plasma_client, mapper_ref)

    def get_meta_ref(self, session_id, chunk_key, local=True):
        from ..scheduler.chunkmeta import ChunkMetaActor, LocalChunkMetaActor
        addr = self.get_scheduler((session_id, chunk_key))
        actor_cls = LocalChunkMetaActor if local else ChunkMetaActor
        return self.ctx.actor_ref(actor_cls.default_name(), address=addr)


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
