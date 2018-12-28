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
from collections import OrderedDict

from ..actors import ActorNotExist
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
        from .chunkstore import PlasmaChunkStore
        self._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        self._chunk_store = PlasmaChunkStore(self._plasma_client)

    def get_meta_ref(self, session_id, chunk_key):
        from ..scheduler.chunkmeta import LocalChunkMetaActor
        addr = self.get_scheduler((session_id, chunk_key))
        return self.ctx.actor_ref(LocalChunkMetaActor.default_name(), address=addr)


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


def concat_operand_keys(graph, sep=','):
    from ..tensor.expressions.datasource import TensorFetchChunk
    graph_op_dict = OrderedDict()
    for c in graph:
        if isinstance(c.op, TensorFetchChunk):
            continue
        graph_op_dict[c.op.key] = type(c.op).__name__
    keys = sep.join(graph_op_dict.keys())
    graph_ops = sep.join(graph_op_dict.values())
    return keys, graph_ops
