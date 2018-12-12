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

import os
import logging
import math
import contextlib

from .. import kvstore
from ..actors import ActorNotExist
from ..config import options
from ..promise import PromiseActor
from ..cluster_info import HasClusterInfoActor


logger = logging.getLogger(__name__)


class WorkerActor(HasClusterInfoActor, PromiseActor):
    """
    Base class of all worker actors, providing necessary utils
    """
    def __init__(self):
        super(WorkerActor, self).__init__()
        self._callbacks = dict()

        self._kv_store_ref = None
        self._kv_store = None
        if options.kv_store:
            self._kv_store = kvstore.get(options.kv_store)

    @classmethod
    def default_name(cls):
        return 'w:{0}'.format(cls.__name__)

    def post_create(self):
        from ..scheduler.kvstore import KVStoreActor

        logger.debug('Actor %s running in process %d', self.uid, os.getpid())
        try:
            self.set_cluster_info_ref()
            has_cluster_info = True
        except ActorNotExist:
            has_cluster_info = False
        if has_cluster_info:
            self._kv_store_ref = self.get_actor_ref(KVStoreActor.default_name())
        self._init_chunk_store()

    def _init_chunk_store(self):
        import pyarrow.plasma as plasma
        from .chunkstore import PlasmaChunkStore
        self._plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)
        self._chunk_store = PlasmaChunkStore(self._plasma_client, self._kv_store_ref)


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
