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
import random
import time
from collections import defaultdict

from ..errors import DependencyMissing
from ..utils import log_unhandled
from .resource import ResourceActor
from .utils import SchedulerActor

logger = logging.getLogger(__name__)


class AssignerActor(SchedulerActor):
    """
    Actor handling worker assignment queries from operands
    and returning appropriate workers.
    """
    def __init__(self):
        super(AssignerActor, self).__init__()
        self._resource_ref = None

        self._worker_metrics = None
        # since worker metrics does not change frequently, we update it
        # only when it is out of date
        self._worker_metric_time = 0

        self._cluster_info_ref = None
        self._resource_actor_ref = None

        self._sufficient_operands = set()
        self._operand_sufficient_time = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        self._resource_ref = self.get_actor_ref(ResourceActor.default_name())

    def _refresh_worker_metrics(self):
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    def filter_alive_workers(self, workers, refresh=False):
        if refresh:
            self._refresh_worker_metrics()
        return [w for w in workers if w in self._worker_metrics] if self._worker_metrics else []

    @log_unhandled
    def get_worker_assignments(self, session_id, op_info):
        """
        Register resource request for an operand
        :param session_id: session id
        :param op_info: operand information, should be a dict
        """
        self._refresh_worker_metrics()

        # already assigned valid target worker, return directly
        target_worker = op_info.get('target_worker')
        if target_worker and target_worker in self._worker_metrics:
            return [target_worker]

        op_io_meta = op_info['io_meta']
        try:
            input_data_keys = op_io_meta['input_data_keys']
        except KeyError:
            input_data_keys = op_io_meta['input_chunks']

        metas = self._get_chunks_meta(session_id, input_data_keys)
        if any(meta is None for meta in metas.values()):
            raise DependencyMissing('Missing dependency meta %r' % [
                key for key, meta in metas.items() if meta is None
            ])

        input_sizes = dict((k, meta.chunk_size) for k, meta in metas.items())
        chunk_workers = dict((k, meta.workers) for k, meta in metas.items())
        candidate_workers = self._get_eps_by_worker_locality(input_data_keys, chunk_workers, input_sizes)

        return candidate_workers

    def _get_chunks_meta(self, session_id, keys):
        return dict(zip(keys, self.chunk_meta.batch_get_chunk_meta(session_id, keys)))

    def _get_eps_by_worker_locality(self, input_keys, chunk_workers, input_sizes):
        locality_data = defaultdict(lambda: 0)
        for k in input_keys:
            if k in chunk_workers:
                for ep in chunk_workers[k]:
                    locality_data[ep] += input_sizes[k]
        workers = list(self._worker_metrics.keys())
        random.shuffle(workers)
        max_locality = -1
        max_eps = []
        for ep in workers:
            if locality_data[ep] > max_locality:
                max_locality = locality_data[ep]
                max_eps = [ep]
            elif locality_data[ep] == max_locality:
                max_eps.append(ep)
        return max_eps
