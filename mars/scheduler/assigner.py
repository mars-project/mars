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

from ..config import options
from ..errors import DependencyMissing
from ..utils import log_unhandled
from .chunkmeta import ChunkMetaActor
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
        self._chunk_meta_ref = None

        self._sufficient_operands = set()
        self._operand_sufficient_time = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        self._resource_ref = self.get_actor_ref(ResourceActor.default_name())
        self._chunk_meta_ref = self.ctx.actor_ref(ChunkMetaActor.default_name())

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    @log_unhandled
    def get_worker_assignments(self, session_id, op_info):
        """
        Register resource request for an operand
        :param session_id: session id
        :param op_info: operand information, should be a dict
        """
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t

        target_worker = op_info.get('target_worker')
        if target_worker and target_worker not in self._worker_metrics:
            target_worker = None

        op_io_meta = op_info['io_meta']
        input_chunk_keys = op_io_meta['input_chunks']
        metas = self._get_chunks_meta(session_id, input_chunk_keys)
        if any(meta is None for meta in metas.values()):
            raise DependencyMissing

        input_sizes = dict((k, meta.chunk_size) for k, meta in metas.items())
        output_size = op_info['output_size']

        if target_worker is None:
            op_name = op_info['op_name']
            chunk_workers = dict((k, meta.workers) for k, meta in metas.items())

            candidate_workers = self._get_eps_by_worker_locality(input_chunk_keys, chunk_workers, input_sizes)
            if self._is_stats_sufficient(op_name):
                ep = self._get_ep_by_worker_stats(input_chunk_keys, chunk_workers, input_sizes, output_size, op_name)
                if ep:
                    candidate_workers.append(ep)
        else:
            candidate_workers = [target_worker]

        return candidate_workers

    def _get_chunks_meta(self, session_id, keys):
        return dict(zip(keys, self._chunk_meta_ref.batch_get_chunk_meta(session_id, keys)))

    def _get_op_metric_item(self, ep, op_name, item):
        return self._get_metric_item(ep, 'calc_speed.' + op_name, item)

    def _get_metric_item(self, ep, metric, item):
        return self._worker_metrics[ep].get('stats', {}).get(metric, {}).get(item, 0)

    def _is_stats_sufficient(self, op_name):
        if not options.scheduler.enable_chunk_relocation:
            return False
        if op_name is None:
            return False

        if op_name in self._sufficient_operands:
            return True

        t = time.time()
        if self._operand_sufficient_time.get(op_name, 0) > t - 1:
            return False

        if any(self._worker_metrics[ep].get('stats', {}).get('max_est_finish_time') is None
               for ep in self._worker_metrics):
            self._operand_sufficient_time[op_name] = t
            return False
        minimal_stats = options.optimize.min_stats_count
        minimal_workers = len(self._worker_metrics) * options.optimize.stats_sufficient_ratio - 1e-6
        if sum(self._get_metric_item(ep, 'net_transfer_speed', 'count') >= minimal_stats
               for ep in self._worker_metrics) < minimal_workers:
            if all(self._get_metric_item(ep, 'net_transfer_speed', 'count') == 0 for ep in self._worker_metrics) or \
                    all(self._worker_metrics[ep].get('submitted_count', 0) > 0 for ep in self._worker_metrics):
                self._operand_sufficient_time[op_name] = t
                return False
        if any(self._get_op_metric_item(ep, op_name, 'count') < minimal_stats for ep in self._worker_metrics):
            self._operand_sufficient_time[op_name] = t
            return False
        if any(abs(self._get_op_metric_item(ep, op_name, 'mean')) < 1e-6 for ep in self._worker_metrics):
            self._operand_sufficient_time[op_name] = t
            return False

        if op_name in self._operand_sufficient_time:
            del self._operand_sufficient_time[op_name]
        self._sufficient_operands.add(op_name)
        return True

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

    def _get_ep_by_worker_stats(self, input_keys, chunk_workers, input_sizes, output_size, op_name):
        ep_net_speeds = dict()
        if any(self._get_metric_item(ep, 'net_transfer_speed', 'count') <= options.optimize.min_stats_count
               for ep in self._worker_metrics):
            sum_speeds, sum_records = 0, 0
            for ep in self._worker_metrics.keys():
                sum_speeds += self._get_metric_item(ep, 'net_transfer_speed', 'mean')
                sum_records += self._get_metric_item(ep, 'net_transfer_speed', 'count')
            avg_speed = sum_speeds * 1.0 / sum_records
        else:
            avg_speed = 0
        for ep in self._worker_metrics:
            if self._get_metric_item(ep, 'net_transfer_speed', 'count') > options.optimize.min_stats_count:
                ep_net_speeds[ep] = self._get_metric_item(ep, 'net_transfer_speed', 'mean')
            else:
                ep_net_speeds[ep] = avg_speed

        start_time = time.time()
        ep_transmit_times = defaultdict(list)
        ep_calc_time = defaultdict(lambda: 0)
        locality_data = defaultdict(lambda: 0)
        for key in input_keys:
            contain_eps = chunk_workers.get(key, set())
            for ep in self._worker_metrics:
                if ep not in contain_eps:
                    ep_transmit_times[ep].append(input_sizes[key] * 1.0 / ep_net_speeds[ep])
                else:
                    locality_data[ep] += input_sizes[key]

        ep_transmit_time = dict()
        for ep in self._worker_metrics:
            if ep_transmit_times[ep]:
                ep_transmit_time[ep] = sum(ep_transmit_times[ep])
            else:
                ep_transmit_time[ep] = 0

        max_eps = []
        max_locality = 0
        for ep, locality in locality_data.items():
            if locality == max_locality:
                max_eps.append(ep)
            elif locality > max_locality:
                max_eps = [ep]
                max_locality = locality
        max_eps = set(max_eps)

        for ep in self._worker_metrics:
            ep_calc_time[ep] = sum(input_sizes.values()) * 1.0 / self._get_op_metric_item(ep, op_name, 'mean')

        min_locality_exec_time = 0
        for ep in max_eps:
            finish_time = max(start_time, self._worker_metrics[ep].get('stats', {}).get('max_est_finish_time', 0.0)) + \
                          ep_transmit_time[ep] + ep_calc_time[ep]
            min_locality_exec_time = max(finish_time, min_locality_exec_time)

        min_finish_time = None
        min_ep = None
        for ep in self._worker_metrics:
            if ep_calc_time[ep] < 2 * ep_transmit_time[ep]:
                continue
            finish_time = max(start_time, self._worker_metrics[ep].get('stats', {}).get('min_est_finish_time', 0.0)) + \
                          ep_transmit_time[ep] + ep_calc_time[ep]
            if ep not in max_eps:
                finish_time += output_size * 1.0 / ep_net_speeds[ep]
            if finish_time - min_locality_exec_time > 1e-6:
                continue
            if min_finish_time is None or min_finish_time > finish_time:
                min_finish_time = finish_time
                min_ep = ep
        return min_ep
