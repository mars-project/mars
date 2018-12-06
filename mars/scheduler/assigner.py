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

import copy
import heapq
import logging
import random
import time
import os
from collections import defaultdict

from .. import promise
from ..config import options
from ..utils import log_unhandled
from .resource import ResourceActor
from .kvstore import KVStoreActor
from .utils import SchedulerActor

logger = logging.getLogger(__name__)


class ChunkPriorityItem(object):
    """
    Class providing an order for operands for assignment
    """
    def __init__(self, session_id, op_key, op_info, callback):
        self._op_key = op_key
        self._session_id = session_id
        self._op_info = op_info
        self._target_worker = op_info.get('target_worker')
        self._callback = callback

        self._priority = ()
        self.update_priority(op_info['optimize'])

    def update_priority(self, priority_data, copyobj=False):
        obj = self
        if copyobj:
            obj = copy.deepcopy(obj)

        priorities = []
        priorities.extend([
            priority_data.get('depth', 0),
            priority_data.get('demand_depths', ()),
            -priority_data.get('successor_size', 0),
            -priority_data.get('placement_order', 0),
            priority_data.get('descendant_size'),
        ])
        obj._priority = tuple(priorities)
        return obj

    @property
    def session_id(self):
        return self._session_id

    @property
    def op_key(self):
        return self._op_key

    @property
    def target_worker(self):
        return self._target_worker

    @target_worker.setter
    def target_worker(self, value):
        self._target_worker = value

    @property
    def callback(self):
        return self._callback

    @property
    def op_info(self):
        return self._op_info

    def __repr__(self):
        return '<ChunkPriorityItem(%s(%s))>' % (self.op_key, self.op_info['op_name'])

    def __lt__(self, other):
        return self._priority > other._priority


class AssignerActor(SchedulerActor):
    """
    Actor handling worker assignment requests from operands.
    Note that this actor does not assign workers itself.
    """
    @staticmethod
    def gen_name(session_id):
        return 's:assigner$%s' % session_id

    def __init__(self):
        super(AssignerActor, self).__init__()
        self._requests = dict()
        self._req_heap = []

        self._cluster_info_ref = None
        self._actual_ref = None
        self._resource_actor_ref = None

        self._worker_metrics = None
        # since worker metrics does not change frequently, we update it
        # only when it is out of date
        self._worker_metric_time = time.time() - 2

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        self._actual_ref = self.ctx.create_actor(AssignEvaluationActor, self.ref())
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_name())

    def allocate_top_resources(self):
        self._actual_ref.allocate_top_resources(_tell=True)

    @promise.reject_on_exception
    @log_unhandled
    def apply_for_resource(self, session_id, op_key, op_info, callback=None):
        """
        Register resource request for an operand
        :param session_id: session id
        :param op_key: operand key
        :param op_info: operand information, should be a dict
        :param callback: promise callback, called when the resource is assigned
        """
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_actor_ref.get_workers_meta()
            self._worker_metric_time = t

        priority_item = ChunkPriorityItem(session_id, op_key, op_info, callback)
        if priority_item.target_worker not in self._worker_metrics:
            priority_item.target_worker = None
        self._requests[op_key] = priority_item
        heapq.heappush(self._req_heap, priority_item)
        self._actual_ref.allocate_top_resources(_tell=True)

    @log_unhandled
    def update_priority(self, op_key, priority_data):
        """
        Update priority data for an operand. The priority item will be
        pushed into priority queue again.
        :param op_key: operand key
        :param priority_data: new priority data
        """
        if op_key not in self._requests:
            return
        obj = self._requests[op_key].update_priority(priority_data, copyobj=True)
        heapq.heappush(self._req_heap, obj)

    @log_unhandled
    def remove_apply(self, op_key):
        """
        Cancel request for an operand
        :param op_key: operand key
        """
        if op_key in self._requests:
            del self._requests[op_key]

    def pop_head(self):
        """
        Pop and obtain top-priority request from queue
        :return: top item
        """
        item = None
        while self._req_heap:
            item = heapq.heappop(self._req_heap)
            if item.op_key in self._requests:
                break
            else:
                item = None
        return item

    def extend(self, items):
        """
        Extend heap by an iterable object. The heap will be reheapified.
        :param items: priority items
        """
        self._req_heap.extend(items)
        heapq.heapify(self._req_heap)


class AssignEvaluationActor(SchedulerActor):
    """
    Actor assigning operands to workers
    """
    def __init__(self, assigner_ref):
        super(AssignEvaluationActor, self).__init__()
        self._worker_metrics = None
        self._worker_metric_time = time.time() - 2

        self._cluster_info_ref = None
        self._assigner_ref = assigner_ref
        self._resource_actor_ref = None
        self._kv_store_ref = None

        self._sufficient_operands = set()
        self._operand_sufficient_time = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(self._assigner_ref)
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_name())
        self._kv_store_ref = self.get_actor_ref(KVStoreActor.default_name())

        self.periodical_allocate()

    def periodical_allocate(self):
        self.allocate_top_resources()
        self.ref().periodical_allocate(_tell=True, _delay=1)

    def allocate_top_resources(self):
        """
        Allocate resources given the order in AssignerActor
        """
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_actor_ref.get_workers_meta()
            self._worker_metric_time = t
        if not self._worker_metrics:
            return

        unassigned = []
        reject_workers = set()
        # the assigning procedure will continue till
        while len(reject_workers) < len(self._worker_metrics):
            item = self._assigner_ref.pop_head()
            if not item:
                break

            try:
                alloc_ep, rejects = self._allocate_resource(
                    item.session_id, item.op_key, item.op_info, item.target_worker,
                    reject_workers=reject_workers, callback=item.callback)
            except:
                logger.exception('Unexpected error occurred in %s', self.uid)
                unassigned.append(item)
                continue

            # collect workers failed to assign operand to
            reject_workers.update(rejects)
            if alloc_ep:
                # assign successfully, we remove the application
                self._assigner_ref.remove_apply(item.op_key)
            else:
                # put the unassigned item into unassigned list to add back to the queue later
                unassigned.append(item)
        if unassigned:
            # put unassigned back to the queue, if any
            self._assigner_ref.extend(unassigned)

    @promise.reject_on_exception
    @log_unhandled
    def _allocate_resource(self, session_id, op_key, op_info, target_worker=None, reject_workers=None, callback=None):
        """
        Allocate resource for single operand
        :param session_id: session id
        :param op_key: operand key
        :param op_info: operand info dict
        :param target_worker: worker to allocate, can be None
        :param reject_workers: workers denied to assign to
        :param callback: promise callback from operands
        """
        if target_worker not in self._worker_metrics:
            target_worker = None

        reject_workers = reject_workers or set()

        op_path = '/sessions/%s/operands/%s' % (session_id, op_key)

        op_io_meta = op_info['io_meta']
        input_chunk_keys = op_io_meta['input_chunks']
        input_sizes = dict(zip(input_chunk_keys, self._get_multiple_chunk_size(session_id, input_chunk_keys)))
        output_size = op_info['output_size']

        if target_worker is None:
            op_name = op_info['op_name']
            who_has = dict(zip(input_chunk_keys, self._get_multiple_who_has(session_id, input_chunk_keys)))

            candidate_workers = self._get_eps_by_worker_locality(input_chunk_keys, who_has, input_sizes)
            locality_workers = set(candidate_workers)
            if self._is_stats_sufficient(op_name):
                ep = self._get_ep_by_worker_stats(input_chunk_keys, who_has, input_sizes, output_size, op_name)
                if ep:
                    candidate_workers.append(ep)
        else:
            candidate_workers = [target_worker]
            locality_workers = {target_worker}

        candidate_workers = [w for w in candidate_workers if w not in reject_workers]
        if not candidate_workers:
            return None, []

        # todo make more detailed allocation plans
        # alloc_dict = dict(cpu=1, memory=sum(six.itervalues(input_sizes)) + output_size)
        alloc_dict = dict(cpu=1, memory=output_size)
        rejects = []
        for worker_ep in candidate_workers:
            if self._resource_actor_ref.allocate_resource(session_id, op_key, worker_ep, alloc_dict):
                if worker_ep in locality_workers:
                    logger.debug('Operand %s(%s) allocated to run in %s', op_key, op_info['op_name'], worker_ep)
                else:
                    logger.debug('Operand %s(%s) allocated to run in %s given collected statistics',
                                 op_key, op_info['op_name'], worker_ep)

                self._kv_store_ref.write('%s/worker' % op_path, worker_ep)
                self.tell_promise(callback, worker_ep)
                return worker_ep, rejects
            rejects.append(worker_ep)
        return None, rejects

    def _get_who_has(self, session_id, chunk_key):
        return [ch.key.rsplit('/', 1)[-1] for ch in self._kv_store_ref.read(
            '/sessions/%s/chunks/%s/workers' % (session_id, chunk_key)).children]

    def _get_multiple_who_has(self, session_id, chunk_keys):
        keys = ['/sessions/%s/chunks/%s/workers' % (session_id, chunk_key) for chunk_key in chunk_keys]
        for result in self._kv_store_ref.read_batch(keys):
            yield [ch.key.rsplit('/', 1)[-1] for ch in result.children]

    def _get_chunk_size(self, session_id, chunk_key):
        return self._kv_store_ref.read(
            '/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key)).value

    def _get_multiple_chunk_size(self, session_id, chunk_keys):
        if not chunk_keys:
            return tuple()
        keys = ['/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key)
                for chunk_key in chunk_keys]
        return (res.value for res in self._kv_store_ref.read_batch(keys))

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

    def _get_eps_by_worker_locality(self, input_keys, who_has, input_sizes):
        locality_data = defaultdict(lambda: 0)
        for k in input_keys:
            if k in who_has:
                for ep in who_has[k]:
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

    def _get_ep_by_worker_stats(self, input_keys, who_has, input_sizes, output_size, op_name):
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
            contain_eps = who_has.get(key, set())
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
