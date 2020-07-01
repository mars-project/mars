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
import heapq
import logging
import os
import random
import sys
import time
from collections import defaultdict

from .. import promise
from ..config import options
from ..errors import DependencyMissing
from ..utils import log_unhandled
from .operands import BaseOperandActor
from .resource import ResourceActor
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
    def gen_uid(session_id):
        return 's:h1:assigner$%s' % session_id

    def __init__(self):
        super().__init__()
        self._requests = dict()
        self._req_heap = []

        self._cluster_info_ref = None
        self._actual_ref = None
        self._resource_ref = None

        self._worker_metrics = None
        # since worker metrics does not change frequently, we update it
        # only when it is out of date
        self._worker_metric_time = 0

        self._allocate_requests = []

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        # the ref of the actor actually handling assignment work
        session_id = self.uid.rsplit('$', 1)[-1]
        self._actual_ref = self.ctx.create_actor(AssignEvaluationActor, self.ref(),
                                                 uid=AssignEvaluationActor.gen_uid(session_id))
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

    def pre_destroy(self):
        self._actual_ref.destroy()

    def allocate_top_resources(self, max_allocates=None):
        self._allocate_requests.append(max_allocates)
        self._actual_ref.allocate_top_resources(fetch_requests=True, _tell=True, _wait=False)

    def get_allocate_requests(self):
        reqs = self._allocate_requests
        self._allocate_requests = []
        return reqs

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0
        self._actual_ref.mark_metrics_expired(_tell=True)

    def _refresh_worker_metrics(self):
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t

    def filter_alive_workers(self, workers, refresh=False):
        if refresh:
            self._refresh_worker_metrics()
        return [w for w in workers if w in self._worker_metrics] if self._worker_metrics else []

    def _enqueue_operand(self, session_id, op_key, op_info, callback=None):
        priority_item = ChunkPriorityItem(session_id, op_key, op_info, callback)
        if priority_item.target_worker not in self._worker_metrics:
            priority_item.target_worker = None
        self._requests[op_key] = priority_item
        heapq.heappush(self._req_heap, priority_item)

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
        self._allocate_requests.append(1)
        self._refresh_worker_metrics()
        self._enqueue_operand(session_id, op_key, op_info, callback)
        logger.debug('Operand %s enqueued', op_key)
        self._actual_ref.allocate_top_resources(fetch_requests=True, _tell=True, _wait=False)

    @log_unhandled
    def apply_for_multiple_resources(self, session_id, applications):
        self._allocate_requests.append(len(applications))
        self._refresh_worker_metrics()
        logger.debug('%d operands applied for session %s', len(applications), session_id)
        for app in applications:
            op_key, op_info = app
            self._enqueue_operand(session_id, op_key, op_info)
        self._actual_ref.allocate_top_resources(fetch_requests=True, _tell=True)

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
                # use latest request item
                item = self._requests[item.op_key]
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
    @classmethod
    def gen_uid(cls, session_id):
        return 's:0:%s$%s' % (cls.__name__, session_id)

    def __init__(self, assigner_ref):
        super().__init__()
        self._worker_metrics = None
        self._worker_metric_time = time.time() - 2

        self._cluster_info_ref = None
        self._assigner_ref = assigner_ref
        self._resource_ref = None

        self._sufficient_operands = set()
        self._operand_sufficient_time = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(self._assigner_ref)
        self._resource_ref = self.get_actor_ref(ResourceActor.default_uid())

        self.periodical_allocate()

    def mark_metrics_expired(self):
        logger.debug('Metrics cache marked as expired.')
        self._worker_metric_time = 0

    def periodical_allocate(self):
        self.allocate_top_resources()
        self.ref().periodical_allocate(_tell=True, _delay=0.5)

    def allocate_top_resources(self, fetch_requests=False):
        """
        Allocate resources given the order in AssignerActor
        """
        t = time.time()
        if self._worker_metrics is None or self._worker_metric_time + 1 < time.time():
            # update worker metrics from ResourceActor
            self._worker_metrics = self._resource_ref.get_workers_meta()
            self._worker_metric_time = t
        if not self._worker_metrics:
            return

        if fetch_requests:
            requests = self._assigner_ref.get_allocate_requests()
            if not requests:
                return
            max_allocates = sys.maxsize if any(v is None for v in requests) else sum(requests)
        else:
            max_allocates = sys.maxsize

        unassigned = []
        reject_workers = set()
        assigned = 0
        # the assigning procedure will continue till all workers rejected
        # or max_allocates reached
        while len(reject_workers) < len(self._worker_metrics) and assigned < max_allocates:
            item = self._assigner_ref.pop_head()
            if not item:
                break

            try:
                alloc_ep, rejects = self._allocate_resource(
                    item.session_id, item.op_key, item.op_info, item.target_worker,
                    reject_workers=reject_workers)
            except:  # noqa: E722
                logger.exception('Unexpected error occurred in %s', self.uid)
                if item.callback:  # pragma: no branch
                    self.tell_promise(item.callback, *sys.exc_info(), _accept=False)
                continue

            # collect workers failed to assign operand to
            reject_workers.update(rejects)
            if alloc_ep:
                # assign successfully, we remove the application
                self._assigner_ref.remove_apply(item.op_key, _tell=True)
                assigned += 1
            else:
                # put the unassigned item into unassigned list to add back to the queue later
                unassigned.append(item)
        if unassigned:
            # put unassigned back to the queue, if any
            self._assigner_ref.extend(unassigned, _tell=True)

        if not fetch_requests:
            self._assigner_ref.get_allocate_requests(_tell=True, _wait=False)

    @log_unhandled
    def _allocate_resource(self, session_id, op_key, op_info, target_worker=None, reject_workers=None):
        """
        Allocate resource for single operand
        :param session_id: session id
        :param op_key: operand key
        :param op_info: operand info dict
        :param target_worker: worker to allocate, can be None
        :param reject_workers: workers denied to assign to
        """
        if target_worker not in self._worker_metrics:
            target_worker = None

        reject_workers = reject_workers or set()

        op_io_meta = op_info.get('io_meta', {})
        try:
            input_metas = op_io_meta['input_data_metas']
            input_data_keys = list(input_metas.keys())
            input_sizes = dict((k, v.chunk_size) for k, v in input_metas.items())
        except KeyError:
            input_data_keys = op_io_meta.get('input_chunks', {})

            input_metas = self._get_chunks_meta(session_id, input_data_keys)
            missing_keys = [k for k, m in input_metas.items() if m is None]
            if missing_keys:
                raise DependencyMissing('Dependencies %r missing for operand %s' % (missing_keys, op_key))

            input_sizes = dict((k, meta.chunk_size) for k, meta in input_metas.items())

        if target_worker is None:
            who_has = dict((k, meta.workers) for k, meta in input_metas.items())
            candidate_workers = self._get_eps_by_worker_locality(input_data_keys, who_has, input_sizes)
        else:
            candidate_workers = [target_worker]

        candidate_workers = [w for w in candidate_workers if w not in reject_workers]
        if not candidate_workers:
            return None, []

        # todo make more detailed allocation plans
        calc_device = op_info.get('calc_device', 'cpu')
        if calc_device == 'cpu':
            alloc_dict = dict(cpu=options.scheduler.default_cpu_usage, memory=sum(input_sizes.values()))
        elif calc_device == 'cuda':
            alloc_dict = dict(cuda=options.scheduler.default_cuda_usage, memory=sum(input_sizes.values()))
        else:  # pragma: no cover
            raise NotImplementedError('Calc device %s not supported.' % calc_device)

        rejects = []
        for worker_ep in candidate_workers:
            if self._resource_ref.allocate_resource(session_id, op_key, worker_ep, alloc_dict):
                logger.debug('Operand %s(%s) allocated to run in %s', op_key, op_info['op_name'], worker_ep)

                self.get_actor_ref(BaseOperandActor.gen_uid(session_id, op_key)) \
                    .submit_to_worker(worker_ep, input_metas, _tell=True, _wait=False)
                return worker_ep, rejects
            rejects.append(worker_ep)
        return None, rejects

    def _get_chunks_meta(self, session_id, keys):
        if not keys:
            return dict()
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
