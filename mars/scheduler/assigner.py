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
import os
import random
import time
from collections import defaultdict

from .. import promise
from ..utils import log_unhandled
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
        session_id = self.uid.rsplit('$', 1)[-1]
        self._actual_ref = self.ctx.create_actor(AssignEvaluationActor, self.ref(),
                                                 uid=AssignEvaluationActor.gen_uid(session_id))
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_uid())

    def pre_destroy(self):
        self._actual_ref.destroy()

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
    @classmethod
    def gen_uid(cls, session_id):
        return 's:0:%s$%s' % (cls.__name__, session_id)

    def __init__(self, assigner_ref):
        super(AssignEvaluationActor, self).__init__()
        self._worker_metrics = None
        self._worker_metric_time = time.time() - 2

        self._cluster_info_ref = None
        self._assigner_ref = assigner_ref
        self._resource_actor_ref = None

        self._sufficient_operands = set()
        self._operand_sufficient_time = dict()

    def post_create(self):
        logger.debug('Actor %s running in process %d', self.uid, os.getpid())

        self.set_cluster_info_ref()
        self._assigner_ref = self.ctx.actor_ref(self._assigner_ref)
        self._resource_actor_ref = self.get_actor_ref(ResourceActor.default_uid())

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

        op_io_meta = op_info['io_meta']
        input_chunk_keys = op_io_meta['input_chunks']
        metas = self._get_chunks_meta(session_id, input_chunk_keys)
        input_sizes = dict((k, meta.chunk_size) for k, meta in metas.items())

        if target_worker is None:
            who_has = dict((k, meta.workers) for k, meta in metas.items())

            candidate_workers = self._get_eps_by_worker_locality(input_chunk_keys, who_has, input_sizes)
        else:
            candidate_workers = [target_worker]

        candidate_workers = [w for w in candidate_workers if w not in reject_workers]
        if not candidate_workers:
            return None, []

        # todo make more detailed allocation plans
        alloc_dict = dict(cpu=1, memory=sum(input_sizes.values()))
        rejects = []
        for worker_ep in candidate_workers:
            if self._resource_actor_ref.allocate_resource(session_id, op_key, worker_ep, alloc_dict):
                logger.debug('Operand %s(%s) allocated to run in %s', op_key, op_info['op_name'], worker_ep)

                self.tell_promise(callback, worker_ep)
                return worker_ep, rejects
            rejects.append(worker_ep)
        return None, rejects

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
