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

import heapq
import logging
import time

from .utils import WorkerActor
from .. import resource
from ..utils import log_unhandled

logger = logging.getLogger(__name__)
_ALLOCATE_PERIOD = 0.5


class ChunkPriorityItem(object):
    """
    Class providing an order for operands for assignment
    """
    __slots__ = '_op_key', '_session_id', '_priority', '_callback'

    def __init__(self, session_id, op_key, priority_data, callback):
        self._op_key = op_key
        self._session_id = session_id
        self._callback = callback

        self._priority = ()
        self.update_priority(priority_data)

    def update_priority(self, priority_data, copy=False):
        """
        Update priority data in the item
        :param priority_data: priority data
        :param copy: if True, the function will return a new item, otherwise the update will be applied locally
        """
        if copy:
            return ChunkPriorityItem(self._session_id, self._op_key, priority_data,
                                     self._callback)
        else:
            self._priority = tuple([
                priority_data.get('depth', 0),
                priority_data.get('demand_depths', ()),
                -priority_data.get('successor_size', 0),
                -priority_data.get('placement_order', 0),
                priority_data.get('descendant_size'),
            ])
            return self

    @property
    def session_id(self):
        return self._session_id

    @property
    def op_key(self):
        return self._op_key

    @property
    def callback(self):
        return self._callback

    def __lt__(self, other):
        """
        :type other: ChunkPriorityItem
        """
        return self._priority > other._priority


class TaskQueueActor(WorkerActor):
    """
    Actor accepting requests and holding the queue
    """
    def __init__(self, parallel_num=None):
        super(TaskQueueActor, self).__init__()
        self._requests = dict()
        self._allocated = set()
        self._allocate_pendings = set()
        self._req_heap = []

        self._allocator_ref = None
        self._parallel_num = parallel_num or resource.cpu_count()

    def post_create(self):
        self._allocator_ref = self.ctx.create_actor(
            TaskQueueAllocatorActor, self.ref(), self._parallel_num,
            uid=TaskQueueAllocatorActor.default_name())

    def enqueue_task(self, session_id, op_key, priority_data, callback):
        """
        Put a task in queue for allocation
        :param session_id: session id
        :param op_key: operand key
        :param priority_data: priority data
        :param callback: callback to invoke when the resources are allocated
        """
        logger.debug('Operand task %s enqueued.', op_key)
        item = ChunkPriorityItem(session_id, op_key, priority_data, callback)
        self._requests[(session_id, op_key)] = item
        heapq.heappush(self._req_heap, item)

        self._allocator_ref.allocate_tasks(_tell=True)

    def update_priority(self, session_id, op_key, priority_data):
        """
        Update priority data for the specified operand
        :param session_id: session id
        :param op_key: operand key
        :param priority_data: new priority data
        """
        logger.debug('Priority data for operand task %s updated.', op_key)
        query_key = (session_id, op_key)
        if query_key not in self._requests:
            return
        item = self._requests[query_key]
        item = item.update_priority(priority_data, copy=True)
        self._requests[query_key] = item
        heapq.heappush(self._req_heap, item)

    def mark_allocate_pending(self, session_id, op_key):
        """
        Mark an operand as being allocated, i.e., it has been submitted to the MemQuotaActor.
        :param session_id: session id
        :param op_key: operand key
        """
        self._allocate_pendings.add((session_id, op_key))

    def handle_allocated(self, session_id, op_key, callback, *args, **kwargs):
        """
        When MemQuotaActor allocates resource for an operand, put the operand into
        allocated and then invoke the callback
        :param session_id: session id
        :param op_key: operand key
        :param callback: callback to invoke
        """
        logger.debug('Operand task %s allocated.', op_key)
        query_key = (session_id, op_key)
        self.tell_promise(callback, *args, **kwargs)
        try:
            self._allocate_pendings.remove(query_key)
        except KeyError:
            pass
        self._allocated.add(query_key)

    def release_task(self, session_id, op_key):
        """
        Remove an operand task from queue
        :param session_id: session id
        :param op_key: operand key
        """
        logger.debug('Operand task %s released.', op_key)
        query_key = (session_id, op_key)
        try:
            del self._requests[(session_id, op_key)]
        except KeyError:
            pass
        try:
            self._allocated.remove(query_key)
        except KeyError:
            pass
        try:
            self._allocate_pendings.remove(query_key)
        except KeyError:
            pass

        # as one task has been released, we can perform allocation again
        self._allocator_ref.enable_quota(_tell=True)
        self._allocator_ref.allocate_tasks(_tell=True)

    def get_allocated_count(self):
        """
        Get total number of operands allocated to run and already running
        """
        return len(self._allocated) + len(self._allocate_pendings)

    def pop_next_request(self):
        """
        Get next unscheduled item from queue. If nothing found, None will
        be returned
        """
        item = None
        while self._req_heap:
            item = heapq.heappop(self._req_heap)
            query_key = (item.session_id, item.op_key)
            # if item is already scheduled or removed, we find next
            if query_key in self._requests:
                del self._requests[query_key]
                break
        return item


class TaskQueueAllocatorActor(WorkerActor):
    """
    Actor performing periodical assignment
    """
    def __init__(self, queue_ref, parallel_num):
        super(TaskQueueAllocatorActor, self).__init__()
        self._parallel_num = parallel_num
        self._has_quota = True

        self._queue_ref = queue_ref
        self._mem_quota_ref = None
        self._execution_ref = None
        self._last_memory_available = 0
        self._last_allocate_time = time.time() - 2

    def post_create(self):
        super(TaskQueueAllocatorActor, self).post_create()

        from .quota import MemQuotaActor
        from .execution import ExecutionActor

        self._queue_ref = self.ctx.actor_ref(self._queue_ref)
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_name())
        self._execution_ref = self.ctx.actor_ref(ExecutionActor.default_name())

        self.ref().allocate_tasks(periodical=True, _delay=_ALLOCATE_PERIOD, _tell=True)

    def enable_quota(self):
        self._has_quota = True

    @log_unhandled
    def allocate_tasks(self, periodical=False):
        # make sure the allocation period is not too dense
        if periodical and self._last_allocate_time > time.time() - _ALLOCATE_PERIOD:
            return
        cur_mem_available = resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            self._has_quota = True
        self._last_memory_available = cur_mem_available

        num_cpu = resource.cpu_count()
        cpu_rate = resource.cpu_percent()
        batch_allocated = 0
        while self._has_quota:
            allocated_count = self._queue_ref.get_allocated_count()
            if allocated_count >= self._parallel_num:
                break
            if allocated_count >= num_cpu / 4 and num_cpu * 100 - 50 < cpu_rate + batch_allocated:
                break
            item = self._queue_ref.pop_next_request()
            if item is None:
                break

            # obtain quota sizes for operands
            quota_request = self._execution_ref.prepare_quota_request(item.session_id, item.op_key)
            logger.debug('Quota request for %s: %r', item.op_key, quota_request)
            if quota_request:
                local_cb = ((self._queue_ref.uid, self._queue_ref.address),
                            TaskQueueActor.handle_allocated.__name__,
                            item.session_id, item.op_key, item.callback)
                self._queue_ref.mark_allocate_pending(item.session_id, item.op_key)
                self._has_quota = self._mem_quota_ref.request_batch_quota(quota_request, local_cb)
                batch_allocated += 1
            elif quota_request is None:
                # already processed, we skip to the next
                self.ctx.sleep(0.001)
                continue
            else:
                # allocate directly when no quota needed
                self._queue_ref.handle_allocated(item.session_id, item.op_key, item.callback, _tell=True)
                batch_allocated += 1
            self.ctx.sleep(0.001)

        self._last_allocate_time = time.time()
        self.ref().allocate_tasks(periodical=True, _delay=_ALLOCATE_PERIOD, _tell=True)
