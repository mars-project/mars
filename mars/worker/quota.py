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
import sys
import time

from .. import resource, promise
from ..compat import six, OrderedDict3
from ..utils import log_unhandled
from .utils import WorkerActor

logger = logging.getLogger(__name__)


class QuotaActor(WorkerActor):
    """
    Actor handling quota request and assignment
    """
    def __init__(self, total_size):
        super(QuotaActor, self).__init__()
        self._status_ref = None

        self._requests = OrderedDict3()

        self._total_size = total_size
        self._allocations = dict()
        self._allocated_size = 0

        self._proc_sizes = dict()
        self._total_proc = 0

        self._hold_sizes = dict()
        self._total_hold = 0

    def post_create(self):
        from .status import StatusActor

        super(QuotaActor, self).post_create()

        status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if self.ctx.has_actor(status_ref):
            self._status_ref = status_ref

    def _has_space(self, delta):
        return self._allocated_size + delta <= self._total_size

    def _log_allocate(self, msg, *args, **kwargs):
        args += (self._allocated_size, self._total_size)
        logger.debug(msg + ' Allocated: %s, Total size: %s', *args, **kwargs)

    @promise.reject_on_exception
    @log_unhandled
    def request_batch_quota(self, batch, callback):
        """
        Request for resources in a batch
        :param batch: the request dict in form {request_key: request_size, ...}
        :param callback: promise callback
        :return: if request is returned immediately, return True, otherwise False
        """
        all_allocated = True
        # check if the request is already allocated
        for key, size in batch.items():
            if key not in self._allocations or size > self._allocations.get(key):
                all_allocated = False
                break

        # if all requested and allocation can still be applied, apply directly
        if all_allocated and self._has_space(0):
            if callback is not None:
                self.tell_promise(callback)
            return True

        self._log_allocate('Receive batch quota request %r on %s.', batch, self.uid)
        sorted_req = sorted(batch.items(), key=lambda tp: tp[0])
        keys = tuple(tp[0] for tp in sorted_req)
        values = tuple(tp[1] for tp in sorted_req)
        delta = sum(v - self._allocations.get(k, 0) for k, v in batch.items())
        # make allocated requests the highest priority to be allocated
        return self._request_quota(keys, values, delta, callback, make_first=all_allocated)

    @promise.reject_on_exception
    @log_unhandled
    def request_quota(self, key, quota_size, callback):
        """
        Request for resource
        :param key: request key
        :param quota_size: size of request quota
        :param callback: promise callback
        :return: if request is returned immediately, return True, otherwise False
        """
        self._log_allocate('Receive quota request for key %s on %s.', key, self.uid)
        quota_size = int(quota_size)
        make_first = False
        # check if the request is already allocated
        if key in self._allocations:
            old_size = self._allocations[key]
            # if all requested and allocation can still be applied, apply directly
            if old_size <= quota_size and self._has_space(0):
                self.tell_promise(callback)
                return True
            else:
                # make allocated requests the highest priority to be allocated
                make_first = True
        else:
            old_size = 0
        return self._request_quota(key, quota_size, quota_size - old_size, callback,
                                   make_first=make_first)

    def _request_quota(self, keys, quota_sizes, delta, callback, make_first=False):
        """
        Actually process requests
        :param keys: request keys
        :param quota_sizes: request sizes
        :param delta: increase of allocate size
        :param callback: promise callback
        :param make_first: whether to move request keys to the highest priority
        :return: if request is returned immediately, return True, otherwise False
        """
        if keys in self._requests:
            # already in request queue, store callback and quit
            self._requests[keys][-1].append(callback)
            if make_first:
                self._requests.move_to_end(keys, False)
            return False

        if not isinstance(keys, tuple) and self._allocations.get(keys, 0) >= quota_sizes:
            # already allocated, inform and quit
            self.tell_promise(callback)
            return True

        if self._has_space(delta):
            if not self._requests:
                # if no previous requests, we can apply directly
                self._log_allocate('Quota request met for key %r on %s.', keys, self.uid)
                self.apply_allocation(keys, quota_sizes)
                if callback:
                    self.tell_promise(callback)
            else:
                # otherwise, previous requests are satisfied first
                self._log_allocate('Quota request queued for key %r on %s.', keys, self.uid)
                if keys not in self._requests:
                    self._requests[keys] = (quota_sizes, delta, time.time(), [])
                    if make_first:
                        self._requests.move_to_end(keys, False)
                if callback:
                    self._requests[keys][-1].append(callback)
                    if make_first:
                        self._requests.move_to_end(keys, False)

            self._process_requests()
            return True
        else:
            # current free space cannot satisfy the request, the request is queued
            self._log_allocate('Quota request unmet for key %r on %s.', keys, self.uid)
            if keys not in self._requests:
                self._requests[keys] = (quota_sizes, delta, time.time(), [])
                if make_first:
                    self._requests.move_to_end(keys, False)
            if callback:
                self._requests[keys][-1].append(callback)
                if make_first:
                    self._requests.move_to_end(keys, False)
            return False

    @log_unhandled
    def cancel_requests(self, keys, reject_exc=None):
        """
        Cancel a request if it is not assigned
        :param keys: request keys
        :param reject_exc: the exception to pass to the original callbacks
        """
        # normalize key as sorted tuple
        if isinstance(keys, six.string_types):
            keys = (keys,)
        else:
            keys = tuple(sorted(keys))
        # clean up requests from request_batch_quota() whose key is a tuple
        keys = keys + (keys,)
        for k in keys:
            if k in self._requests:
                if reject_exc:
                    for cb in self._requests[k][-1]:
                        self.tell_promise(cb, *reject_exc, **dict(_accept=False))
                del self._requests[k]
                logger.debug('Quota request %s cancelled', k)

    @log_unhandled
    def process_quota(self, key):
        """
        Mark request quota as being processed
        :param key: request key
        """
        if key not in self._allocations:
            return
        alloc_size = self._allocations[key]
        self._total_proc += alloc_size - self._proc_sizes.get(key, 0)
        self._proc_sizes[key] = alloc_size

    @log_unhandled
    def hold_quota(self, key):
        """
        Mark request quota as already been hold
        :param key: request key
        """
        if key not in self._allocations:
            return
        alloc_size = self._allocations[key]
        self._total_hold += alloc_size - self._hold_sizes.get(key, 0)
        self._hold_sizes[key] = alloc_size

        if key in self._proc_sizes:
            self._total_proc -= self._proc_sizes[key]
            del self._proc_sizes[key]

    @log_unhandled
    def release_quota(self, key):
        """
        Release allocated quota
        :param key: request key
        """
        if key not in self._allocations:
            return
        alloc_size = self._allocations[key]
        self._allocated_size -= alloc_size
        del self._allocations[key]

        if key in self._proc_sizes:
            self._total_proc -= self._proc_sizes[key]
            del self._proc_sizes[key]

        if key in self._hold_sizes:
            self._total_hold -= self._hold_sizes[key]
            del self._hold_sizes[key]

        self._process_requests()
        self._log_allocate('Quota key %s released on %s.', key, self.uid)

    @log_unhandled
    def release_quotas(self, keys):
        """
        Release allocated quota in batch
        :param keys: request keys
        """
        for k in keys:
            self.release_quota(k)

    def dump_state(self):
        logger.debug('State dump of %s:\nAllocations:%r\nRequests:%r',
                     self.uid, self._allocations, self._requests)

    def get_allocated_size(self):
        # get total allocated size, for debug purpose
        return self._allocated_size

    @log_unhandled
    def apply_allocation(self, key, quota_size, handle_shrink=True):
        """
        Accept a request
        :param key: request key
        :param quota_size: requested quota size
        :param handle_shrink: if True and the quota size less than the original, process requests in the queue
        """
        if isinstance(key, tuple):
            for k, s in zip(key, quota_size):
                self.apply_allocation(k, s, handle_shrink=handle_shrink)
            return
        quota_size = int(quota_size)
        old_size = self._allocations.get(key, 0)
        self._allocated_size += quota_size - old_size
        self._allocations[key] = quota_size
        if key in self._proc_sizes:
            self._total_proc += quota_size - self._proc_sizes[key]
            self._proc_sizes[key] = quota_size
        if key in self._hold_sizes:
            self._total_hold += quota_size - self._hold_sizes[key]
            self._hold_sizes[key] = quota_size
        self._log_allocate('Quota key %s applied on %s.', key, self.uid)
        if handle_shrink and quota_size < old_size:
            self._process_requests()

    @log_unhandled
    def _process_requests(self):
        """
        Process quota requests in the queue
        """
        removed = []
        for k, req in six.iteritems(self._requests):
            req_size, delta, req_time, callbacks = req
            try:
                if self._has_space(delta):
                    self.apply_allocation(k, req_size, handle_shrink=False)
                    for cb in callbacks:
                        self.tell_promise(cb)
                    if self._status_ref:
                        self._status_ref.update_mean_stats(
                            'wait_time.' + self.uid.replace('Actor', ''), time.time() - req_time,
                            _tell=True, _wait=False)
                    removed.append(k)
                else:
                    # Quota left cannot satisfy the next request, we quit
                    break
            except:  # noqa: E722
                removed.append(k)
                # just in case the quota is allocated
                self.release_quota(k)
                for cb in callbacks:
                    self.tell_promise(cb, *sys.exc_info(), **dict(_accept=False))
        for k in removed:
            self._requests.pop(k, None)


class MemQuotaActor(QuotaActor):
    """
    Actor handling worker memory quota
    """
    def __init__(self, total_size, overall_size=None):
        super(MemQuotaActor, self).__init__(total_size)
        self._overall_size = overall_size or total_size
        self._last_memory_available = 0

        self._dispatch_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor

        super(MemQuotaActor, self).post_create()
        self.update_mem_stats()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())

        if self._status_ref:
            self._status_ref.set_mem_quota_allocations(
                dict(allocated=self._allocated_size, hold=self._total_hold, total=self._total_size),
                _tell=True, _wait=False)

    def update_mem_stats(self):
        """
        Refresh memory usage
        """
        cur_mem_available = resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            self._process_requests()
        self._last_memory_available = cur_mem_available
        self.ref().update_mem_stats(_tell=True, _delay=10)

    def _has_space(self, delta):
        mem_stats = resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._overall_size) \
            - self._total_proc
        if delta >= available_size:
            logger.warning('%s met hard memory limitation: request %d, available %d, hard limit %d',
                           self.uid, delta, available_size, self._overall_size)
            for slot in self._dispatch_ref.get_slots('process_helper'):
                self.ctx.actor_ref(slot).free_mkl_buffers(_tell=True, _wait=False)
            return False
        return super(MemQuotaActor, self)._has_space(delta)

    def _log_allocate(self, msg, *args, **kwargs):
        mem_stats = resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._overall_size) \
            - self._total_proc
        args += (self._allocated_size, self._total_size, mem_stats.available, available_size,
                 self._overall_size, self._total_proc)

        logger.debug(
            msg + ' Allocated: %s, Total size: %s, Phy available: %s, Hard available: %s,'
                  ' Hard limit: %s, Processing: %s',
            *args, **kwargs
        )

    def apply_allocation(self, key, quota_size, handle_shrink=True):
        ret = super(MemQuotaActor, self).apply_allocation(
            key, quota_size, handle_shrink=handle_shrink)
        if self._status_ref:
            self._status_ref.set_mem_quota_allocations(
                dict(allocated=self._allocated_size, hold=self._total_hold, total=self._total_size),
                _tell=True, _wait=False)
        return ret

    def release_quota(self, key):
        ret = super(MemQuotaActor, self).release_quota(key)
        if self._status_ref:
            self._status_ref.set_mem_quota_allocations(
                dict(allocated=self._allocated_size, total=self._total_size),
                _tell=True, _wait=False)
        return ret

    def dump_keys(self):
        return list(self._allocations.keys())
