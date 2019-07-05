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

import itertools
import logging
import sys
import time
from collections import namedtuple

from .. import resource, promise
from ..compat import OrderedDict3
from ..utils import log_unhandled
from .utils import WorkerActor

logger = logging.getLogger(__name__)
QuotaDumpType = namedtuple('QuotaDumpType', 'allocations requests proc_sizes hold_sizes')


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
    def request_batch_quota(self, batch, callback=None):
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
            self._log_allocate('Quota request %r already allocated.', batch)
            if callback is not None:
                self.tell_promise(callback)
            return True

        self._log_allocate('Receive batch quota request %r on %s.', batch, self.uid)
        sorted_req = sorted(batch.items(), key=lambda tp: tp[0])
        keys = tuple(tp[0] for tp in sorted_req)
        values = tuple(tp[1] for tp in sorted_req)
        delta = sum(v - self._allocations.get(k, 0) for k, v in batch.items())
        # make allocated requests the highest priority to be allocated
        return self._request_quota(keys, values, delta, callback, multiple=True,
                                   make_first=all_allocated)

    @promise.reject_on_exception
    @log_unhandled
    def request_quota(self, key, quota_size, callback=None):
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
            if old_size >= quota_size and self._has_space(0):
                if callback is not None:
                    self.tell_promise(callback)
                return True
            else:
                # make allocated requests the highest priority to be allocated
                make_first = True
        else:
            old_size = 0
        return self._request_quota(key, quota_size, quota_size - old_size, callback,
                                   make_first=make_first)

    def _request_quota(self, keys, quota_sizes, delta, callback, multiple=False,
                       make_first=False):
        """
        Actually process requests
        :param keys: request keys
        :param quota_sizes: request sizes
        :param delta: increase of allocate size
        :param callback: promise callback
        :param make_first: whether to move request keys to the highest priority
        :return: if request is returned immediately, return True, otherwise False
        """
        if delta > self._total_size:
            raise ValueError('Cannot allocate size larger than the total capacity.')

        if keys in self._requests:
            # already in request queue, store callback and quit
            if callback is not None:
                self._requests[keys][-1].append(callback)
            if make_first:
                self._requests.move_to_end(keys, False)
            return False

        if self._has_space(delta):
            if not self._requests:
                # if no previous requests, we can apply directly
                allocated = True

                self._log_allocate('Quota request met for key %r on %s.', keys, self.uid)

                alter_allocation = self.alter_allocations if multiple else self.alter_allocation
                alter_allocation(keys, quota_sizes)

                if callback:
                    self.tell_promise(callback)
            else:
                # otherwise, previous requests are satisfied first
                allocated = False

                self._log_allocate('Quota request queued for key %r on %s.', keys, self.uid)
                self._enqueue_request(keys, (quota_sizes, delta, time.time(), multiple, []),
                                      callback=callback, make_first=make_first)

            self._process_requests()
            return allocated
        else:
            # current free space cannot satisfy the request, the request is queued
            self._log_allocate('Quota request unmet for key %r on %s.', keys, self.uid)
            self._enqueue_request(keys, (quota_sizes, delta, time.time(), multiple, []),
                                  callback=callback, make_first=make_first)
            return False

    def _enqueue_request(self, keys, items, callback=None, make_first=False):
        if keys not in self._requests:
            self._requests[keys] = items
        if callback is not None:
            self._requests[keys][-1].append(callback)

        if make_first:
            self._requests.move_to_end(keys, False)

    @log_unhandled
    def cancel_requests(self, keys, reject_exc=None):
        """
        Cancel a request if it is not assigned
        :param keys: request keys
        :param reject_exc: the exception to pass to the original callbacks
        """
        # normalize key as sorted tuple
        keys = tuple(sorted(keys))
        # clean up requests from request_batch_quota() whose key is a tuple
        keys = keys + (keys,)
        for k in keys:
            try:
                if reject_exc:
                    for cb in self._requests[k][-1]:
                        self.tell_promise(cb, *reject_exc, **dict(_accept=False))
                del self._requests[k]
                logger.debug('Quota request %s cancelled', k)
            except KeyError:
                pass
        self._process_requests()

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

    def dump_data(self):
        return QuotaDumpType(self._allocations, self._requests, self._proc_sizes, self._hold_sizes)

    def get_allocated_size(self):
        # get total allocated size, for debug purpose
        return self._allocated_size

    def alter_allocations(self, keys, quota_sizes=None, handle_shrink=True, new_keys=None):
        """
        Alter multiple requests
        :param keys: keys to update
        :param quota_sizes: new quota sizes, if None, no changes will be made
        :param handle_shrink: if True and the quota size less than the original, process requests in the queue
        :param new_keys: new allocation keys to replace current keys, if None, no changes will be made
        :return:
        """
        quota_sizes = quota_sizes or itertools.repeat(None)
        new_keys = new_keys or itertools.repeat(None)
        shrink = False
        for k, s, nk in zip(keys, quota_sizes, new_keys):
            shrink = shrink or self.alter_allocation(k, s, handle_shrink=False, new_key=nk)
        if shrink and handle_shrink:
            self._process_requests()

    @log_unhandled
    def alter_allocation(self, key, quota_size=None, handle_shrink=True, new_key=None):
        """
        Alter a single request by changing its name or request size
        :param key: request key
        :param quota_size: requested quota size
        :param handle_shrink: if True and the quota size less than the original, process requests in the queue
        :param new_key: new allocation key to replace current key
        """
        old_size = self._allocations.get(key, 0)

        if quota_size is not None and quota_size != old_size:
            quota_size = int(quota_size)
            size_diff = quota_size - old_size
            self._allocated_size += size_diff
            self._allocations[key] = quota_size
            if key in self._proc_sizes:
                self._total_proc += quota_size - self._proc_sizes[key]
                self._proc_sizes[key] = quota_size
            if key in self._hold_sizes:
                self._total_hold += quota_size - self._hold_sizes[key]
                self._hold_sizes[key] = quota_size
            self._log_allocate('Quota key %s applied on %s. Diff: %s,', key, self.uid, size_diff)

        if key in self._allocations and new_key is not None and new_key != key:
            self._allocations[new_key] = self._allocations[key]
            del self._allocations[key]
            try:
                self._proc_sizes[new_key] = self._proc_sizes[key]
                del self._proc_sizes[key]
            except KeyError:
                pass
            try:
                self._hold_sizes[new_key] = self._hold_sizes[key]
                del self._hold_sizes[key]
            except KeyError:
                pass

        if quota_size is not None and quota_size < old_size:
            if handle_shrink:
                self._process_requests()
            return True
        return False

    @log_unhandled
    def _process_requests(self):
        """
        Process quota requests in the queue
        """
        removed = []
        for k, req in self._requests.items():
            req_size, delta, req_time, multiple, callbacks = req
            try:
                if self._has_space(delta):
                    alter_allocation = self.alter_allocations if multiple else self.alter_allocation
                    alter_allocation(k, req_size, handle_shrink=False)
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
    def __init__(self, total_size, overall_size=None, refresh_time=None):
        super(MemQuotaActor, self).__init__(total_size)
        self._overall_size = overall_size or total_size
        self._last_memory_available = 0
        self._refresh_time = refresh_time or 10

        self._dispatch_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor

        super(MemQuotaActor, self).post_create()
        self.update_mem_stats()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())

        self._update_status(allocated=self._allocated_size, hold=self._total_hold, total=self._total_size)

    def update_mem_stats(self):
        """
        Refresh memory usage
        """
        cur_mem_available = resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            self._process_requests()
        self._last_memory_available = cur_mem_available
        self.ref().update_mem_stats(_tell=True, _delay=self._refresh_time)

    def _has_space(self, delta):
        mem_stats = resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._overall_size) \
            - self._total_proc
        if max(delta, 0) >= available_size:
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

    def _update_status(self, **kwargs):
        if self._status_ref:
            self._status_ref.set_mem_quota_allocations(kwargs, _tell=True, _wait=False)

    def alter_allocation(self, key, quota_size=None, handle_shrink=True, new_key=None):
        ret = super(MemQuotaActor, self).alter_allocation(
            key, quota_size, handle_shrink=handle_shrink, new_key=new_key)
        self._update_status(allocated=self._allocated_size, hold=self._total_hold, total=self._total_size)
        return ret

    def release_quota(self, key):
        ret = super(MemQuotaActor, self).release_quota(key)
        self._update_status(allocated=self._allocated_size, total=self._total_size)
        return ret
