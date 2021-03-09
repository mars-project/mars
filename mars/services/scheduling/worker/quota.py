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

import asyncio
import itertools
import logging
import time
from collections import namedtuple, OrderedDict

from .... import oscar as mo
from .... import resource as mars_resource

logger = logging.getLogger(__name__)

QuotaDumpType = namedtuple('QuotaDumpType', 'allocations requests proc_sizes hold_sizes')


class QuotaRequest:
    __slots__ = 'req_size', 'delta', 'req_time', 'multiple', 'process_quota', \
                'event', 'wait_task'

    def __init__(self, req_size: int, delta: int, req_time: float,
                 multiple: bool, process_quota: bool, event: asyncio.Event):
        self.req_size = req_size
        self.delta = delta
        self.req_time = req_time
        self.multiple = multiple
        self.process_quota = process_quota
        self.event = event
        self.wait_task = None


class QuotaActor(mo.Actor):
    def __init__(self, total_size):
        super().__init__()
        self._requests = OrderedDict()

        self._total_size = total_size
        self._allocations = dict()
        self._allocated_size = 0

        self._proc_sizes = dict()
        self._total_proc = 0

        self._hold_sizes = dict()
        self._total_hold = 0

    def _has_space(self, delta):
        return self._allocated_size + delta <= self._total_size

    def _log_allocate(self, msg, *args, **kwargs):
        args += (self._allocated_size, self._total_size)
        logger.debug(msg + ' Allocated: %s, Total size: %s', *args, **kwargs)

    async def request_batch_quota(self, batch, process_quota=False):
        """
        Request for resources in a batch
        :param batch: the request dict in form {request_key: request_size, ...}
        :param process_quota: once handled, treat quota as processing
        :return: if request is returned immediately, return True, otherwise False
        """
        all_allocated = True
        # check if the request is already allocated
        for key, size in batch.items():
            if key not in self._allocations or size > self._allocations.get(key):
                all_allocated = False
                break

        self._log_allocate('Receive batch quota request %r on %s.', batch, self.uid)
        sorted_req = sorted(batch.items(), key=lambda tp: tp[0])
        keys = tuple(tp[0] for tp in sorted_req)
        values = tuple(tp[1] for tp in sorted_req)
        delta = sum(v - self._allocations.get(k, 0) for k, v in batch.items())

        # if all requested and allocation can still be applied, apply directly
        if all_allocated and self._has_space(delta):
            self._log_allocate('Quota request %r already allocated.', batch)
            if process_quota:
                self.process_quotas(list(batch.keys()))
            return

        # make allocated requests the highest priority to be allocated
        return self._request_quota(
            keys, values, delta, multiple=True, make_first=all_allocated,
            process_quota=process_quota)

    async def request_quota(self, key, quota_size, process_quota=False):
        """
        Request for resource
        :param key: request key
        :param quota_size: size of request quota
        :param process_quota: once handled, treat quota as processing
        :return: if request is returned immediately, return True, otherwise False
        """
        self._log_allocate('Receive quota request for key %s on %s.', key, self.uid)
        quota_size = int(quota_size)
        make_first = False
        # check if the request is already allocated
        if key in self._allocations:
            old_size = self._allocations[key]
            # if all requested and allocation can still be applied, apply directly
            if old_size >= quota_size and self._has_space(quota_size - old_size):
                if process_quota:
                    self.process_quotas([key])
                return
            else:
                # make allocated requests the highest priority to be allocated
                make_first = True
        else:
            old_size = 0
        return self._request_quota(
            key, quota_size, quota_size - old_size, make_first=make_first,
            process_quota=process_quota)

    async def _request_quota(self, keys, quota_sizes, delta, multiple=False,
                             make_first=False, process_quota=False):
        """
        Actually process requests
        :param keys: request keys
        :param quota_sizes: request sizes
        :param delta: increase of allocate size
        :param make_first: whether to move request keys to the highest priority
        :param process_quota: once handled, treat quota as processing
        :return: if request is returned immediately, return True, otherwise False
        """
        if delta > self._total_size:
            raise ValueError(f'Cannot allocate quota size {delta} '
                             f'larger than total capacity {self._total_size}.')

        if keys in self._requests:
            quota_request = self._requests[keys]
        else:
            has_space = self._has_space(delta)
            if has_space and not self._requests:
                # if no previous requests, we can apply directly
                self._log_allocate('Quota request met for key %r on %s.', keys, self.uid)

                alter_allocation = self.alter_allocations if multiple else self.alter_allocation
                alter_allocation(keys, quota_sizes, allocate=True, process_quota=process_quota)
                return
            else:
                # current free space cannot satisfy the request, the request is queued
                if not has_space:
                    self._log_allocate('Quota request unmet for key %r on %s.', keys, self.uid)
                else:
                    self._log_allocate('Quota request queued for key %r on %s.', keys, self.uid)
                quota_request = QuotaRequest(quota_sizes, delta, time.time(), multiple,
                                             process_quota, asyncio.Event())
                self._enqueue_request(keys, quota_request, make_first=make_first)

        try:
            quota_request.wait_task = asyncio.create_task(quota_request.event.wait())
            await quota_request.wait_task
        except asyncio.CancelledError:
            self._requests.pop(keys, None)
            self._process_requests()
            raise

    def _enqueue_request(self, keys, request, make_first=False):
        process_quota = request.process_quota
        try:
            request = self._requests[keys]
            request.process_quota = process_quota
        except KeyError:
            self._requests[keys] = request

        if make_first:
            self._requests.move_to_end(keys, False)

    def process_quotas(self, keys):
        """
        Mark request quota as being processed
        :param keys: request keys
        """
        for key in keys:
            try:
                alloc_size = self._allocations[key]
            except KeyError:
                continue
            self._total_proc += alloc_size - self._proc_sizes.get(key, 0)
            self._proc_sizes[key] = alloc_size

    def hold_quotas(self, keys):
        """
        Mark request quota as already been hold
        :param keys: request keys
        """
        for key in keys:
            try:
                alloc_size = self._allocations[key]
            except KeyError:
                continue
            self._total_hold += alloc_size - self._hold_sizes.get(key, 0)
            self._hold_sizes[key] = alloc_size

            self._total_proc -= self._proc_sizes.pop(key, 0)

    def release_quotas(self, keys):
        """
        Release allocated quota in batch
        :param keys: request keys
        """
        total_alloc_size = 0

        for key in keys:
            try:
                alloc_size = self._allocations.pop(key)
                total_alloc_size += alloc_size
            except KeyError:
                continue
            self._total_proc -= self._proc_sizes.pop(key, 0)
            self._total_hold -= self._hold_sizes.pop(key, 0)

        self._allocated_size -= total_alloc_size
        if total_alloc_size:
            self._process_requests()
            self._log_allocate('Quota keys %s released on %s.', keys, self.uid)

    def dump_data(self):
        return QuotaDumpType(self._allocations, self._requests, self._proc_sizes, self._hold_sizes)

    def get_allocated_size(self):
        # get total allocated size, for debug purpose
        return self._allocated_size

    def alter_allocations(self, keys, quota_sizes=None, handle_shrink=True, new_keys=None,
                          allocate=False, process_quota=False):
        """
        Alter multiple requests
        :param keys: keys to update
        :param quota_sizes: new quota sizes, if None, no changes will be made
        :param handle_shrink: if True and the quota size less than the original, process requests in the queue
        :param new_keys: new allocation keys to replace current keys, if None, no changes will be made
        :param allocate: if True, will allocate resources for new items
        :param process_quota: call process_quotas() after allocated
        :return:
        """
        quota_sizes = quota_sizes or itertools.repeat(None)
        new_keys = new_keys or itertools.repeat(None)
        total_old_size, total_diff = 0, 0
        for k, s, nk in zip(keys, quota_sizes, new_keys):
            old_size, diff = self.alter_allocation(k, s, handle_shrink=False, new_key=nk, allocate=allocate,
                                                   process_quota=process_quota, log_allocate=False)
            total_old_size += old_size
            total_diff += diff
        if handle_shrink and total_diff < 0:
            self._process_requests()
        self._log_allocate('Quota keys %r applied on %s. Total old Size: %s, Total diff: %s,',
                           keys, self.uid, total_old_size, total_diff)

    def alter_allocation(self, key, quota_size=None, handle_shrink=True, new_key=None,
                         allocate=False, process_quota=False, log_allocate=True):
        """
        Alter a single request by changing its name or request size
        :param key: request key
        :param quota_size: requested quota size
        :param handle_shrink: if True and the quota size less than the original, process requests in the queue
        :param new_key: new allocation key to replace current key
        :param allocate: if True, will allocate resources for new items
        :param process_quota: call process_quotas() after allocated
        """
        old_size = self._allocations.get(key, 0)
        size_diff = 0

        if not allocate and key not in self._allocations:
            return old_size, 0

        if quota_size is not None and quota_size != old_size:
            quota_size = int(quota_size)
            size_diff = quota_size - old_size
            self._allocated_size += size_diff
            self._allocations[key] = quota_size
            try:
                self._total_proc += quota_size - self._proc_sizes[key]
                self._proc_sizes[key] = quota_size
            except KeyError:
                pass
            try:
                self._total_hold += quota_size - self._hold_sizes[key]
                self._hold_sizes[key] = quota_size
            except KeyError:
                pass
            if log_allocate:
                self._log_allocate('Quota key %s applied on %s. Old Size: %s, Diff: %s,',
                                   key, self.uid, old_size, size_diff)

        if process_quota:
            self.process_quotas([key])

        if new_key is not None and new_key != key:
            self._allocations[new_key] = self._allocations.pop(key)
            self._proc_sizes[new_key] = self._proc_sizes.pop(key, 0)
            self._hold_sizes[new_key] = self._hold_sizes.pop(key, 0)

        if quota_size is not None and quota_size < old_size:
            if handle_shrink:
                self._process_requests()
        return old_size, size_diff

    def _process_requests(self):
        """
        Process quota requests in the queue
        """
        removed = []
        for k, req in self._requests.items():
            if self._has_space(req.delta):
                alter_allocation = self.alter_allocations if req.multiple else self.alter_allocation
                alter_allocation(k, req.req_size, handle_shrink=False, allocate=True,
                                 process_quota=req.process_quota)
                req.event.set()
                removed.append(k)
            else:
                # Quota left cannot satisfy the next request, we quit
                break
        for k in removed:
            self._requests.pop(k, None)


class MemQuotaActor(QuotaActor):
    """
    Actor handling worker memory quota
    """
    def __init__(self, total_size, overall_size=None, refresh_time=None):
        super().__init__(total_size)
        self._overall_size = overall_size or total_size
        self._last_memory_available = 0
        self._refresh_time = refresh_time or 10

    async def __post_create__(self):
        self.ref().update_mem_stats.tell_delay(delay=self._refresh_time)

    def update_mem_stats(self):
        """
        Refresh memory usage
        """
        cur_mem_available = mars_resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            self._process_requests()
        self._last_memory_available = cur_mem_available
        self.ref().update_mem_stats.tell_delay(delay=self._refresh_time)

    def _has_space(self, delta):
        mem_stats = mars_resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._overall_size) \
            - self._total_proc
        if max(delta, 0) >= available_size:
            logger.warning('%s met hard memory limitation: request %d, available %d, hard limit %d',
                           self.uid, delta, available_size, self._overall_size)
            return False
        return super()._has_space(delta)

    def _log_allocate(self, msg, *args, **kwargs):
        mem_stats = mars_resource.virtual_memory()
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
