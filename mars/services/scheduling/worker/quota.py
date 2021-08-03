# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from .... import oscar as mo
from .... import resource as mars_resource
from ....typing import BandType
from ...cluster import QuotaInfo

logger = logging.getLogger(__name__)

QuotaDumpType = namedtuple('QuotaDumpType', 'allocations requests hold_sizes')


@dataclass
class QuotaRequest:
    req_size: Tuple
    delta: int
    req_time: float
    event: asyncio.Event


class QuotaActor(mo.Actor):
    @classmethod
    def gen_uid(cls, band_name: str):
        return f'{band_name}_quota'

    def __init__(self, band: BandType, quota_size: int, **kw):
        super().__init__()
        self._requests = OrderedDict()

        self._cluster_api = None

        self._band = band
        self._band_name = band[1]

        self._quota_size = quota_size
        self._allocations = dict()
        self._total_allocated = 0

        self._hold_sizes = dict()
        self._total_hold = 0

        if kw:  # pragma: no cover
            logger.warning('Keywords for QuotaActor %r not used', list(kw.keys()))

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        try:
            self._cluster_api = await ClusterAPI.create(self.address)
            self._report_quota_info()
        except mo.ActorNotExist:
            pass

    async def _has_space(self, delta: int):
        return self._total_allocated + delta <= self._quota_size

    def _log_allocate(self, msg: str, *args, **kwargs):
        args += (self._total_allocated, self._quota_size)
        logger.debug(msg + ' Allocated: %s, Total size: %s', *args, **kwargs)

    def _report_quota_info(self):
        if self._cluster_api is not None:
            quota_info = QuotaInfo(
                quota_size=self._quota_size,
                allocated_size=self._total_allocated,
                hold_size=self._total_hold
            )
            asyncio.create_task(self._cluster_api.set_band_quota_info(
                self._band_name, quota_info))

    async def request_batch_quota(self, batch: Dict):
        """
        Request for resources in a batch
        :param batch: the request dict in form {request_key: request_size, ...}
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
        quota_sizes = tuple(tp[1] for tp in sorted_req)
        delta = sum(v - self._allocations.get(k, 0) for k, v in batch.items())

        # if all requested and allocation can still be applied, apply directly
        if all_allocated and await self._has_space(delta):
            self._log_allocate('Quota request %r already allocated.', batch)
            return

        if delta > self._quota_size:
            raise ValueError(f'Cannot allocate quota size {delta} '
                             f'larger than total capacity {self._quota_size}.')

        if keys in self._requests:
            event = self._requests[keys].event
        else:
            has_space = await self._has_space(delta)
            if has_space and not self._requests:
                # if no previous requests, we can apply directly
                self._log_allocate('Quota request met for key %r on %s.', keys, self.uid)
                await self.alter_allocations(keys, quota_sizes, allocate=True)
                return
            else:
                # current free space cannot satisfy the request, the request is queued
                if not has_space:
                    self._log_allocate('Quota request unmet for key %r on %s.', keys, self.uid)
                else:
                    self._log_allocate('Quota request queued for key %r on %s.', keys, self.uid)
                event = asyncio.Event()
                quota_request = QuotaRequest(quota_sizes, delta, time.time(), event)
                if keys not in self._requests:
                    self._requests[keys] = quota_request

        async def waiter():
            try:
                await event.wait()
            except asyncio.CancelledError as ex:
                await self.ref().remove_requests.tell(keys)
                raise ex
        return waiter()

    async def remove_requests(self, keys: Tuple):
        self._requests.pop(keys, None)
        await self._process_requests()

    def hold_quotas(self, keys: Tuple):
        """
        Mark request quota as already been hold

        Parameters
        ----------
        keys : Tuple
            request keys
        """
        for key in keys:
            try:
                alloc_size = self._allocations[key]
            except KeyError:
                continue
            self._total_hold += alloc_size - self._hold_sizes.get(key, 0)
            self._hold_sizes[key] = alloc_size

    async def release_quotas(self, keys: Tuple):
        """
        Release allocated quota in batch

        Parameters
        ----------
        keys : Tuple
            request keys
        """
        total_alloc_size = 0

        for key in keys:
            try:
                alloc_size = self._allocations.pop(key)
                total_alloc_size += alloc_size
            except KeyError:
                continue
            self._total_hold -= self._hold_sizes.pop(key, 0)

        self._total_allocated -= total_alloc_size
        if total_alloc_size:
            await self._process_requests()

            self._report_quota_info()
            self._log_allocate('Quota keys %s released on %s.', keys, self.uid)

    def dump_data(self):
        return QuotaDumpType(self._allocations, self._requests, self._hold_sizes)

    def get_allocated_size(self):
        # get total allocated size, for debug purpose
        return self._total_allocated

    async def alter_allocations(self, keys: Tuple, quota_sizes: Tuple,
                                handle_shrink: bool = True, allocate: bool = False):
        """
        Alter multiple requests

        Parameters
        ----------
        keys : Tuple
            keys to update
        quota_sizes : Tuple
            new quota sizes, if None, no changes will be made
        handle_shrink : bool
            if True and the quota size less than the original, process requests in the queue
        allocate : bool
            if True, will allocate resources for new items
        """
        quota_sizes = quota_sizes or itertools.repeat(None)
        total_old_size, total_diff = 0, 0
        for k, s in zip(keys, quota_sizes):
            old_size = self._allocations.get(k, 0)
            size_diff = 0

            if not allocate and k not in self._allocations:
                total_old_size += old_size
                continue

            if s != old_size:
                s = int(s)
                size_diff = s - old_size
                self._total_allocated += size_diff
                self._allocations[k] = s
                try:
                    self._total_hold += s - self._hold_sizes[k]
                    self._hold_sizes[k] = s
                except KeyError:
                    pass

            total_old_size += old_size
            total_diff += size_diff
        if handle_shrink and total_diff < 0:
            await self._process_requests()

        self._report_quota_info()
        self._log_allocate('Quota keys %r applied on %s. Total old Size: %s, Total diff: %s,',
                           keys, self.uid, total_old_size, total_diff)

    async def _process_requests(self):
        """
        Process quota requests in the queue
        """
        removed = []
        for k, req in self._requests.items():
            if await self._has_space(req.delta):
                await self.alter_allocations(k, req.req_size, handle_shrink=False, allocate=True)
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
    def __init__(self, band: BandType, quota_size: int, hard_limit: int = None,
                 refresh_time: Union[int, float] = None,
                 enable_kill_slot: bool = True):
        super().__init__(band, quota_size)
        self._hard_limit = hard_limit
        self._last_memory_available = 0
        self._refresh_time = refresh_time or 1

        self._enable_kill_slot = enable_kill_slot

        self._stat_refresh_task = None
        self._slot_manager_ref = None

    async def __post_create__(self):
        await super().__post_create__()
        self._stat_refresh_task = self.ref().update_mem_stats.tell_delay(delay=self._refresh_time)

        from .workerslot import BandSlotManagerActor
        try:
            self._slot_manager_ref = await mo.actor_ref(
                uid=BandSlotManagerActor.gen_uid(self._band[1]), address=self.address)
        except mo.ActorNotExist:  # pragma: no cover
            pass

    async def __pre_destroy__(self):
        self._stat_refresh_task.cancel()

    async def update_mem_stats(self):
        """
        Refresh memory usage
        """
        cur_mem_available = mars_resource.virtual_memory().available
        if cur_mem_available > self._last_memory_available:
            # memory usage reduced: try reallocate existing requests
            await self._process_requests()
        self._last_memory_available = cur_mem_available
        self._report_quota_info()
        self.ref().update_mem_stats.tell_delay(delay=self._refresh_time)

    async def _has_space(self, delta: int):
        if self._hard_limit is None:
            return await super()._has_space(delta)

        mem_stats = mars_resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._hard_limit) \
            - (self._total_allocated - self._total_hold)
        if max(delta, 0) >= available_size:
            logger.warning('%s met hard memory limitation: request %d, available %d, hard limit %d',
                           self.uid, delta, available_size, self._hard_limit)

            if self._enable_kill_slot and self._slot_manager_ref is not None:
                logger.info('Restarting free slots to obtain more memory')
                await self._slot_manager_ref.restart_free_slots()
            return False
        return await super()._has_space(delta)

    def _log_allocate(self, msg: str, *args, **kwargs):  # pragma: no cover
        if logger.getEffectiveLevel() > logging.DEBUG:
            return

        if self._hard_limit is None:
            return super()._log_allocate(msg, *args, **kwargs)

        mem_stats = mars_resource.virtual_memory()
        # calc available physical memory
        available_size = mem_stats.available - max(0, mem_stats.total - self._hard_limit) \
            - (self._total_allocated - self._total_hold)
        args += (self._total_allocated, self._quota_size, mem_stats.available, available_size,
                 self._hard_limit, self._total_hold)

        logger.debug(
            msg + ' Allocated: %s, Quota size: %s, Phy available: %s, Hard available: %s,'
                  ' Hard limit: %s, Holding: %s',
            *args, **kwargs
        )


class WorkerQuotaManagerActor(mo.Actor):
    def __init__(self, default_config: Dict, band_configs: Optional[Dict] = None):
        self._cluster_api = None
        self._default_config = default_config
        self._band_configs = band_configs or dict()

        self._band_quota_refs = dict()  # type: Dict[str, mo.ActorRef]

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        self._cluster_api = await ClusterAPI.create(self.address)

        band_to_slots = await self._cluster_api.get_bands()
        for band in band_to_slots.keys():
            band_config = self._band_configs.get(band[1], self._default_config)
            self._band_quota_refs[band] = await mo.create_actor(
                MemQuotaActor, band, **band_config,
                uid=MemQuotaActor.gen_uid(band[1]),
                address=self.address)

    async def __pre_destroy__(self):
        await asyncio.gather(*[
            mo.destroy_actor(ref) for ref in self._band_quota_refs.values()
        ])
