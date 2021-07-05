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
import importlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Any

from ...cluster.api import ClusterAPI
from ...core import BandType
from ....lib.aio import alru_cache
from .... import oscar as mo

logger = logging.getLogger(__name__)


class AutoscalerActor(mo.Actor):
    __slots__ = '_config'

    def __init__(self, autoscale_conf: Dict[str, Any]):
        self._enabled = autoscale_conf.get('enabled', False)
        self._autoscale_conf = autoscale_conf
        self.cluster_api = None
        self.queueing_refs = dict()
        self.global_slot_ref = None
        self.band_total_slots = None
        self.worker_bands = defaultdict(list)

    async def __post_create__(self):
        strategy = self._autoscale_conf.get('strategy')
        if strategy:
            module, name = strategy.rsplit('.', 1)
            strategy_cls = getattr(importlib.import_module(module), name)
        else:
            strategy_cls = PendingTaskBacklogStrategy
        from ..supervisor import GlobalSlotManagerActor
        self.global_slot_ref = await mo.actor_ref(
            GlobalSlotManagerActor.default_uid(), address=self.address)
        self.cluster_api = await ClusterAPI.create(self.address)
        self._strategy = await strategy_cls.create(self._autoscale_conf, self)
        await self._strategy.start()

        async def watch_slots():
            while True:
                self.band_total_slots = await self.cluster_api.get_available_bands(watch=True)
                worker_bands = {}
                for band in self.band_total_slots.keys():
                    worker_address, resource_type = band
                    worker_bands[worker_address].append(band)
                self.worker_bands = worker_bands

        self._band_watch_task = asyncio.create_task(watch_slots())

    async def __pre_destroy__(self):
        await self._strategy.stop()
        self._band_watch_task.cancel()

    async def register_session(self, session_id: str, address: str):
        from .queueing import SubtaskQueueingActor
        self.queueing_refs[session_id] = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(session_id), address=address)

    async def unregister_session(self, session_id: str):
        self.queueing_refs.pop(session_id, None)

    async def is_worker_idle(self, band):
        pass

    def get_worker_bands(self, worker_address):
        return self.worker_bands[worker_address]


class AbstractScaleStrategy(ABC):

    @classmethod
    @abstractmethod
    async def create(cls, autoscale_conf: Dict[str, Any], autoscaler):
        """Create a autoscale strategy which will decide when to scale in/.out"""

    @abstractmethod
    async def start(self):
        """Start auto scale"""

    @abstractmethod
    async def stop(self):
        """Stop auto scale"""


class PendingTaskBacklogStrategy(AbstractScaleStrategy):
    _task: Optional[asyncio.Task]

    def __init__(self, autoscale_conf: Dict[str, Any], autoscaler):
        self._autoscaler = autoscaler
        self._scheduler_check_interval = int(autoscale_conf.get(
            'scheduler_check_interval', 1))
        self._scheduler_backlog_timeout = int(autoscale_conf.get(
            'scheduler_backlog_timeout', 10))
        self._sustained_scheduler_backlog_timeout = int(autoscale_conf.get(
            'sustained_scheduler_backlog_timeout', self._scheduler_backlog_timeout))
        self._worker_idle_timeout = int(autoscale_conf.get('worker_idle_timeout', 10))
        self._min_workers = int(autoscale_conf.get('min_workers', 1))
        self._max_workers = int(autoscale_conf.get('max_workers', 100))
        self._dynamic_worker_count = 0
        self._task = None

    @classmethod
    async def create(cls, autoscale_conf: Dict[str, Any], autoscaler):
        return cls(autoscale_conf, autoscaler)

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        await asyncio.sleep(self._scheduler_check_interval)
        queueing_refs = list(self._autoscaler.queueing_refs)
        if any(await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs):
            await self._scale_out(queueing_refs)
        else:
            # try to scale in
            idle_bands = set(await self._autoscaler.global_slot_ref.get_idle_bands(self._worker_idle_timeout))
            idle_bands = [band for band in idle_bands if idle_bands.issuperset(
                set(self._autoscaler.get_worker_bands(band[0])))]
            self._autoscaler.get_worker_bands()
            if idle_bands:
                await self._scale_in(idle_bands)

    async def _scale_out(self, queueing_refs):
        await self._autoscaler.cluster_api.request_worker_node()
        await asyncio.sleep(self._scheduler_backlog_timeout)
        rnd = 1
        while any(await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs):
            worker_num = 2 ** rnd
            if self._dynamic_worker_count + worker_num > self._max_workers:
                worker_num = self._max_workers - self._dynamic_worker_count
            await asyncio.gather(*[await self._autoscaler.cluster_api.request_worker_node()
                                   for _ in range(worker_num)])
            self._dynamic_worker_count += worker_num
            rnd += 1
            await asyncio.sleep(self._sustained_scheduler_backlog_timeout)

    async def _scale_in(self, idle_bands):
        for band in idle_bands:
            await self._autoscaler.global_slot_ref.add_to_blocklist(band)
        for band in idle_bands:
            execution_ref = await self._get_execution_ref(band)
            while not await execution_ref.is_worker_idle():
                await asyncio.sleep(0.1)
        # TODO [data migration]
        # TODO [update meta]
        worker_addresses = set(band[0] for band in idle_bands)
        # release workers
        await asyncio.gather(*[await self._autoscaler.cluster_api.release_worker_node(worker_address)
                               for worker_address in worker_addresses])

    @alru_cache(maxsize=10000)
    async def _get_execution_ref(self, band: BandType):
        from ..worker.execution import SubtaskExecutionActor
        return await mo.actor_ref(SubtaskExecutionActor.default_uid(), address=band[0])

    async def stop(self):
        self._task.cancel()
