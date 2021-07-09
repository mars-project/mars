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
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Dict, Optional, Any

from ...cluster.api import ClusterAPI
from ...core import BandType
from .... import oscar as mo

import ray
logging.basicConfig(format=ray.ray_constants.LOGGER_FORMAT,
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class AutoscalerActor(mo.Actor):
    __slots__ = '_config'

    def __init__(self, autoscale_conf: Dict[str, Any]):
        self._enabled = autoscale_conf.get('enabled', False)
        self._migrate_data = autoscale_conf.get('migrate_data', True)
        self._autoscale_conf = autoscale_conf
        self._cluster_api = None
        self.queueing_refs = dict()
        self.global_slot_ref = None
        self.band_total_slots = None
        self.worker_bands = defaultdict(list)
        self._dynamic_workers: Set[str] = set()

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
        self._cluster_api = await ClusterAPI.create(self.address)
        self._strategy = await strategy_cls.create(self._autoscale_conf, self)
        logger.info(f'Auto scale strategy %s started', self._strategy)
        await self._strategy.start()

        async def watch_slots():
            while True:
                self.band_total_slots = await self._cluster_api.get_available_bands(watch=True)
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

    async def request_worker_node(
            self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None) -> str:
        worker_address = await self._cluster_api.request_worker_node(worker_cpu, worker_mem, timeout)
        self._dynamic_workers.add(worker_address)
        return worker_address

    async def release_worker_node(self, address: str):
        await self._cluster_api.release_worker_node(address)
        self._dynamic_workers.remove(address)
    
    def get_dynamic_workers(self) -> Set[str]:
        return self._dynamic_workers

    def get_dynamic_worker_nums(self) -> int:
        return len(self._dynamic_workers)

    def get_worker_bands(self, worker_address) -> List[BandType]:
        return self.worker_bands[worker_address]

    async def migrate_data_of_bands(self, bands: List[BandType]):
        """Move data from `bands` to other available bands"""
        if self._migrate_data:
            raise NotImplementedError
        # TODO [data migration]


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
        self._scheduler_check_interval = float(autoscale_conf.get(
            'scheduler_check_interval', 1))
        self._scheduler_backlog_timeout = float(autoscale_conf.get(
            'scheduler_backlog_timeout', 10))
        self._sustained_scheduler_backlog_timeout = float(autoscale_conf.get(
            'sustained_scheduler_backlog_timeout', self._scheduler_backlog_timeout))
        self._worker_idle_timeout = float(autoscale_conf.get('worker_idle_timeout', 10))
        self._min_workers = int(autoscale_conf.get('min_workers', 1))
        self._max_workers = int(autoscale_conf.get('max_workers', 100))
        self._task = None

    @classmethod
    async def create(cls, autoscale_conf: Dict[str, Any], autoscaler):
        return cls(autoscale_conf, autoscaler)

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        if self._autoscaler.get_dynamic_worker_nums() < self._min_workers:
            logger.info(f'Start to request initial workers to %s', self._min_workers)
            initial_worker_addresses = await asyncio.gather(*[
                self._autoscaler.request_worker_node() for _ in range(
                    self._min_workers - self._autoscaler.get_dynamic_worker_nums())])
            logger.info(f'Finished requesting initial workers %s', initial_worker_addresses)
        while True:
            await asyncio.sleep(self._scheduler_check_interval)
            try:
                await self._run_round()
            except Exception as e:  # pragma: no cover
                logger.exception('Exception occurred when try to auto scale')
                self._task.cancel()
                raise e

    async def _run_round(self):
        queueing_refs = list(self._autoscaler.queueing_refs.values())
        if any([await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs]):
            await self._scale_out(queueing_refs)
        else:
            await self._scale_in()

    async def _scale_out(self, queueing_refs):
        logger.info("Try to scale out, current dynamic workers %s", self._autoscaler.get_dynamic_worker_nums())
        start_time = time.time()
        worker_address = await self._autoscaler.request_worker_node()
        logger.info("Requested new worker %s", worker_address)
        await asyncio.sleep(self._scheduler_backlog_timeout)
        rnd = 1
        while any([await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs]):
            worker_num = 2 ** rnd
            if self._autoscaler.get_dynamic_worker_nums() + worker_num > self._max_workers:
                worker_num = self._max_workers - self._autoscaler.get_dynamic_worker_nums()
            worker_addresses = await asyncio.gather(
                *[self._autoscaler.request_worker_node() for _ in range(worker_num)])
            logger.info("Requested new workers %s, current dynamic workers %s",
                        worker_addresses, self._autoscaler.get_dynamic_worker_nums())
            rnd += 1
            await asyncio.sleep(self._sustained_scheduler_backlog_timeout)
        logger.info("Scale out finished in %s round, took %s seconds, current dynamic workers %s",
                    rnd, time.time() - start_time, self._autoscaler.get_dynamic_worker_nums())

    async def _scale_in(self):
        idle_bands = set(await self._autoscaler.global_slot_ref.get_idle_bands(self._worker_idle_timeout))
        # ensure all bands of the worker are idle
        idle_bands = [band for band in idle_bands if idle_bands.issuperset(
            set(self._autoscaler.get_worker_bands(band[0])))]
        # exclude non-dynamic created workers
        idle_bands = [band for band in idle_bands if band[0] in self._autoscaler.get_dynamic_workers()]
        worker_addresses = set(band[0] for band in idle_bands)
        if idle_bands:
            logger.info("Bands %s of workers % has been idle for as least %s seconds.",
                        idle_bands, worker_addresses, self._worker_idle_timeout)
        while worker_addresses and \
                self._autoscaler.get_dynamic_worker_nums() - len(worker_addresses) < self._min_workers:
            logger.info("Idle workers %s is less than min workers %s. Current total dynamic workers is %s.",
                        len(worker_addresses), self._min_workers, self._autoscaler.get_dynamic_worker_nums())
            worker_address = worker_addresses.pop()
            for band in self._autoscaler.get_worker_bands(worker_address):
                idle_bands.remove(band)
        if worker_addresses:
            logger.info("Try to offline bands %s of workers %s.", idle_bands, worker_addresses)
            await asyncio.gather(*[self._autoscaler.global_slot_ref.add_to_blocklist(band) for band in idle_bands])
            for band in idle_bands:
                while not await self._autoscaler.global_slot_ref.is_band_idle(band):
                    await asyncio.sleep(0.1)
            await self._autoscaler.migrate_data_of_bands(idle_bands)
            # release workers
            await asyncio.gather(*[self._autoscaler.release_worker_node(worker_address)
                                   for worker_address in worker_addresses])
            logger.info('Finished offline workers %s', worker_addresses)

    async def stop(self):
        self._task.cancel()
