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
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Dict, Optional, Any

from ....lib.aio import alru_cache
from ...cluster.api import ClusterAPI
from ...core import BandType
from .... import oscar as mo
from mars.services.cluster.core import NodeRole, NodeStatus

import ray
logging.basicConfig(format=ray.ray_constants.LOGGER_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoscalerActor(mo.Actor):
    __slots__ = '_config'

    def __init__(self, autoscale_conf: Dict[str, Any]):
        self._enabled = autoscale_conf.get('enabled', False)
        self._autoscale_conf = autoscale_conf
        self._cluster_api = None
        self.queueing_refs = dict()
        self.global_slot_ref = None
        self.band_total_slots = None
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

    async def register_session(self, session_id: str, address: str):
        from .queueing import SubtaskQueueingActor
        self.queueing_refs[session_id] = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(session_id), address=address)

    async def unregister_session(self, session_id: str):
        self.queueing_refs.pop(session_id, None)

    async def request_worker_node(
            self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None) -> str:
        start_time = time.time()
        worker_address = await self._cluster_api.request_worker_node(worker_cpu, worker_mem, timeout)
        self._dynamic_workers.add(worker_address)
        logger.info("Requested new workers %s in %.4f seconds, current dynamic worker nums is %s",
                    worker_address, time.time() - start_time, self.get_dynamic_worker_nums())
        return worker_address

    async def release_worker_node(self, address: str):
        """
        Release a worker node.
        Parameters
        ----------
        address : str
            The address of the specified node.
        """
        bands = await self.get_worker_bands(address)
        logger.info("Start to release worker %s which has bands %s.", address, bands)
        start_time = time.time()
        await self._cluster_api.set_node_status(
            node=address, role=NodeRole.WORKER, status=NodeStatus.STOPPING)
        # Ensure global_slot_manager get latest bands timely, so that we can invoke `is_band_idle`
        # to ensure there won't be new tasks scheduled to the stopping worker.
        await self.global_slot_ref.refresh_bands()
        for band in bands:
            while not await self.global_slot_ref.is_band_idle(band):
                await asyncio.sleep(0.1)
        await self.migrate_data_of_bands(bands)
        await self._cluster_api.release_worker_node(address)
        self._dynamic_workers.remove(address)
        logger.info("Release worker %s succeeds in %.4f seconds.", address, time.time() - start_time)

    def get_dynamic_workers(self) -> Set[str]:
        return self._dynamic_workers

    def get_dynamic_worker_nums(self) -> int:
        return len(self._dynamic_workers)

    async def get_worker_bands(self, worker_address) -> List[BandType]:
        node_info = (await self._cluster_api.get_nodes_info(
            [worker_address], resource=True, exclude_statuses=set()))[worker_address]
        return [(worker_address, resource_type) for resource_type in node_info['resource'].keys()]

    async def migrate_data_of_bands(self, bands: List[BandType]):
        """Move data from `bands` to other available bands"""
        session_ids = list(self.queueing_refs.keys())
        for session_id in session_ids:
            from mars.services.meta import MetaAPI
            meta_api = await MetaAPI.create(session_id, self.address)
            for src_band in bands:
                band_data_keys = await meta_api.get_band_chunks(src_band)
                for data_key in band_data_keys:
                    dest_band = self._select_target_band(src_band, data_key)
                    # For ray backend, there will only be meta update rather than data transfer
                    await (await self._get_storage_api(session_id, dest_band[0])).fetch(
                        data_key, band_name=src_band[1], dest_address=src_band[0])
                    await (await self._get_storage_api(session_id, src_band[0])).delete(data_key)
                    chunk_bands = (await meta_api.get_chunk_meta(data_key, fields=['bands'])).get('bands')
                    chunk_bands.remove(src_band)
                    chunk_bands.append(dest_band)
                    await meta_api.set_chunk_bands(data_key, chunk_bands)

    def _select_target_band(self, band: BandType, data_key: str):
        bands = list(b for b in self.band_total_slots.keys() if b[1] == band[1] and b != band)
        # TODO select band based on remaining store space size of other bands
        return random.choice(bands)

    @alru_cache(cache_exceptions=False)
    async def _get_storage_api(self, session_id: str, address: str):
        from mars.services.storage import StorageAPI
        return await StorageAPI.create(session_id, address)


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
        self._scheduler_check_interval = autoscale_conf.get('scheduler_check_interval', 1)
        self._scheduler_backlog_timeout = autoscale_conf.get('scheduler_backlog_timeout', 10)
        self._sustained_scheduler_backlog_timeout = autoscale_conf.get(
            'sustained_scheduler_backlog_timeout', self._scheduler_backlog_timeout)
        self._worker_idle_timeout = autoscale_conf.get('worker_idle_timeout', 10)
        self._min_workers = autoscale_conf.get('min_workers', 1)
        assert self._min_workers >= 1, 'Mars need at least 1 worker.'
        self._max_workers = autoscale_conf.get('max_workers', 100)
        self._task = None

    @classmethod
    async def create(cls, autoscale_conf: Dict[str, Any], autoscaler):
        return cls(autoscale_conf, autoscaler)

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        if self._autoscaler.get_dynamic_worker_nums() < self._min_workers:
            logger.info(f'Start to request %s initial workers.', self._min_workers)
            initial_worker_addresses = await asyncio.gather(*[
                self._autoscaler.request_worker_node() for _ in range(
                    self._min_workers - self._autoscaler.get_dynamic_worker_nums())])
            logger.info(f'Finished requesting %s initial workers %s',
                        len(initial_worker_addresses), initial_worker_addresses)
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
        await self._autoscaler.request_worker_node()
        await asyncio.sleep(self._scheduler_backlog_timeout)
        rnd = 1
        while any([await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs]):
            worker_num = 2 ** rnd
            if self._autoscaler.get_dynamic_worker_nums() + worker_num > self._max_workers:
                worker_num = self._max_workers - self._autoscaler.get_dynamic_worker_nums()
            await asyncio.gather(
                *[self._autoscaler.request_worker_node() for _ in range(worker_num)])
            rnd += 1
            await asyncio.sleep(self._sustained_scheduler_backlog_timeout)
        logger.info("Scale out finished in %s round, took %s seconds, current dynamic workers %s",
                    rnd, time.time() - start_time, self._autoscaler.get_dynamic_worker_nums())

    async def _scale_in(self):
        idle_bands = set(await self._autoscaler.global_slot_ref.get_idle_bands(self._worker_idle_timeout))
        # ensure all bands of the worker are idle
        idle_bands = [band for band in idle_bands if idle_bands.issuperset(
            set(await self._autoscaler.get_worker_bands(band[0])))]
        # exclude non-dynamic created workers
        idle_bands = set(band for band in idle_bands if band[0] in self._autoscaler.get_dynamic_workers())
        worker_addresses = set(band[0] for band in idle_bands)
        if worker_addresses:
            logger.debug("Bands %s of workers % has been idle for as least %s seconds.",
                         idle_bands, worker_addresses, self._worker_idle_timeout)
            while worker_addresses and \
                    self._autoscaler.get_dynamic_worker_nums() - len(worker_addresses) < self._min_workers:
                worker_address = worker_addresses.pop()
                logger.debug("Skip offline idle worker %s to keep at least %s dynamic workers. "
                             "Current total dynamic workers is %s.",
                             worker_address, self._min_workers, self._autoscaler.get_dynamic_worker_nums())
                idle_bands.difference_update(set(await self._autoscaler.get_worker_bands(worker_address)))
        if worker_addresses:
            start_time = time.time()
            logger.info("Try to offline idle workers %s with bands %s.", worker_addresses, idle_bands)
            # Release workers one by one to ensure others workers which the current is moving data to
            # is not being releasing.
            for worker_address in worker_addresses:
                await self._autoscaler.release_worker_node(worker_address)
            logger.info('Finished offline workers %s in %.4f seconds', worker_addresses, time.time() - start_time)

    async def stop(self):
        self._task.cancel()
