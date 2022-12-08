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

from .... import oscar as mo
from ....typing import BandType
from ....lib.aio import alru_cache
from ...cluster.api import ClusterAPI
from ...cluster.core import NodeRole, NodeStatus
from ..errors import NoAvailableBand

logger = logging.getLogger(__name__)


class AutoscalerActor(mo.Actor):
    def __init__(self, autoscale_conf: Dict[str, Any]):
        self._enabled = autoscale_conf.get("enabled", False)
        self._autoscale_conf = autoscale_conf
        self._cluster_api = None
        self.queueing_refs = dict()
        self.global_resource_ref = None
        self._dynamic_workers: Set[str] = set()
        self._autoscale_in_disable_counter = 0

    async def __post_create__(self):
        strategy = self._autoscale_conf.get("strategy")
        if strategy:  # pragma: no cover
            module, name = strategy.rsplit(".", 1)
            strategy_cls = getattr(importlib.import_module(module), name)
        else:
            strategy_cls = PendingTaskBacklogStrategy
        from ..supervisor import GlobalResourceManagerActor

        self.global_resource_ref = await mo.actor_ref(
            GlobalResourceManagerActor.default_uid(), address=self.address
        )
        self._cluster_api = await ClusterAPI.create(self.address)
        self._strategy = await strategy_cls.create(self._autoscale_conf, self)
        if self._enabled:
            logger.info(f"Auto scale strategy %s started", self._strategy)
            await self._strategy.start()

    async def __pre_destroy__(self):
        if self._enabled:
            await self._strategy.stop()

    async def register_session(self, session_id: str, address: str):
        from .queueing import SubtaskQueueingActor

        self.queueing_refs[session_id] = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(session_id), address=address
        )

    async def unregister_session(self, session_id: str):
        self.queueing_refs.pop(session_id, None)

    async def request_worker(
        self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None
    ) -> str:
        start_time = time.time()
        worker_address = await self._cluster_api.request_worker(
            worker_cpu, worker_mem, timeout
        )
        if worker_address:
            self._dynamic_workers.add(worker_address)
            logger.warning(
                "Requested new worker %s in %.4f seconds, current dynamic worker nums is %s",
                worker_address,
                time.time() - start_time,
                self.get_dynamic_worker_nums(),
            )
            return worker_address
        else:
            logger.warning(
                "Request worker with resource %s failed in %.4f seconds.",
                dict(worker_cpu=worker_cpu, worker_mem=worker_mem),
                time.time() - start_time,
            )

    async def disable_autoscale_in(self):
        self._autoscale_in_disable_counter += 1
        if self._enabled:
            logger.info("Disabled autoscale_in")

    async def try_enable_autoscale_in(self):
        self._autoscale_in_disable_counter -= 1
        if self._autoscale_in_disable_counter == 0 and self._enabled:
            logger.info("Enabled autoscale_in")

    async def release_workers(self, addresses: List[str]) -> List[str]:
        """
        Release a group of worker nodes.
        Parameters
        ----------
        addresses : List[str]
            The addresses of the specified node.
        """
        if self._autoscale_in_disable_counter > 0:
            return []
        workers_bands = {
            address: await self.get_worker_bands(address) for address in addresses
        }
        logger.info(
            "Start to release workers %s which have bands %s.",
            addresses,
            workers_bands,
        )
        for address in addresses:
            await self._cluster_api.set_node_status(
                node=address, role=NodeRole.WORKER, status=NodeStatus.STOPPING
            )
        # Ensure global_slot_manager get latest bands timely, so that we can invoke `wait_band_idle`
        # to ensure there won't be new tasks scheduled to the stopping worker.
        await self.global_resource_ref.refresh_bands()
        excluded_bands = set(b for bands in workers_bands.values() for b in bands)

        async def release_worker(address):
            logger.info("Start to release worker %s.", address)
            worker_bands = workers_bands[address]
            await asyncio.gather(
                *[
                    self.global_resource_ref.wait_band_idle(band)
                    for band in worker_bands
                ]
            )
            await self._migrate_data_of_bands(worker_bands, excluded_bands)
            await self._cluster_api.release_worker(address)
            self._dynamic_workers.remove(address)
            logger.info("Released worker %s.", address)

        # Release workers one by one to ensure others workers which the current is moving data to
        # is not being releasing.
        for address in addresses:
            await release_worker(address)
        return addresses

    def get_dynamic_workers(self) -> Set[str]:
        return self._dynamic_workers

    def get_dynamic_worker_nums(self) -> int:
        return len(self._dynamic_workers)

    async def get_worker_bands(self, worker_address) -> List[BandType]:
        node_info = (
            await self._cluster_api.get_nodes_info(
                [worker_address], resource=True, exclude_statuses=set()
            )
        )[worker_address]
        return [
            (worker_address, resource_type)
            for resource_type in node_info["resource"].keys()
        ]

    async def _migrate_data_of_bands(
        self, bands: List[BandType], excluded_bands: Set[BandType]
    ):
        """Move data from `bands` to other available bands"""
        session_ids = list(self.queueing_refs.keys())
        for session_id in session_ids:
            from ...meta import MetaAPI

            meta_api = await MetaAPI.create(session_id, self.address)

            batch_fetch, batch_delete = defaultdict(list), defaultdict(list)
            batch_add_chunk_bands, batch_remove_chunk_bands = [], []
            for src_band in bands:
                band_data_keys = await meta_api.get_band_chunks(src_band)
                for data_key in band_data_keys:
                    dest_band = await self._select_target_band(
                        src_band, data_key, excluded_bands
                    )
                    logger.debug(
                        "Move chunk % from band %s to band %s.",
                        data_key,
                        src_band,
                        dest_band,
                    )
                    dest_storage_api = await self._get_storage_api(
                        session_id, dest_band[0]
                    )
                    # For ray backend, there will only be meta update rather than data transfer
                    batch_fetch[dest_storage_api].append(
                        dest_storage_api.fetch.delay(
                            data_key, band_name=src_band[1], remote_address=src_band[0]
                        )
                    )
                    src_storage_api = await self._get_storage_api(
                        session_id, src_band[0]
                    )
                    batch_delete[src_storage_api].append(
                        src_storage_api.delete.delay(data_key)
                    )
                    batch_add_chunk_bands.append(
                        meta_api.add_chunk_bands.delay(data_key, [dest_band])
                    )
                    batch_remove_chunk_bands.append(
                        meta_api.remove_chunk_bands.delay(data_key, [src_band])
                    )
            await asyncio.gather(
                *[api.fetch.batch(*fetches) for api, fetches in batch_fetch.items()]
            )
            await meta_api.add_chunk_bands.batch(*batch_add_chunk_bands)
            await meta_api.remove_chunk_bands.batch(*batch_remove_chunk_bands)
            await asyncio.gather(
                *[api.delete.batch(*deletes) for api, deletes in batch_delete.items()]
            )

    async def _select_target_band(
        self, band: BandType, data_key: str, excluded_bands: Set[BandType]
    ):
        all_bands = await self._cluster_api.get_all_bands()
        bands = list(
            b
            for b in all_bands.keys()
            if (b[1] == band[1] and b != band and b not in excluded_bands)
        )
        if not bands:  # pragma: no cover
            raise NoAvailableBand(
                f"No bands to migrate data to, "
                f"all available bands is {all_bands}, "
                f"current band is {band}, "
                f"excluded bands are {excluded_bands}."
            )
        # TODO select band based on remaining store space size of other bands
        return random.choice(bands)

    @alru_cache(cache_exceptions=False)
    async def _get_storage_api(self, session_id: str, address: str):
        from ...storage import StorageAPI

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
        self._scheduler_check_interval = autoscale_conf.get(
            "scheduler_check_interval", 1
        )
        self._scheduler_backlog_timeout = autoscale_conf.get(
            "scheduler_backlog_timeout", 20
        )
        self._sustained_scheduler_backlog_timeout = autoscale_conf.get(
            "sustained_scheduler_backlog_timeout", self._scheduler_backlog_timeout
        )
        # Make worker_idle_timeout greater than scheduler_backlog_timeout to
        # avoid cluster fluctuate back and forth。
        self._worker_idle_timeout = autoscale_conf.get(
            "worker_idle_timeout", 2 * self._scheduler_backlog_timeout
        )
        self._min_workers = autoscale_conf.get("min_workers", 1)
        assert self._min_workers >= 1, "Mars need at least 1 worker."
        self._max_workers = autoscale_conf.get("max_workers", 100)
        self._task = None

    @classmethod
    async def create(cls, autoscale_conf: Dict[str, Any], autoscaler):
        return cls(autoscale_conf, autoscaler)

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            delta = self._min_workers - self._autoscaler.get_dynamic_worker_nums()
            while delta > 0:
                logger.info(f"Start to request %s initial workers.", delta)
                initial_worker_addresses = await asyncio.gather(
                    *[self._autoscaler.request_worker() for _ in range(delta)]
                )
                initial_worker_addresses = [
                    addr for addr in initial_worker_addresses if addr is not None
                ]
                logger.info(
                    f"Requested %s initial workers %s",
                    len(initial_worker_addresses),
                    initial_worker_addresses,
                )
                delta = self._min_workers - self._autoscaler.get_dynamic_worker_nums()
            while True:
                await asyncio.sleep(self._scheduler_check_interval)
                await self._run_round()
        except asyncio.CancelledError:  # pragma: no cover
            logger.info("Canceled pending task backlog strategy.")
        except Exception as e:  # pragma: no cover
            logger.exception("Exception occurred when try to auto scale")
            raise e

    async def _run_round(self):
        queueing_refs = list(self._autoscaler.queueing_refs.values())
        if any([await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs]):
            await self._scale_out(queueing_refs)
        else:
            await self._scale_in()

    async def _scale_out(self, queueing_refs):
        logger.info(
            "Try to scale out, current dynamic workers %s",
            self._autoscaler.get_dynamic_worker_nums(),
        )
        start_time = time.time()
        while not await self._autoscaler.request_worker():
            logger.warning(
                "Request worker failed, wait %s seconds and retry.",
                self._scheduler_check_interval,
            )
            await asyncio.sleep(self._scheduler_check_interval)
        await asyncio.sleep(self._scheduler_backlog_timeout)
        rnd = 1
        while any(
            [await queueing_ref.all_bands_busy() for queueing_ref in queueing_refs]
        ):
            worker_num = 2**rnd
            if (
                self._autoscaler.get_dynamic_worker_nums() + worker_num
                > self._max_workers
            ):
                worker_num = (
                    self._max_workers - self._autoscaler.get_dynamic_worker_nums()
                )
            while set(
                await asyncio.gather(
                    *[self._autoscaler.request_worker() for _ in range(worker_num)]
                )
            ) == {None}:
                logger.warning(
                    "Request %s workers all failed, wait %s seconds and retry.",
                    worker_num,
                    self._scheduler_check_interval,
                )
                await asyncio.sleep(self._scheduler_check_interval)
            rnd += 1
            await asyncio.sleep(self._sustained_scheduler_backlog_timeout)
        logger.info(
            "Scale out finished in %s round, took %s seconds, current dynamic workers %s",
            rnd,
            time.time() - start_time,
            self._autoscaler.get_dynamic_worker_nums(),
        )

    async def _scale_in(self):
        idle_bands = set(
            await self._autoscaler.global_resource_ref.get_idle_bands(
                self._worker_idle_timeout
            )
        )
        # exclude non-dynamic created workers and ensure all bands of the worker are idle
        idle_bands = {
            band
            for band in idle_bands
            if band[0] in self._autoscaler.get_dynamic_workers()
            and idle_bands.issuperset(
                set(await self._autoscaler.get_worker_bands(band[0]))
            )
        }
        worker_addresses = set(band[0] for band in idle_bands)
        if worker_addresses:
            logger.debug(
                "Bands %s of workers % has been idle for as least %s seconds.",
                idle_bands,
                worker_addresses,
                self._worker_idle_timeout,
            )
            while (
                worker_addresses
                and self._autoscaler.get_dynamic_worker_nums() - len(worker_addresses)
                < self._min_workers
            ):
                worker_address = worker_addresses.pop()
                logger.debug(
                    "Skip offline idle worker %s to keep at least %s dynamic workers. "
                    "Current total dynamic workers is %s.",
                    worker_address,
                    self._min_workers,
                    self._autoscaler.get_dynamic_worker_nums(),
                )
                idle_bands.difference_update(
                    set(await self._autoscaler.get_worker_bands(worker_address))
                )
        if worker_addresses:
            start_time = time.time()
            logger.info(
                "Try to offline idle workers %s with bands %s.",
                worker_addresses,
                idle_bands,
            )
            try:
                worker_addresses = await self._autoscaler.release_workers(
                    worker_addresses
                )
                logger.info(
                    "Finished offline workers %s in %.4f seconds",
                    worker_addresses,
                    time.time() - start_time,
                )
            except NoAvailableBand as e:  # pragma: no cover
                logger.warning(
                    "No enough bands, offline workers %s failed with exception %s.",
                    worker_addresses,
                    e,
                )

    async def stop(self):
        self._task.cancel()
        await self._task
