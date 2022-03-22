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
import logging
import time
from collections import defaultdict
from typing import List, DefaultDict, Dict, Tuple

from .... import oscar as mo
from ....resource import Resource, ZeroResource
from ....typing import BandType

logger = logging.getLogger(__name__)


class GlobalResourceManagerActor(mo.Actor):
    # {(address, resource_type): {(session_id, subtask_id): Resource(...)}}
    _band_stid_resources: DefaultDict[BandType, Dict[Tuple[str, str], Resource]]
    _band_used_resources: Dict[BandType, Resource]
    _band_total_resources: Dict[BandType, Resource]

    def __init__(self):
        self._band_stid_resources = defaultdict(dict)
        self._band_used_resources = defaultdict(lambda: ZeroResource)
        self._band_idle_start_time = dict()
        self._band_idle_events = dict()
        self._band_total_resources = dict()
        self._cluster_api = None
        self._band_watch_task = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI

        self._cluster_api = await ClusterAPI.create(self.address)

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands():
                old_bands = set(self._band_total_resources.keys())
                await self._refresh_bands(bands)
                new_bands = set(bands.keys()) - old_bands
                for band in new_bands:
                    self._update_band_usage(band, ZeroResource)

        self._band_watch_task = asyncio.create_task(watch_bands())

    async def __pre_destroy__(self):
        self._band_watch_task.cancel()

    async def refresh_bands(self):
        bands = await self._cluster_api.get_all_bands()
        await self._refresh_bands(bands)

    async def _refresh_bands(self, bands):
        # TODO add `num_mem_bytes` after supported report worker memory
        band_total_resources = {}
        for band, slot in bands.items():
            if band[1].startswith("gpu"):
                band_total_resources[band] = Resource(num_gpus=slot)
            elif band[1].startswith("numa"):
                band_total_resources[band] = Resource(num_cpus=slot)
            else:
                raise NotImplementedError(f"Unsupported band type {band}")
        self._band_total_resources = band_total_resources

    @mo.extensible
    async def apply_subtask_resources(
        self,
        band: BandType,
        session_id: str,
        subtask_ids: List[str],
        subtask_resources: List[Resource],
    ) -> List[str]:
        if (
            not self._band_total_resources or band not in self._band_total_resources
        ):  # pragma: no cover
            await self.refresh_bands()
        idx = 0
        # only ready bands will pass
        if band in self._band_total_resources:
            total_resource = self._band_total_resources[band]
            for stid, subtask_resource in zip(subtask_ids, subtask_resources):
                band_used_resource = self._band_used_resources[band]
                if band_used_resource + subtask_resource > total_resource:
                    break
                self._band_stid_resources[band][(session_id, stid)] = subtask_resource
                self._update_band_usage(band, subtask_resource)
                idx += 1
        if idx == 0:
            logger.debug(
                "No resources available, status: %r, request: %r",
                self._band_used_resources,
                subtask_resources,
            )
        return subtask_ids[:idx]

    @mo.extensible
    def update_subtask_resources(
        self, band: BandType, session_id: str, subtask_id: str, resource: Resource
    ):
        session_subtask_id = (session_id, subtask_id)
        subtask_resources = self._band_stid_resources[band]
        if session_subtask_id not in subtask_resources:
            return

        resource_delta = resource - subtask_resources[session_subtask_id]
        subtask_resources[session_subtask_id] = resource
        self._update_band_usage(band, resource_delta)

    @mo.extensible
    def release_subtask_resource(
        self, band: BandType, session_id: str, subtask_id: str
    ):
        # todo ensure slots released when subtasks ends in all means
        resource_delta = self._band_stid_resources[band].pop(
            (session_id, subtask_id), ZeroResource
        )
        self._update_band_usage(band, -resource_delta)

    def _update_band_usage(self, band: BandType, band_usage_delta: Resource):
        self._band_used_resources[band] += band_usage_delta
        # some code path doesn't call `apply_subtask_resources`
        band_total_resource = self._band_total_resources.get(band)
        if (
            band_total_resource is not None
            and self._band_used_resources[band] > band_total_resource
        ):  # pragma: no cover
            raise Exception(
                f"Resource exceed: band used resource {self._band_used_resources[band]} "
                f"band total resource {self._band_total_resources[band]}"
            )
        if self._band_used_resources[band] <= ZeroResource:
            self._band_used_resources.pop(band)
            self._band_idle_start_time[band] = time.time()
            if band in self._band_idle_events:
                self._band_idle_events.pop(band).set()
        else:
            self._band_idle_start_time[band] = -1

    def get_used_resources(self) -> Dict[BandType, Resource]:
        return self._band_used_resources

    def get_remaining_resources(self) -> Dict[BandType, Resource]:
        resources = {}
        for band, resource in self._band_total_resources.items():
            used_resource = self.get_used_resources()[band]
            resources[band] = resource - used_resource
        return resources

    async def get_idle_bands(self, idle_duration: int):
        """Return a band list which all bands has been idle for at least `idle_duration` seconds."""
        now = time.time()
        idle_bands = []
        for band in self._band_total_resources.keys():
            idle_start_time = self._band_idle_start_time.get(band)
            if idle_start_time is None:  # pragma: no cover
                # skip new requested band for this round scale in.
                self._band_idle_start_time[band] = now
            elif idle_start_time > 0 and now >= idle_start_time + idle_duration:
                idle_bands.append(band)
        return idle_bands

    async def wait_band_idle(self, band: BandType):
        if self._band_idle_start_time[band] <= 0:
            if band in self._band_idle_events:
                event = self._band_idle_events[band]
            else:
                event = asyncio.Event()
                self._band_idle_events[band] = event
            return event.wait()
