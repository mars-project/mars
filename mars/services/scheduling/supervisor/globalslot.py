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
from collections import defaultdict
from typing import List, DefaultDict, Dict, Tuple

from .... import oscar as mo
from ....typing import BandType

logger = logging.getLogger(__name__)


class GlobalSlotManagerActor(mo.Actor):
    _band_stid_slots: DefaultDict[BandType, Dict[Tuple[str, str], int]]
    _band_used_slots: DefaultDict[BandType, int]
    _band_total_slots: Dict[BandType, int]

    def __init__(self):
        self._band_stid_slots = defaultdict(dict)
        self._band_used_slots = defaultdict(lambda: 0)
        self._band_total_slots = dict()

        self._cluster_api = None

        self._band_watch_task = None

    async def __post_create__(self):
        from ...cluster.api import ClusterAPI
        self._cluster_api = await ClusterAPI.create(self.address)

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands():
                self._band_total_slots = bands

        self._band_watch_task = asyncio.create_task(watch_bands())

    async def __pre_destroy__(self):
        self._band_watch_task.cancel()

    async def apply_subtask_slots(self, band: Tuple, session_id: str,
                                  subtask_ids: List[str], subtask_slots: List[int]) -> List[str]:
        if not self._band_total_slots or band not in self._band_total_slots:
            self._band_total_slots = await self._cluster_api.get_all_bands()

        idx = 0
        # only ready bands will pass
        if band in self._band_total_slots:
            total_slots = self._band_total_slots[band]
            for stid, slots in zip(subtask_ids, subtask_slots):
                if self._band_used_slots[band] + slots > total_slots:
                    break
                self._band_stid_slots[band][(session_id, stid)] = slots
                self._band_used_slots[band] += slots
                idx += 1
        if idx == 0:
            logger.debug('No slots available, status: %r, request: %r',
                         self._band_used_slots, subtask_slots)
        return subtask_ids[:idx]

    @mo.extensible
    def update_subtask_slots(self, band: Tuple, session_id: str, subtask_id: str, slots: int):
        session_subtask_id = (session_id, subtask_id)
        subtask_slots = self._band_stid_slots[band]

        if session_subtask_id not in subtask_slots:
            return

        slots_delta = slots - subtask_slots[session_subtask_id]
        subtask_slots[session_subtask_id] = slots
        self._band_used_slots[band] += slots_delta

    @mo.extensible
    def release_subtask_slots(self, band: Tuple, session_id: str, subtask_id: str):
        # todo ensure slots released when subtasks ends in all means
        slots_delta = self._band_stid_slots[band].pop((session_id, subtask_id), 0)
        self._band_used_slots[band] -= slots_delta

    def get_used_slots(self):
        return self._band_used_slots
