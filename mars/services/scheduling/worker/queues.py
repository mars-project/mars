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
from typing import Dict, NamedTuple, Tuple

from .... import oscar as mo
from ...cluster import ClusterAPI

logger = logging.getLogger(__name__)


class QueueItem(NamedTuple):
    subtask_id: str
    priority: tuple

    def __lt__(self, other: "QueueItem"):
        return self.priority > other.priority


class SlotsContainer(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._item_set = set(self)

    def append(self, item) -> None:
        super().append(item)
        self._item_set.add(item)

    def insert(self, index, item) -> None:
        super().insert(index, item)
        self._item_set.add(index)

    def pop(self, index=None):
        if index is not None:
            popped = super().pop(index)
        else:
            popped = super().pop()
        self._item_set.remove(popped)
        return popped

    def __contains__(self, item):
        return item in self._item_set


class SubtaskPriorityQueueActor(mo.StatelessActor):
    _subtask_id_to_item: Dict[str, QueueItem]
    _subtask_id_to_band_slot: Dict[str, Tuple[str, str]]

    _band_n_slots: Dict[str, int]
    _band_active_slots: Dict[str, SlotsContainer]
    _semaphores: Dict[str, asyncio.Semaphore]
    _queues: Dict[str, asyncio.PriorityQueue]

    def __init__(self):
        self._subtask_id_to_item = dict()
        self._subtask_id_to_band_slot = dict()

        self._semaphores = dict()
        self._queues = dict()
        self._band_n_slots = dict()
        self._band_active_slots = dict()

    async def __post_create__(self):
        cluster_api = await ClusterAPI.create(self.address)
        bands = await cluster_api.get_bands()

        for (_a, band_name), slot_num in bands.items():
            self._init_band(band_name, slot_num)

    def _init_band(self, band_name: str, slot_num: int):
        self._band_n_slots[band_name] = slot_num
        self._band_active_slots[band_name] = SlotsContainer(range(slot_num))
        self._semaphores[band_name] = asyncio.Semaphore(slot_num)
        self._queues[band_name] = asyncio.PriorityQueue()

    @mo.extensible
    def put(self, subtask_id: str, band_name: str, priority: Tuple):
        logger.debug(f"Subtask {subtask_id} enqueued in band {band_name}")
        item = QueueItem(subtask_id, priority)
        self._subtask_id_to_item[subtask_id] = item
        self._queues[band_name].put_nowait(item)

    @mo.extensible
    def update_priority(self, subtask_id: str, band_name: str, priority: Tuple):
        try:
            old_item = self._subtask_id_to_item[subtask_id]
        except KeyError:
            return

        if old_item.priority >= priority:
            return
        new_item = QueueItem(old_item.subtask_id, priority)
        self._subtask_id_to_item[subtask_id] = new_item
        self._queues[band_name].put_nowait(new_item)

    async def get(self, band_name: str):
        item = None
        while True:
            try:
                item = await self._queues[band_name].get()

                subtask_id = item.subtask_id
                if item.subtask_id not in self._subtask_id_to_item:
                    continue

                await self._semaphores[band_name].acquire()
                if item.subtask_id not in self._subtask_id_to_item:
                    self._semaphores[band_name].release()
                    continue
                slot = self._band_active_slots[band_name].pop()
                break
            except asyncio.CancelledError:
                if item is not None:
                    # put back enqueued item
                    self._queues[band_name].put_nowait(item)
                raise

        self._subtask_id_to_item.pop(subtask_id)
        self._subtask_id_to_band_slot[subtask_id] = (band_name, slot)
        return item.subtask_id, slot

    def release_slot(self, subtask_id: str, errors: str = "raise"):
        try:
            band_name, slot_id = self._subtask_id_to_band_slot.pop(subtask_id)
        except KeyError:
            if errors == "raise":
                raise
            else:
                return
        self._band_active_slots[band_name].append(slot_id)
        self._semaphores[band_name].release()

    @mo.extensible
    def remove(self, subtask_id: str):
        removed = self._subtask_id_to_item.pop(subtask_id, None)
        self.release_slot(subtask_id, errors="ignore")
        return None if removed is None else subtask_id


class SubtaskPrepareQueueActor(SubtaskPriorityQueueActor):
    pass


class SubtaskExecutionQueueActor(SubtaskPriorityQueueActor):
    async def __post_create__(self):
        await super().__post_create__()

        from .slotmanager import SlotManagerActor

        self._slot_manager = await mo.actor_ref(
            SlotManagerActor.default_uid(), address=self.address
        )

    async def restart_free_slots(self, band_name: str):
        slots = []
        sem = self._semaphores[band_name]

        # occupy all free slots
        while not sem.locked():
            try:
                await asyncio.wait_for(sem.acquire(), timeout=0.1)
                slots.append(self._band_active_slots[band_name].pop())
            except asyncio.TimeoutError:
                break

        for slot in slots:
            await self._slot_manager.kill_slot(band_name, int(slot))

        for slot in slots:
            self._band_active_slots[band_name].append(slot)
            sem.release()
