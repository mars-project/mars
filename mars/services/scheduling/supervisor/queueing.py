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
import copy
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple, Union, Set

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....utils import dataslots, create_task_with_error_log
from ...subtask import Subtask
from ...task import TaskAPI
from ..utils import redirect_subtask_errors

logger = logging.getLogger(__name__)

_DEFAULT_SUBMIT_PERIOD = 0


@dataslots
@dataclass
class HeapItem:
    subtask: Subtask
    priority: Tuple

    def __lt__(self, other: "HeapItem"):
        return self.priority > other.priority


class SubtaskQueueingActor(mo.Actor):
    _stid_to_bands: DefaultDict[str, List[Tuple]]
    _stid_to_items: Dict[str, HeapItem]
    _band_queues: DefaultDict[Tuple, List[HeapItem]]

    @classmethod
    def gen_uid(cls, session_id: str):
        return f"{session_id}_subtask_queueing"

    def __init__(self, session_id: str, submit_period: Union[float, int] = None):
        self._session_id = session_id
        self._stid_to_bands = defaultdict(list)
        self._stid_to_items = dict()
        # Note that we need to ensure top item in every band heap queue is valid,
        # so that we can ensure band queue is busy if the band queue is not empty.
        self._band_queues = defaultdict(list)

        self._cluster_api = None
        self._slots_ref = None
        self._assigner_ref = None

        self._band_slot_nums = dict()
        self._band_watch_task = None
        self._max_enqueue_id = 0

        self._periodical_submit_task = None
        self._submit_period = submit_period or _DEFAULT_SUBMIT_PERIOD

    async def __post_create__(self):
        from ...cluster import ClusterAPI

        self._cluster_api = await ClusterAPI.create(self.address)
        self._band_slot_nums = {}
        from .globalslot import GlobalSlotManagerActor

        [self._slots_ref] = await self._cluster_api.get_supervisor_refs(
            [GlobalSlotManagerActor.default_uid()]
        )
        from .assigner import AssignerActor

        self._assigner_ref = await mo.actor_ref(
            AssignerActor.gen_uid(self._session_id), address=self.address
        )

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands():
                # confirm ready bands indeed changed
                if bands != self._band_slot_nums:
                    old_band_slot_nums = self._band_slot_nums
                    self._band_slot_nums = copy.deepcopy(bands)
                    await self._slots_ref.refresh_bands()
                    all_bands = {*bands.keys(), *old_band_slot_nums.keys()}
                    bands_delta = {}
                    for b in all_bands:
                        delta = bands.get(b, 0) - old_band_slot_nums.get(b, 0)
                        if delta != 0:
                            bands_delta[b] = delta
                    # Submit tasks on new bands manually, otherwise some subtasks
                    # will never got submitted. Note that we must ensure every new
                    # band will get at least one subtask submitted successfully.
                    # Later subtasks submit on the band will be triggered by the
                    # success of previous subtasks on the same band.
                    logger.info(
                        "Bands changed with delta %s, submit all bands.",
                        bands_delta,
                    )
                    # submit_subtasks may empty _band_queues, so use `ref()` to wait previous `submit_subtasks` call
                    # finish.
                    await self.ref().balance_queued_subtasks()
                    # Refresh global slot manager to get latest bands,
                    # so that subtasks reassigned to the new bands can be
                    # ensured to get submitted as least one subtask every band
                    # successfully.
                    await self.ref().submit_subtasks()

        self._band_watch_task = create_task_with_error_log(watch_bands())

        if self._submit_period > 0:
            self._periodical_submit_task = self.ref().periodical_submit.tell_delay(
                delay=self._submit_period
            )

    async def __pre_destroy__(self):
        self._band_watch_task.cancel()
        if self._periodical_submit_task is not None:  # pragma: no branch
            self._periodical_submit_task.cancel()

    async def periodical_submit(self):
        await self.ref().submit_subtasks.tell()
        self._periodical_submit_task = self.ref().periodical_submit.tell_delay(
            delay=self._submit_period
        )

    @alru_cache
    async def _get_task_api(self):
        return await TaskAPI.create(self._session_id, self.address)

    @alru_cache(cache_exceptions=False)
    async def _get_manager_ref(self):
        from .manager import SubtaskManagerActor

        return await mo.actor_ref(
            SubtaskManagerActor.gen_uid(self._session_id), address=self.address
        )

    async def add_subtasks(
        self,
        subtasks: List[Subtask],
        priorities: List[Tuple],
        exclude_bands: Set[Tuple] = None,
        random_when_unavailable: bool = True,
    ):
        bands = await self._assigner_ref.assign_subtasks(
            subtasks, exclude_bands, random_when_unavailable
        )
        for subtask, band, priority in zip(subtasks, bands, priorities):
            assert band is not None
            self._stid_to_bands[subtask.subtask_id].append(band)
            heap_item = self._stid_to_items[subtask.subtask_id] = HeapItem(
                subtask, priority + (self._max_enqueue_id,)
            )
            self._max_enqueue_id += 1
            heapq.heappush(self._band_queues[band], heap_item)
            logger.debug(
                "Subtask %s enqueued to band %s excluded from %s.",
                subtask.subtask_id,
                band,
                exclude_bands,
            )
        logger.debug("%d subtasks enqueued", len(subtasks))

    async def submit_subtasks(self, band: Tuple = None, limit: Optional[int] = None):
        logger.debug("Submitting subtasks with limit %s to band %s", limit, band)

        if not limit and band not in self._band_slot_nums:
            self._band_slot_nums = await self._cluster_api.get_all_bands()

        bands = [band] if band is not None else list(self._band_slot_nums.keys())
        submit_aio_tasks = []
        manager_ref = await self._get_manager_ref()
        apply_delays = []
        submit_items_list = []
        submitted_bands = []

        for band in bands:
            band_limit = limit or self._band_slot_nums[band]
            task_queue = self._band_queues[band]
            submit_items = dict()
            while (
                self._ensure_top_item_valid(task_queue)
                and len(submit_items) < band_limit
            ):
                item = heapq.heappop(task_queue)
                submit_items[item.subtask.subtask_id] = item

            subtask_ids = list(submit_items)
            if not subtask_ids:
                continue
            submitted_bands.append(band)
            submit_items_list.append(submit_items)
            subtask_resources = [
                Resource(
                    num_cpus=item.subtask.num_cpus,
                    num_mem_bytes=item.subtask.num_mem_bytes if self._hbo_enabled else 0,
                )
                for item in submit_items.values()
            ]
            apply_delays.append(
                self._slots_ref.apply_subtask_resources.delay(
                    band, self._session_id, subtask_ids, subtask_resources
                )
            )

        async with redirect_subtask_errors(
                self,
                [
                    item.subtask
                    for submit_items in submit_items_list
                    for item in submit_items.values()
                ],
        ):
            submitted_ids_list = await self._slots_ref.apply_subtask_resources.batch(
                *apply_delays
            )
        for band, submit_items, submitted_ids in zip(
                submitted_bands, submit_items_list, submitted_ids_list
        ):
            subtask_ids = list(submit_items)
            task_queue = self._band_queues[band]
            async with redirect_subtask_errors(
                self, [item.subtask for item in submit_items.values()]
            ):
                non_submitted_ids = [k for k in submit_items if k not in submitted_ids]
                metrics_options = {
                    "session_id": self._session_id,
                    "band": band[0] if band else "",
                }
                self._submitted_subtask_count.record(
                    len(submitted_ids), metrics_options
                )
                self._unsubmitted_subtask_count.record(
                    len(non_submitted_ids), metrics_options
                )
                if submitted_ids:
                    for stid in subtask_ids:
                        if stid not in submitted_ids:
                            continue
                        item = submit_items[stid]
                        item.subtask.submitted = True
                        logger.debug(
                            "Submit subtask %s to band %r",
                            item.subtask.subtask_id,
                            band,
                        )
                        submit_aio_tasks.append(
                            asyncio.create_task(
                                manager_ref.submit_subtask_to_band.tell(
                                    item.subtask.subtask_id, band
                                )
                            )
                        )
                        await asyncio.sleep(0)
                        self.remove_queued_subtasks([item.subtask.subtask_id])
                else:
                    logger.debug("No slots available")

            for stid in non_submitted_ids:
                item = submit_items[stid]
                self._max_enqueue_id += 1
                # lower priority to ensure other subtasks can be scheduled.
                item.priority = item.priority[:-1] + (self._max_enqueue_id,)
                heapq.heappush(task_queue, item)
            if non_submitted_ids:
                log_func = logger.info if self._periodical_submit_task is None else logger.debug
                log_func("No slots available, band queues status: %s", self.band_queue_subtask_nums())

        if submit_aio_tasks:
            yield asyncio.gather(*submit_aio_tasks)
        else:
            logger.debug("No subtasks to submit, perhaps because of the lack of resources.")

    def _ensure_top_item_valid(self, task_queue):
        """Clean invalid subtask item from the queue to ensure that when the queue is not empty,
        there is always some subtasks waiting being scheduled."""
        while (
            task_queue and task_queue[0].subtask.subtask_id not in self._stid_to_items
        ):
            #  skip removed items (as they may be re-pushed into the queue)
            heapq.heappop(task_queue)
        return bool(task_queue)

    @mo.extensible
    def update_subtask_priority(self, subtask_id: str, priority: Tuple):
        if subtask_id not in self._stid_to_bands:
            return
        for band in self._stid_to_bands[subtask_id]:
            new_item = HeapItem(self._stid_to_items[subtask_id].subtask, priority)
            self._stid_to_items[subtask_id] = new_item
            heapq.heappush(self._band_queues[band], new_item)

    def remove_queued_subtasks(self, subtask_ids: List[str]):
        for stid in subtask_ids:
            bands = self._stid_to_bands.pop(stid, [])
            self._stid_to_items.pop(stid, None)
            for band in bands:
                band_queue = self._band_queues.get(band)
                self._ensure_top_item_valid(band_queue)

    async def all_bands_busy(self) -> bool:
        """Return True if all bands queue has tasks waiting to be submitted."""
        bands = set(self._band_slot_nums.keys())
        if set(self._band_queues.keys()).issuperset(bands):
            return all(len(self._band_queues[band]) > 0 for band in bands)
        return False

    def band_queue_subtask_nums(self):
        return {q: len(subtasks) for q, subtasks in self._band_queues.items()}

    async def balance_queued_subtasks(self):
        used_slots = {band: slots for band, slots in (await self._slots_ref.get_used_slots()).items() if slots > 0}
        remaining_slots = await self._slots_ref.get_remaining_slots()
        # record length of band queues
        band_num_queued_subtasks = {}
        for band, queue in self._band_queues.items():
            queue_size = len([item for item in queue if item.subtask.subtask_id in self._stid_to_items])
            if queue_size > 0:
                band_num_queued_subtasks[band] = queue_size
        logger.info("Start to balance subtasks:\n"
                    "used_slots %s\n remaining_slots %s\n queue size %s\n band_num_queued_subtasks %s",
                    used_slots, remaining_slots,
                    {band: len(queue) for band, queue in self._band_queues.items() if len(queue) > 0},
                    band_num_queued_subtasks)
        if sum(band_num_queued_subtasks.values()) == 0:
            logger.info("No subtasks in queue, skip balance.")
            return
        move_queued_subtasks = await self._assigner_ref.reassign_subtasks(
            band_num_queued_subtasks, used_slots=used_slots
        )
        items = []
        # rewrite band queues according to feedbacks from assigner
        for band, move in move_queued_subtasks.items():
            task_queue = self._band_queues[band]
            queue_size = len([item for item in task_queue if item.subtask.subtask_id in self._stid_to_items])
            assert queue_size + move >= 0, f"move {move} from {band, task_queue} " \
                                           f"move_queued_subtasks {move_queued_subtasks} " \
                                           f"band_num_queued_subtasks {band_num_queued_subtasks}"
            for _ in range(abs(move)):
                if move < 0:
                    # TODO: pop item of low priority
                    self._ensure_top_item_valid(task_queue)
                    item = heapq.heappop(task_queue)
                    self._stid_to_bands[item.subtask.subtask_id].remove(band)
                    items.append(item)
                elif move > 0:
                    item = items.pop()
                    subtask = item.subtask
                    if subtask.bands_specified and band not in subtask.expect_bands:
                        logger.warning("Skip reschedule subtask %s to band %s because it's band is specified to %s.",
                                       subtask.subtask_id, band, subtask.expect_bands)
                        specified_band = subtask.expect_band
                        specified_band_queue = self._band_queues[specified_band]
                        heapq.heappush(specified_band_queue, item)
                        self._stid_to_bands[item.subtask.subtask_id].append(specified_band)
                    else:
                        subtask.expect_bands = [band]
                        self._stid_to_bands[item.subtask.subtask_id].append(band)
                        heapq.heappush(task_queue, item)
            if len(task_queue) == 0:
                self._band_queues.pop(band)
        balanced_num_queued_subtasks = {}
        for band, queue in self._band_queues.items():
            band_length = len([item for item in queue if item.subtask.subtask_id in self._stid_to_items])
            if band_length > 0:
                balanced_num_queued_subtasks[band] = band_length
        logger.info("Balance subtasks succeed:\n move_queued_subtasks %s\n "
                    "balanced_num_queued_subtasks %s", move_queued_subtasks, balanced_num_queued_subtasks)
