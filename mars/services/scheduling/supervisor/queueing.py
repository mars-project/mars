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
from ....metrics import Metrics
from ....resource import ZeroResource
from ....typing import BandType
from ....utils import dataslots
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
    _submit_requests: List[Optional[Dict[BandType, int]]]

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

        self._submit_requests = []
        self._submit_request_event = asyncio.Event()
        self._submit_request_task = None

        self._cluster_api = None
        self._slots_ref = None
        self._assigner_ref = None

        self._band_to_resource = dict()
        self._band_watch_task = None
        self._max_enqueue_id = 0

        self._submit_period = submit_period or _DEFAULT_SUBMIT_PERIOD
        self._submitted_subtask_number = Metrics.gauge(
            "mars.band.submitted_subtask_number",
            "The number of submitted subtask to a band.",
            ("session_id", "band"),
        )
        self._unsubmitted_subtask_number = Metrics.gauge(
            "mars.band.unsubmitted_subtask_number",
            "The number of unsubmitted subtask to a band.",
            ("session_id", "band"),
        )

    async def __post_create__(self):
        from ...cluster import ClusterAPI

        self._cluster_api = await ClusterAPI.create(self.address)
        self._band_to_resource = {}

        async def watch_bands():
            async for bands in self._cluster_api.watch_all_bands():
                # confirm ready bands indeed changed
                if bands != self._band_to_resource:
                    old_band_resource = self._band_to_resource
                    self._band_to_resource = copy.deepcopy(bands)
                    if self._band_queues:
                        await self.balance_queued_subtasks()
                        # Refresh global slot manager to get latest bands,
                        # so that subtasks reassigned to the new bands can be
                        # ensured to get submitted as least one subtask every band
                        # successfully.
                        await self._slots_ref.refresh_bands()
                        all_bands = {*bands.keys(), *old_band_resource.keys()}
                        bands_delta = {}
                        for b in all_bands:
                            new_resource = bands.get(b, ZeroResource)
                            old_resource = old_band_resource.get(b, ZeroResource)
                            delta = new_resource - old_resource
                            if delta != ZeroResource:
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
                        await self.ref().submit_subtasks()

        self._band_watch_task = asyncio.create_task(watch_bands())

        from .globalresource import GlobalResourceManagerActor

        [self._slots_ref] = await self._cluster_api.get_supervisor_refs(
            [GlobalResourceManagerActor.default_uid()]
        )
        from .assigner import AssignerActor

        self._assigner_ref = await mo.actor_ref(
            AssignerActor.gen_uid(self._session_id), address=self.address
        )

        self._submit_request_task = asyncio.create_task(self._submission_task_func())

    async def __pre_destroy__(self):
        self._band_watch_task.cancel()
        if self._submit_request_task is not None:  # pragma: no branch
            self._submit_request_task.cancel()

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
                "Subtask %s enqueued to band %s. exclude_bands=%s.",
                subtask.subtask_id,
                band,
                exclude_bands,
            )
        logger.debug("%d subtasks enqueued", len(subtasks))

    def submit_subtasks(self, band_to_limit: Dict[BandType, int] = None):
        self._submit_requests.append(band_to_limit)
        self._submit_request_event.set()

    async def _submission_task_func(self):
        while True:
            try:
                periodical_triggered = False
                if not self._submit_requests:  # pragma: no branch
                    try:
                        if self._submit_period:
                            await asyncio.wait_for(
                                self._submit_request_event.wait(), self._submit_period
                            )
                        else:
                            await self._submit_request_event.wait()

                        self._submit_request_event.clear()
                    except asyncio.TimeoutError:
                        periodical_triggered = True

                requests = self._submit_requests
                self._submit_requests = []
                if not periodical_triggered and not requests:  # pragma: no cover
                    continue

                merged_band_to_limit = dict()
                for req in requests:
                    if req is None:
                        merged_band_to_limit = None
                        break
                    merged_band_to_limit.update(req)
                await self._submit_subtask_request(merged_band_to_limit)
            except asyncio.CancelledError:
                break

    async def _submit_subtask_request(self, band_to_limit: Dict[BandType, int] = None):
        if band_to_limit:
            logger.debug(
                "TMP_QUEUE_PROBE: Submitting subtasks with limits: %r", band_to_limit
            )

        if not self._band_to_resource or any(
            not limit and band not in self._band_to_resource
            for band, limit in band_to_limit or ()
        ):
            self._band_to_resource = await self._cluster_api.get_all_bands()

        if not band_to_limit:
            band_to_limit = {band: None for band in self._band_to_resource.keys()}

        apply_delays = []
        submit_items_list = []
        submitted_bands = []

        def _load_items_to_submit():
            for band, limit in band_to_limit.items():
                band_limit = limit or (
                    self._band_to_resource[band].num_cpus
                    or self._band_to_resource[band].num_gpus
                )
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

                # Before hbo, when a manager finish a subtask, it will schedule one subtask successfully because
                # there is a slot idle. But now we have memory requirements, so the subtask may apply resource
                # from supervisor failed. In such cases, those subtasks will never got scheduled.
                # TODO We can use `_periodical_submit_task` to submit those subtasks.
                subtask_resources = [
                    item.subtask.required_resource for item in submit_items.values()
                ]
                apply_delays.append(
                    self._slots_ref.apply_subtask_resources.delay(
                        band, self._session_id, subtask_ids, subtask_resources
                    )
                )

        await asyncio.to_thread(_load_items_to_submit)

        logger.debug("TMP_QUEUE_PROBE: Finished picking top subtasks")

        async with redirect_subtask_errors(
            self,
            (
                item.subtask
                for submit_items in submit_items_list
                for item in submit_items.values()
            ),
        ):
            submitted_ids_list = await self._slots_ref.apply_subtask_resources.batch(
                *apply_delays
            )

        logger.debug(
            "TMP_QUEUE_PROBE: Finished band resource allocation, %d subtasks submitted",
            sum(len(ids) for ids in submitted_ids_list),
        )

        manager_ref = await self._get_manager_ref()
        submit_delays = []

        def _gather_submissions():
            for band, submit_items, submitted_ids in zip(
                submitted_bands, submit_items_list, submitted_ids_list
            ):
                subtask_ids = list(submit_items)
                task_queue = self._band_queues[band]
                submitted_id_set = set(submitted_ids)

                non_submitted_ids = [
                    k for k in submit_items if k not in submitted_id_set
                ]
                tags = {
                    "session_id": self._session_id,
                    "band": band[0] if band else "",
                }
                self._submitted_subtask_number.record(len(submitted_ids), tags)
                self._unsubmitted_subtask_number.record(len(non_submitted_ids), tags)

                if not submitted_ids:
                    if non_submitted_ids:
                        logger.debug("No slots available on band %s", band)
                else:
                    for stid in subtask_ids:
                        if stid not in submitted_id_set:
                            continue
                        item = submit_items[stid]
                        logger.debug("Submit subtask %r to band %r", item.subtask, band)
                        submit_delays.append(
                            manager_ref.submit_subtask_to_band.delay(
                                item.subtask.subtask_id, band
                            )
                        )
                        self.remove_queued_subtasks([item.subtask.subtask_id])

                for stid in non_submitted_ids:
                    # TODO if subtasks submit failed due to lacking memory/cpu/gpu resources, lower the priority so that
                    # other subtasks can be submitted.
                    heapq.heappush(task_queue, submit_items[stid])

        await asyncio.to_thread(_gather_submissions)

        logger.debug("TMP_QUEUE_PROBE: Start subtask submission in batch")
        await manager_ref.submit_subtask_to_band.batch(*submit_delays)
        logger.debug("TMP_QUEUE_PROBE: Finished subtask submission")

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
        bands = set(self._band_to_resource.keys())
        if set(self._band_queues.keys()).issuperset(bands):
            return all(len(self._band_queues[band]) > 0 for band in bands)
        return False

    async def balance_queued_subtasks(self):
        # record length of band queues
        band_num_queued_subtasks = {
            band: len(queue) for band, queue in self._band_queues.items()
        }
        move_queued_subtasks = await self._assigner_ref.reassign_subtasks(
            band_num_queued_subtasks
        )
        items = []
        # rewrite band queues according to feedbacks from assigner
        for band, move in move_queued_subtasks.items():
            task_queue = self._band_queues[band]
            assert move + len(task_queue) >= 0
            for _ in range(abs(move)):
                if move < 0:
                    # TODO: pop item of low priority
                    item = heapq.heappop(task_queue)
                    self._stid_to_bands[item.subtask.subtask_id].remove(band)
                    items.append(item)
                elif move > 0:
                    item = items.pop()
                    self._stid_to_bands[item.subtask.subtask_id].append(band)
                    heapq.heappush(task_queue, item)
            if len(task_queue) == 0:
                self._band_queues.pop(band)
