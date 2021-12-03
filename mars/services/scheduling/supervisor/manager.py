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
import functools
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union
from weakref import WeakValueDictionary

from .... import oscar as mo
from ....core.operand import Fetch
from ....lib.aio import alru_cache
from ....oscar.backends.message import ProfilingContext
from ....oscar.errors import MarsError
from ....typing import BandType
from ....utils import dataslots
from ...subtask import Subtask, SubtaskResult, SubtaskStatus
from ...task import TaskAPI
from ..core import SubtaskScheduleSummary
from ..utils import redirect_subtask_errors

logger = logging.getLogger(__name__)


# the default times to reschedule subtask.
DEFAULT_SUBTASK_MAX_RESCHEDULES = 0


@dataslots
@dataclass
class SubtaskScheduleInfo:
    __slots__ = ("__weakref__",)

    subtask: Subtask
    priority: Tuple
    max_reschedules: int = 0
    num_reschedules: int = 0
    cached_workers: Set[str] = field(default_factory=set)
    submitted_bands: Set[BandType] = field(default_factory=set)

    def to_summary(self, **kwargs) -> SubtaskScheduleSummary:
        return SubtaskScheduleSummary(
            task_id=self.subtask.task_id,
            subtask_id=self.subtask.subtask_id,
            bands=list(self.submitted_bands),
            num_reschedules=self.num_reschedules,
            **kwargs,
        )


class SubtaskManagerActor(mo.Actor):
    # subtask id -> schedule info
    _subtask_infos: Dict[str, SubtaskScheduleInfo]
    # subtask id -> summary
    _subtask_summaries: Dict[str, SubtaskScheduleSummary]
    # chunk key -> schedule info
    _chunk_key_to_subtask_info: Mapping[str, SubtaskScheduleInfo]

    @classmethod
    def gen_uid(cls, session_id: str):
        return f"{session_id}_subtask_manager"

    def __init__(
        self,
        session_id: str,
        subtask_max_reschedules: int = DEFAULT_SUBTASK_MAX_RESCHEDULES,
    ):
        self._session_id = session_id
        self._subtask_infos = dict()
        self._subtask_summaries = dict()
        self._chunk_key_to_subtask_info = WeakValueDictionary()
        self._subtask_max_reschedules = subtask_max_reschedules

        self._assigner_ref = None
        self._global_slot_ref = None

    async def __post_create__(self):
        from .assigner import AssignerActor

        self._assigner_ref = await mo.actor_ref(
            AssignerActor.gen_uid(self._session_id), address=self.address
        )

    @alru_cache
    async def _get_task_api(self):
        return await TaskAPI.create(self._session_id, self.address)

    def _put_subtask_with_priority(self, subtask: Subtask, priority: Tuple = None):
        # if already placed, just update priority and return
        if subtask.subtask_id in self._subtask_infos:
            self._subtask_infos[subtask.subtask_id].priority = priority
            return

        # the extra_config may be None. the extra config overwrites the default value.
        subtask_max_reschedules = (
            subtask.extra_config.get("subtask_max_reschedules")
            if subtask.extra_config
            else None
        )
        if subtask_max_reschedules is None:
            subtask_max_reschedules = self._subtask_max_reschedules
        subtask_info = self._subtask_infos[subtask.subtask_id] = SubtaskScheduleInfo(
            subtask, priority, max_reschedules=subtask_max_reschedules
        )

        if subtask.chunk_graph:
            for result_chunk in subtask.chunk_graph.results:
                self._chunk_key_to_subtask_info[result_chunk.key] = subtask_info

    async def _filter_virtual_subtasks(
        self, subtasks: List[Subtask], priorities: List[Tuple]
    ):
        # filter out virtual subtasks
        has_virtual = False
        task_api = None
        for subtask, priority in zip(subtasks, priorities):
            if subtask.virtual:
                if task_api is None:
                    task_api = await self._get_task_api()
                has_virtual = True
                await task_api.set_subtask_result(
                    SubtaskResult(
                        subtask_id=subtask.subtask_id,
                        session_id=subtask.session_id,
                        task_id=subtask.task_id,
                        progress=1.0,
                        status=SubtaskStatus.succeeded,
                    )
                )
        if not has_virtual:
            return subtasks, priorities
        else:
            subtasks_new, priorities_new = [], []
            for subtask, priority in zip(subtasks, priorities):
                if subtask.virtual:
                    continue
                subtasks_new.append(subtask)
                priorities_new.append(priority)
            return subtasks_new, priorities_new

    async def add_subtasks(self, subtasks: List[Subtask], priorities: List[Tuple]):
        # filter out virtual subtasks
        subtasks, priorities = await self._filter_virtual_subtasks(subtasks, priorities)
        async with redirect_subtask_errors(self, subtasks):
            for subtask, priority in zip(subtasks, priorities):
                self._put_subtask_with_priority(subtask, priority)

            band_list = await self._assigner_ref.assign_subtasks(subtasks)

            band_to_subtasks = defaultdict(list)
            band_to_priorities = defaultdict(list)

            for subtask, priority, band in zip(subtasks, priorities, band_list):
                info = self._subtask_infos[subtask.subtask_id]
                if band[0] not in info.cached_workers:
                    subtask_data = subtask
                else:
                    subtask_data = subtask.subtask_id
                band_to_subtasks[band].append(subtask_data)
                band_to_priorities[band].append(priority)

            coros = []
            for band in band_to_subtasks.keys():
                execution_ref = await self._get_execution_ref(band[0])
                coro = execution_ref.submit_subtasks(
                    band_to_subtasks[band],
                    band_to_priorities[band],
                    self.address,
                    band[1],
                )
                coros.append(coro)

            yield tuple(coros)

            for subtask, band in zip(subtasks, band_list):
                self._subtask_infos[subtask.subtask_id].submitted_bands.add(band)

    async def cache_subtasks(
        self, subtasks: List[Subtask], priorities: Optional[List[Tuple]] = None
    ):
        band_to_subtasks = defaultdict(list)
        band_to_priorities = defaultdict(list)
        subtask_id_to_workers = dict()
        for subtask, priority in zip(subtasks, priorities):
            if subtask.virtual:
                continue
            self._put_subtask_with_priority(subtask, priority)

            bands_list = []
            if subtask.chunk_graph:
                for chunk in subtask.chunk_graph:
                    if not isinstance(chunk.op, Fetch):
                        continue
                    try:
                        src_subtask_info = self._chunk_key_to_subtask_info[chunk.key]
                    except KeyError:
                        continue
                    bands_list.append(src_subtask_info.submitted_bands)

            if not bands_list:
                cache_bands = []
            else:
                cache_bands = list(functools.reduce(operator.and_, bands_list))

            for band in cache_bands:
                band_to_subtasks[band].append(subtask)
                band_to_priorities[band].append(priority)

            subtask_id_to_workers[subtask.subtask_id] = set(
                band[0] for band in cache_bands
            )

        coros = []
        for band in band_to_subtasks.keys():
            execution_ref = await self._get_execution_ref(band[0])
            coro = execution_ref.cache_subtasks(
                band_to_subtasks[band],
                band_to_priorities[band],
                self.address,
                band[1],
            )
            coros.append(coro)
        yield tuple(coros)

        for subtask, priority in zip(subtasks, priorities):
            if subtask.virtual:
                continue
            self._subtask_infos[
                subtask.subtask_id
            ].cached_workers = subtask_id_to_workers[subtask.subtask_id]

    async def update_subtask_priorities(
        self, subtask_ids: List[str], priorities: List[Tuple] = None
    ):
        worker_to_subtask_ids = defaultdict(list)
        worker_to_priorities = defaultdict(list)
        for subtask_id, priority in zip(subtask_ids, priorities):
            try:
                info = self._subtask_infos[subtask_id]
            except KeyError:
                continue
            info.priority = priority
            workers = set(info.cached_workers) | set(
                band[0] for band in info.submitted_bands
            )
            for worker in workers:
                worker_to_subtask_ids[worker].append(subtask_id)
                worker_to_priorities[worker].append(priority)
        coros = []
        for worker in worker_to_subtask_ids.keys():
            ref = await self._get_execution_ref(worker)
            coro = ref.update_subtask_priorities(
                worker_to_subtask_ids[worker],
                worker_to_priorities[worker],
            )
            coros.append(coro)
        yield tuple(coros)

    @alru_cache(maxsize=10000)
    async def _get_execution_ref(self, address: str):
        from ..worker.exec import SubtaskExecutionActor

        return await mo.actor_ref(SubtaskExecutionActor.default_uid(), address=address)

    async def finish_subtasks(self, subtask_ids: List[str], schedule_next: bool = True):
        band_tasks = defaultdict(lambda: 0)
        for subtask_id in subtask_ids:
            subtask_info = self._subtask_infos.pop(subtask_id, None)
            if subtask_info is not None:
                self._subtask_summaries[subtask_id] = subtask_info.to_summary(
                    is_finished=True
                )
                if schedule_next:
                    for band in subtask_info.submitted_bands:
                        band_tasks[band] += 1

    def _get_subtasks_by_ids(self, subtask_ids: List[str]) -> List[Optional[Subtask]]:
        subtasks = []
        for stid in subtask_ids:
            try:
                subtasks.append(self._subtask_infos[stid].subtask)
            except KeyError:
                subtasks.append(None)
        return subtasks

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Union[float, int] = 5
    ):
        task_api = await self._get_task_api()

        async def cancel_and_wait(worker: str, subtask_ids: List[str]):
            execution_ref = await self._get_execution_ref(worker)
            await execution_ref.cancel_subtasks(subtask_ids, kill_timeout=kill_timeout)
            await execution_ref.wait_subtasks(subtask_ids)

            result_batches = []
            for subtask_id in subtask_ids:
                subtask = self._subtask_infos[subtask_id].subtask
                result = SubtaskResult(
                    subtask_id=subtask.subtask_id,
                    session_id=subtask.session_id,
                    task_id=subtask.task_id,
                    status=SubtaskStatus.cancelled,
                )
                result_batches.append(task_api.set_subtask_result.delay(result))
            await task_api.set_subtask_result.batch(*result_batches)

        worker_to_subtask_ids = defaultdict(list)

        for subtask_id in subtask_ids:
            try:
                info = self._subtask_infos[subtask_id]
            except KeyError:
                continue
            workers = set(band[0] for band in info.submitted_bands) | set(
                info.cached_workers
            )
            for worker in workers:
                worker_to_subtask_ids[worker].append(subtask_id)

        cancel_coros = []
        for worker, subtask_ids in worker_to_subtask_ids.items():
            cancel_coros.append(cancel_and_wait(worker, subtask_ids))
        if cancel_coros:
            await asyncio.gather(*cancel_coros)

        for subtask_id in subtask_ids:
            subtask_info = self._subtask_infos.pop(subtask_id, None)
            if subtask_info is not None:
                self._subtask_summaries[subtask_id] = subtask_info.to_summary(
                    is_finished=True, is_cancelled=True
                )

    def get_schedule_summaries(self, task_id: Optional[str] = None):
        if task_id is not None:
            summaries = {
                subtask_id: summary
                for subtask_id, summary in self._subtask_summaries.items()
                if summary.task_id == task_id
            }
        else:
            summaries = dict(self._subtask_summaries)
        for info in self._subtask_infos.values():
            if task_id is None or info.subtask.task_id == task_id:
                summaries[info.subtask.subtask_id] = info.to_summary()
        return list(summaries.values())
