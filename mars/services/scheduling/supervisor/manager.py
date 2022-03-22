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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....oscar.backends.message import ProfilingContext
from ....oscar.errors import MarsError
from ....oscar.profiling import ProfilingData, MARS_ENABLE_PROFILING
from ....typing import BandType
from ....utils import dataslots, Timer
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
    subtask: Subtask
    band_futures: Dict[BandType, asyncio.Future] = field(default_factory=dict)
    start_time: int = -1
    end_time: int = -1
    max_reschedules: int = 0
    num_reschedules: int = 0
    num_speculative_concurrent_run: int = 0

    def to_summary(self, **kwargs) -> SubtaskScheduleSummary:
        return SubtaskScheduleSummary(
            task_id=self.subtask.task_id,
            subtask_id=self.subtask.subtask_id,
            bands=list(self.band_futures.keys()),
            num_reschedules=self.num_reschedules,
            **kwargs,
        )


class SubtaskManagerActor(mo.Actor):
    _subtask_infos: Dict[str, SubtaskScheduleInfo]  # subtask id -> schedule info
    _subtask_summaries: Dict[str, SubtaskScheduleSummary]  # subtask id -> summary

    @classmethod
    def gen_uid(cls, session_id: str):
        return f"{session_id}_subtask_manager"

    def __init__(
        self,
        session_id: str,
        subtask_max_reschedules: int = DEFAULT_SUBTASK_MAX_RESCHEDULES,
        subtask_cancel_timeout: int = 5,
        speculation_config: Dict[str, object] = None,
    ):
        self._session_id = session_id
        self._subtask_infos = dict()
        self._subtask_summaries = dict()
        self._subtask_max_reschedules = subtask_max_reschedules
        self._subtask_cancel_timeout = subtask_cancel_timeout
        self._speculation_config = speculation_config or {}
        self._queueing_ref = None
        self._global_resource_ref = None
        logger.info(
            "Created SubtaskManager with subtask_max_reschedules %s, "
            "speculation_config %s",
            self._subtask_max_reschedules,
            speculation_config,
        )

    async def __post_create__(self):
        from .queueing import SubtaskQueueingActor

        self._queueing_ref = await mo.actor_ref(
            SubtaskQueueingActor.gen_uid(self._session_id), address=self.address
        )
        from ..supervisor import GlobalResourceManagerActor

        self._global_resource_ref = await mo.actor_ref(
            GlobalResourceManagerActor.default_uid(), address=self.address
        )
        from .speculation import SpeculativeScheduler

        self._speculation_execution_scheduler = SpeculativeScheduler(
            self._queueing_ref, self._global_resource_ref, self._speculation_config
        )
        await self._speculation_execution_scheduler.start()

    async def __pre_destroy__(self):
        await self._speculation_execution_scheduler.stop()

    @alru_cache
    async def _get_task_api(self):
        return await TaskAPI.create(self._session_id, self.address)

    async def add_subtasks(self, subtasks: List[Subtask], priorities: List[Tuple]):
        async with redirect_subtask_errors(self, subtasks):
            for subtask in subtasks:
                # the extra_config may be None. the extra config overwrites the default value.
                subtask_max_reschedules = (
                    subtask.extra_config.get("subtask_max_reschedules")
                    if subtask.extra_config
                    else None
                )
                if subtask_max_reschedules is None:
                    subtask_max_reschedules = self._subtask_max_reschedules
                if subtask.subtask_id in self._subtask_infos:  # pragma: no cover
                    raise KeyError(f"Subtask {subtask.subtask_id} already added.")
                self._subtask_infos[subtask.subtask_id] = SubtaskScheduleInfo(
                    subtask, max_reschedules=subtask_max_reschedules
                )

            virtual_subtasks = [subtask for subtask in subtasks if subtask.virtual]
            for subtask in virtual_subtasks:
                task_api = await self._get_task_api()
                await task_api.set_subtask_result(
                    SubtaskResult(
                        subtask_id=subtask.subtask_id,
                        session_id=subtask.session_id,
                        task_id=subtask.task_id,
                        stage_id=subtask.stage_id,
                        progress=1.0,
                        status=SubtaskStatus.succeeded,
                    )
                )
            await self._queueing_ref.add_subtasks(
                [subtask for subtask in subtasks if not subtask.virtual], priorities
            )
            await self._queueing_ref.submit_subtasks.tell()

    @alru_cache(maxsize=10000)
    async def _get_execution_ref(self, band: BandType):
        from ..worker.execution import SubtaskExecutionActor

        return await mo.actor_ref(SubtaskExecutionActor.default_uid(), address=band[0])

    async def finish_subtasks(
        self,
        subtask_ids: List[str],
        bands: List[BandType] = None,
        schedule_next: bool = True,
    ):
        logger.debug("Finished subtasks %s.", subtask_ids)
        band_tasks = defaultdict(lambda: 0)
        bands = bands or [None] * len(subtask_ids)
        for subtask_id, subtask_band in zip(subtask_ids, bands):
            subtask_info = self._subtask_infos.get(subtask_id, None)
            if subtask_info is not None:
                self._subtask_summaries[subtask_id] = subtask_info.to_summary(
                    is_finished=True
                )
                subtask_info.end_time = time.time()
                self._speculation_execution_scheduler.finish_subtask(subtask_info)
                #  Cancel subtask on other bands.
                aio_task = subtask_info.band_futures.pop(subtask_band, None)
                if aio_task:
                    await aio_task
                    if schedule_next:
                        band_tasks[subtask_band] += 1
                if subtask_info.band_futures:
                    # Cancel subtask here won't change subtask status.
                    # See more in `TaskProcessorActor.set_subtask_result`
                    logger.info(
                        "Try to cancel subtask %s on bands %s.",
                        subtask_id,
                        set(subtask_info.band_futures.keys()),
                    )
                    # Cancel subtask can be async and may need to kill slot which need more time.
                    # Can't use `tell` here because next line remove subtask info which is needed by
                    # `cancel_subtasks`.
                    yield self.ref().cancel_subtasks([subtask_id])
                # cancel subtask first then pop subtask info.
                self._subtask_infos.pop(subtask_id, None)
                if schedule_next:
                    for band in subtask_info.band_futures.keys():
                        band_tasks[band] += 1
        await self._queueing_ref.remove_queued_subtasks(subtask_ids)
        if band_tasks:
            coros = []
            for band, subtask_count in band_tasks.items():
                coros.append(
                    self._queueing_ref.submit_subtasks.tell(band, subtask_count)
                )
            await asyncio.wait(coros)

    def _get_subtasks_by_ids(self, subtask_ids: List[str]) -> List[Optional[Subtask]]:
        subtasks = []
        for stid in subtask_ids:
            try:
                subtasks.append(self._subtask_infos[stid].subtask)
            except KeyError:
                subtasks.append(None)
        return subtasks

    async def submit_subtask_to_band(self, subtask_id: str, band: BandType):
        if subtask_id not in self._subtask_infos:  # pragma: no cover
            logger.info(
                "Subtask %s is not in added subtasks set, it may be finished or canceled, skip it.",
                subtask_id,
            )
            return
        async with redirect_subtask_errors(
            self, self._get_subtasks_by_ids([subtask_id])
        ):
            try:
                subtask_info = self._subtask_infos[subtask_id]
                execution_ref = await self._get_execution_ref(band)
                extra_config = subtask_info.subtask.extra_config
                enable_profiling = MARS_ENABLE_PROFILING or (
                    extra_config and extra_config.get("enable_profiling")
                )
                profiling_context = (
                    ProfilingContext(subtask_info.subtask.task_id)
                    if enable_profiling
                    else None
                )
                logger.debug("Start run subtask %s in band %s.", subtask_id, band)
                with Timer() as timer:
                    task = asyncio.create_task(
                        execution_ref.run_subtask.options(
                            profiling_context=profiling_context
                        ).send(subtask_info.subtask, band[1], self.address)
                    )
                    subtask_info.band_futures[band] = task
                    subtask_info.start_time = time.time()
                    self._speculation_execution_scheduler.add_subtask(subtask_info)
                    result = yield task
                ProfilingData.collect_subtask(
                    subtask_info.subtask, band, timer.duration
                )
                task_api = await self._get_task_api()
                logger.debug("Finished subtask %s with result %s.", subtask_id, result)
                await task_api.set_subtask_result(result)
            except (OSError, MarsError) as ex:
                # TODO: We should handle ServerClosed Error.
                if (
                    subtask_info.subtask.retryable
                    and subtask_info.num_reschedules < subtask_info.max_reschedules
                ):
                    logger.error(
                        "Reschedule subtask %s due to %s",
                        subtask_info.subtask.subtask_id,
                        ex,
                    )
                    subtask_info.num_reschedules += 1
                    await self._queueing_ref.add_subtasks(
                        [subtask_info.subtask],
                        [subtask_info.subtask.priority or tuple()],
                        exclude_bands=set(subtask_info.band_futures.keys()),
                    )
                else:
                    raise ex
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                if (
                    subtask_info.subtask.retryable
                    and subtask_info.num_reschedules < subtask_info.max_reschedules
                ):
                    logger.error(
                        "Failed to reschedule subtask %s, "
                        "num_reschedules: %s, max_reschedules: %s, unhandled exception: %s",
                        subtask_info.subtask.subtask_id,
                        subtask_info.num_reschedules,
                        subtask_info.max_reschedules,
                        ex,
                    )
                raise ex
            finally:
                # make sure slot is released before marking tasks as finished
                await self._global_resource_ref.release_subtask_resource(
                    band,
                    subtask_info.subtask.session_id,
                    subtask_info.subtask.subtask_id,
                )
                logger.debug(
                    "Slot released for band %s after subtask %s",
                    band,
                    subtask_info.subtask.subtask_id,
                )
                # We should call submit_subtasks after the resource is released.
                # If submit_subtasks runs before release_subtask_resource
                # then the rescheduled subtask may not be submitted due to
                # no available resource. The mars will hangs.
                if subtask_info.num_reschedules > 0:
                    await self._queueing_ref.submit_subtasks.tell()

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Union[float, int] = None
    ):
        kill_timeout = kill_timeout or self._subtask_cancel_timeout
        logger.info(
            "Start to cancel subtasks %s, kill timeout is %s.",
            subtask_ids,
            kill_timeout,
        )
        queued_subtask_ids = []
        single_cancel_tasks = []

        task_api = await self._get_task_api()

        async def cancel_single_task(subtask, raw_tasks, cancel_tasks):
            if cancel_tasks:
                await asyncio.wait(cancel_tasks)
            if raw_tasks:
                dones, _ = await asyncio.wait(raw_tasks)
            else:
                dones = []
            if not dones or all(fut.cancelled() for fut in dones):
                await task_api.set_subtask_result(
                    SubtaskResult(
                        subtask_id=subtask.subtask_id,
                        session_id=subtask.session_id,
                        task_id=subtask.task_id,
                        stage_id=subtask.stage_id,
                        status=SubtaskStatus.cancelled,
                    )
                )

        for subtask_id in subtask_ids:
            if subtask_id not in self._subtask_infos:
                # subtask may already finished or not submitted at all
                logger.info(
                    "Skip cancel subtask %s, it may already finished or not submitted at all",
                    subtask_id,
                )
                continue

            subtask_info = self._subtask_infos[subtask_id]
            raw_tasks_to_cancel = list(subtask_info.band_futures.values())

            if not raw_tasks_to_cancel:
                queued_subtask_ids.append(subtask_id)
                single_cancel_tasks.append(
                    asyncio.create_task(
                        cancel_single_task(subtask_info.subtask, [], [])
                    )
                )
            else:
                cancel_tasks = []
                for band in subtask_info.band_futures.keys():
                    execution_ref = await self._get_execution_ref(band)
                    cancel_tasks.append(
                        asyncio.create_task(
                            execution_ref.cancel_subtask(
                                subtask_id, kill_timeout=kill_timeout
                            )
                        )
                    )
                single_cancel_tasks.append(
                    asyncio.create_task(
                        cancel_single_task(
                            subtask_info.subtask, raw_tasks_to_cancel, cancel_tasks
                        )
                    )
                )
        if queued_subtask_ids:
            # Don't use `finish_subtasks` because it may remove queued
            await self._queueing_ref.remove_queued_subtasks(queued_subtask_ids)
        if single_cancel_tasks:
            yield asyncio.wait(single_cancel_tasks)

        for subtask_id in subtask_ids:
            subtask_info = self._subtask_infos.pop(subtask_id, None)
            if subtask_info is not None:
                self._subtask_summaries[subtask_id] = subtask_info.to_summary(
                    is_finished=True, is_cancelled=True
                )
        await self._queueing_ref.submit_subtasks.tell()
        logger.info("Subtasks %s canceled.", subtask_ids)

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
