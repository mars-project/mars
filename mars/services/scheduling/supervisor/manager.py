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
from ....metrics import Metrics
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
    subtask: Subtask
    band_futures: Dict[BandType, asyncio.Future] = field(default_factory=dict)
    start_time: int = -1
    end_time: int = -1
    cancel_pending: bool = False
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

        self._submitted_subtask_count = Metrics.counter(
            "mars.scheduling.submitted_subtask_count",
            "The count of submitted subtasks to all bands.",
            ("session_id", "task_id", "stage_id"),
        )
        self._finished_subtask_count = Metrics.counter(
            "mars.scheduling.finished_subtask_count",
            "The count of finished subtasks of all bands.",
            ("session_id", "task_id", "stage_id"),
        )
        self._cancelled_subtask_count = Metrics.counter(
            "mars.scheduling.canceled_subtask_count",
            "The count of canceled subtasks of all bands.",
            ("session_id", "task_id", "stage_id"),
        )

        logger.info(
            "Created SubtaskManagerActor with subtask_max_reschedules %s, "
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

    async def _handle_subtask_result(
        self, info: SubtaskScheduleInfo, result: SubtaskResult, band: BandType
    ):
        subtask_id = info.subtask.subtask_id
        async with redirect_subtask_errors(self, [info.subtask], reraise=False):
            try:
                info.band_futures[band].set_result(result)
                if result.error is not None:
                    raise result.error.with_traceback(result.traceback)
                logger.debug("Finished subtask %s with result %s.", subtask_id, result)
            except (OSError, MarsError) as ex:
                # TODO: We should handle ServerClosed Error.
                if (
                    info.subtask.retryable
                    and info.num_reschedules < info.max_reschedules
                ):
                    logger.error(
                        "Reschedule subtask %s due to %s",
                        info.subtask.subtask_id,
                        ex,
                    )
                    info.num_reschedules += 1
                    await self._queueing_ref.add_subtasks(
                        [info.subtask],
                        [info.subtask.priority or tuple()],
                        exclude_bands=set(info.band_futures.keys()),
                    )
                else:
                    raise ex
            except asyncio.CancelledError:
                raise
            except BaseException as ex:
                if (
                    info.subtask.retryable
                    and info.num_reschedules < info.max_reschedules
                ):
                    logger.error(
                        "Failed to reschedule subtask %s, "
                        "num_reschedules: %s, max_reschedules: %s, unhandled exception: %s",
                        info.subtask.subtask_id,
                        info.num_reschedules,
                        info.max_reschedules,
                        ex,
                    )
                raise ex
            finally:
                # make sure slot is released before marking tasks as finished
                await self._global_resource_ref.release_subtask_resource(
                    band,
                    info.subtask.session_id,
                    info.subtask.subtask_id,
                )
                logger.debug(
                    "Slot released for band %s after subtask %s",
                    band,
                    info.subtask.subtask_id,
                )
                # We should call submit_subtasks after the resource is released.
                # If submit_subtasks runs before release_subtask_resource
                # then the rescheduled subtask may not be submitted due to
                # no available resource. The mars will hangs.
                if info.num_reschedules > 0:
                    await self._queueing_ref.submit_subtasks.tell()

    async def finish_subtasks(
        self,
        subtask_results: List[SubtaskResult],
        bands: List[BandType] = None,
        schedule_next: bool = True,
    ):
        subtask_ids = [result.subtask_id for result in subtask_results]
        logger.debug("Finished subtasks %s.", subtask_ids)
        band_tasks = defaultdict(lambda: 0)
        bands = bands or [None] * len(subtask_ids)
        for result, subtask_band in zip(subtask_results, bands):
            subtask_id = result.subtask_id
            subtask_info = self._subtask_infos.get(subtask_id, None)

            if subtask_info is not None:
                if subtask_band is not None:
                    await self._handle_subtask_result(
                        subtask_info, result, subtask_band
                    )

                self._finished_subtask_count.record(
                    1,
                    {
                        "session_id": self._session_id,
                        "task_id": subtask_info.subtask.task_id,
                        "stage_id": subtask_info.subtask.stage_id,
                    },
                )
                self._subtask_summaries[subtask_id] = subtask_info.to_summary(
                    is_finished=True,
                    is_cancelled=result.status == SubtaskStatus.cancelled,
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
        if band_tasks:
            await self._queueing_ref.submit_subtasks.tell(dict(band_tasks))

    def _get_subtasks_by_ids(self, subtask_ids: List[str]) -> List[Optional[Subtask]]:
        subtasks = []
        for stid in subtask_ids:
            try:
                subtasks.append(self._subtask_infos[stid].subtask)
            except KeyError:
                subtasks.append(None)
        return subtasks

    @mo.extensible
    async def submit_subtask_to_band(self, subtask_id: str, band: BandType):
        raise NotImplementedError

    @submit_subtask_to_band.batch
    async def batch_submit_subtask_to_band(self, args_list, kwargs_list):
        band_to_subtask_ids = defaultdict(list)
        res_release_delays = []
        for args, kwargs in zip(args_list, kwargs_list):
            subtask_id, band = self.submit_subtask_to_band.bind(*args, **kwargs)
            try:
                info = self._subtask_infos[subtask_id]
                if info.cancel_pending:
                    res_release_delays.append(
                        self._global_resource_ref.release_subtask_resource.delay(
                            band, info.subtask.session_id, info.subtask.subtask_id
                        )
                    )
                    continue
            except KeyError:  # pragma: no cover
                logger.info(
                    "Subtask %s is not in added subtasks set, it may be finished or canceled, skip it.",
                    subtask_id,
                )
                continue
            band_to_subtask_ids[band].append(subtask_id)

        if res_release_delays:
            await self._global_resource_ref.release_subtask_resource.batch(
                *res_release_delays
            )

        for band, subtask_ids in band_to_subtask_ids.items():
            asyncio.create_task(self._submit_subtasks_to_band(band, subtask_ids))

    async def _submit_subtasks_to_band(self, band: BandType, subtask_ids: List[str]):
        execution_ref = await self._get_execution_ref(band)
        delays = []

        async with redirect_subtask_errors(
            self, self._get_subtasks_by_ids(subtask_ids)
        ):
            for subtask_id in subtask_ids:
                subtask_info = self._subtask_infos[subtask_id]
                subtask = subtask_info.subtask
                self._submitted_subtask_count.record(
                    1,
                    {
                        "session_id": self._session_id,
                        "task_id": subtask.task_id,
                        "stage_id": subtask.stage_id,
                    },
                )
                logger.debug("Start run subtask %s in band %s.", subtask_id, band)
                delays.append(
                    execution_ref.run_subtask.delay(subtask, band[1], self.address)
                )
                subtask_info.band_futures[band] = asyncio.Future()
                subtask_info.start_time = time.time()
                self._speculation_execution_scheduler.add_subtask(subtask_info)
            await execution_ref.run_subtask.batch(*delays, send=False)

    async def cancel_subtasks(
        self, subtask_ids: List[str], kill_timeout: Union[float, int] = None
    ):
        kill_timeout = kill_timeout or self._subtask_cancel_timeout
        logger.info(
            "Start to cancel subtasks %s, kill timeout is %s.",
            subtask_ids,
            kill_timeout,
        )

        task_api = await self._get_task_api()

        async def cancel_task_in_band(band):
            cancel_delays = band_to_cancel_delays.get(band) or []
            execution_ref = await self._get_execution_ref(band)
            if cancel_delays:
                await execution_ref.cancel_subtask.batch(*cancel_delays)
            band_futures = band_to_futures.get(band)
            if band_futures:
                await asyncio.wait(band_futures)

        queued_subtask_ids = []
        cancel_tasks = []
        band_to_cancel_delays = defaultdict(list)
        band_to_futures = defaultdict(list)
        for subtask_id in subtask_ids:
            if subtask_id not in self._subtask_infos:
                # subtask may already finished or not submitted at all
                logger.info(
                    "Skip cancel subtask %s, it may already finished or not submitted at all",
                    subtask_id,
                )
                continue

            info = self._subtask_infos[subtask_id]
            info.cancel_pending = True
            raw_tasks_to_cancel = list(info.band_futures.values())

            if not raw_tasks_to_cancel:
                # not submitted yet: mark subtasks as cancelled
                result = SubtaskResult(
                    subtask_id=info.subtask.subtask_id,
                    session_id=info.subtask.session_id,
                    task_id=info.subtask.task_id,
                    stage_id=info.subtask.stage_id,
                    status=SubtaskStatus.cancelled,
                )
                cancel_tasks.append(task_api.set_subtask_result(result))
                queued_subtask_ids.append(subtask_id)
            else:
                for band, future in info.band_futures.items():
                    execution_ref = await self._get_execution_ref(band)
                    band_to_cancel_delays[band].append(
                        execution_ref.cancel_subtask.delay(subtask_id, kill_timeout)
                    )
                    band_to_futures[band].append(future)

        for band in band_to_futures:
            cancel_tasks.append(asyncio.create_task(cancel_task_in_band(band)))

        if queued_subtask_ids:
            # Don't use `finish_subtasks` because it may remove queued
            await self._queueing_ref.remove_queued_subtasks(queued_subtask_ids)

        if cancel_tasks:
            yield asyncio.gather(*cancel_tasks)

        for subtask_id in subtask_ids:
            info = self._subtask_infos.pop(subtask_id, None)
            if info is not None:
                self._subtask_summaries[subtask_id] = info.to_summary(
                    is_finished=True, is_cancelled=True
                )
                self._cancelled_subtask_count.record(
                    1,
                    {
                        "session_id": self._session_id,
                        "task_id": info.subtask.task_id,
                        "stage_id": info.subtask.stage_id,
                    },
                )
        await self._queueing_ref.submit_subtasks.tell()
        logger.info("Subtasks %s cancelled.", subtask_ids)

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
