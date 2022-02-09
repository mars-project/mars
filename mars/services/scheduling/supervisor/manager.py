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
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....oscar.debug import create_task_with_ex_logged
from ....oscar.backends.message import ProfilingContext
from ....oscar.errors import MarsError
from ....typing import BandType
from ....utils import dataslots, parse_readable_size
from ...subtask import Subtask, SubtaskResult, SubtaskStatus
from ...task import TaskAPI
from ..core import SubtaskScheduleSummary
from ..errors import NoAvailableBand
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
        self._global_slot_ref = None
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
        from ..supervisor import GlobalSlotManagerActor

        self._global_slot_ref = await mo.actor_ref(
            GlobalSlotManagerActor.default_uid(), address=self.address
        )
        self._speculation_execution_scheduler = SpeculativeScheduler(
            self._queueing_ref, self._global_slot_ref, self._speculation_config
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
                if subtask.subtask_id in self._subtask_infos:
                    raise Exception(f"Subtask {subtask.subtask_id} already added.")
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
        if subtask_id not in self._subtask_infos:
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
                profiling_context = (
                    ProfilingContext(subtask_info.subtask.task_id)
                    if extra_config and extra_config.get("enable_profiling")
                    else None
                )
                logger.debug("Start run subtask %s in band %s.", subtask_id, band)
                task = asyncio.create_task(
                    execution_ref.run_subtask.options(
                        profiling_context=profiling_context
                    ).send(subtask_info.subtask, band[1], self.address)
                )
                subtask_info.band_futures[band] = task
                subtask_info.start_time = time.time()
                self._speculation_execution_scheduler.add_subtask(subtask_info)
                result = yield task
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
                    await self._queueing_ref.submit_subtasks.tell()
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
                await self._global_slot_ref.release_subtask_slots(
                    band,
                    subtask_info.subtask.session_id,
                    subtask_info.subtask.subtask_id,
                )
                logger.debug(
                    "Slot released for band %s after subtask %s",
                    band,
                    subtask_info.subtask.subtask_id,
                )

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


# the default times to speculative subtask execution.
DEFAULT_SUBTASK_SPECULATION_THRESHOLD = 0.75
DEFAULT_SUBTASK_SPECULATION_INTERVAL = 5  # time unit: seconds
DEFAULT_SUBTASK_SPECULATION_MIN_TASK_RUNTIME = 3
DEFAULT_SUBTASK_SPECULATION_MULTIPLIER = 1.5
DEFAULT_SUBTASK_MAX_CONCURRENT_RUN = 3


class SpeculativeScheduler:
    _grouped_unfinished_subtasks: Dict[
        str, Dict[str, SubtaskScheduleInfo]
    ]  # key is subtask logic id
    _grouped_finished_subtasks: Dict[
        str, Dict[str, SubtaskScheduleInfo]
    ]  # key is subtask logic id
    _speculation_execution_scheduler: Optional["SpeculativeScheduler"]

    def __init__(
        self, queueing_ref, global_slot_ref, speculation_config: Dict[str, any]
    ):
        self._grouped_unfinished_subtasks = defaultdict(dict)
        self._grouped_finished_subtasks = defaultdict(dict)
        self._queueing_ref = queueing_ref
        self._global_slot_ref = global_slot_ref
        self._speculation_config = speculation_config
        self._subtask_speculation_enabled = speculation_config.get("enabled", False)
        assert self._subtask_speculation_enabled in (True, False)
        self._subtask_speculation_dry = speculation_config.get("dry", False)
        self._subtask_speculation_threshold = parse_readable_size(
            speculation_config.get("threshold", DEFAULT_SUBTASK_SPECULATION_THRESHOLD)
        )[0]
        self._subtask_speculation_interval = speculation_config.get(
            "interval", DEFAULT_SUBTASK_SPECULATION_INTERVAL
        )
        self._subtask_speculation_min_task_runtime = speculation_config.get(
            "min_task_runtime", DEFAULT_SUBTASK_SPECULATION_MIN_TASK_RUNTIME
        )
        self._subtask_speculation_multiplier = speculation_config.get(
            "multiplier", DEFAULT_SUBTASK_SPECULATION_MULTIPLIER
        )
        self._subtask_speculation_max_concurrent_run = speculation_config.get(
            "max_concurrent_run", DEFAULT_SUBTASK_MAX_CONCURRENT_RUN
        )
        if self._subtask_speculation_enabled:
            assert 1 >= self._subtask_speculation_threshold > 0
            assert self._subtask_speculation_interval > 0
            assert self._subtask_speculation_min_task_runtime > 0
            assert self._subtask_speculation_multiplier > 0
            assert self._subtask_speculation_max_concurrent_run > 0
        self._speculation_execution_task = None

    async def start(self):
        if self._subtask_speculation_enabled:
            self._speculation_execution_task = create_task_with_ex_logged(
                    self._speculative_execution()
                )
            logger.info(
                "Speculative execution started with config %s.", self._speculation_config
            )

    async def stop(self):
        if self._subtask_speculation_enabled:
            self._speculation_execution_task.cancel()
            await self._speculation_execution_task
            logger.info("Speculative execution stopped.")

    def add_subtask(self, subtask_info: SubtaskScheduleInfo):
        # duplicate subtask add will be handled in `_speculative_execution`.
        subtask = subtask_info.subtask
        self._grouped_unfinished_subtasks[subtask.logic_id][
            subtask.subtask_id
        ] = subtask_info

    def finish_subtask(self, subtask_info: SubtaskScheduleInfo):
        subtask = subtask_info.subtask
        grouped_finished_subtasks = self._grouped_finished_subtasks[subtask.logic_id]
        grouped_finished_subtasks[subtask.subtask_id] = subtask_info
        if len(grouped_finished_subtasks) == subtask.parallelism:
            self._grouped_finished_subtasks.pop(subtask.logic_id)
            self._grouped_unfinished_subtasks.pop(subtask.logic_id, None)
            logger.info(
                "Subtask group with logic id %s parallelism %s finished.",
                subtask.logic_id,
                subtask.parallelism,
            )

    async def _speculative_execution(self):
        while True:
            await asyncio.sleep(self._subtask_speculation_interval)
            for logic_id, subtask_infos_dict in self._grouped_finished_subtasks.items():
                if subtask_infos_dict:
                    subtask_infos = subtask_infos_dict.values()
                    one_subtask = next(iter(subtask_infos)).subtask
                    parallelism = one_subtask.parallelism
                    spec_threshold = max(
                        1, int(self._subtask_speculation_threshold * parallelism)
                    )
                    if parallelism > len(subtask_infos) >= spec_threshold:
                        unfinished_subtask_infos = self._grouped_unfinished_subtasks[
                            logic_id
                        ].values()
                        duration_array = np.sort(
                            np.array(
                                [
                                    info.end_time - info.start_time
                                    for info in subtask_infos
                                ]
                            )
                        )
                        median = np.percentile(duration_array, 50)
                        duration_threshold = max(
                            median * self._subtask_speculation_multiplier,
                            self._subtask_speculation_min_task_runtime,
                        )
                        now = time.time()
                        unfinished_subtask_infos = [
                            info
                            for info in unfinished_subtask_infos
                            if info not in subtask_infos
                            and now - info.start_time > duration_threshold
                        ]
                        if unfinished_subtask_infos:
                            exclude_bands = set()
                            for info in unfinished_subtask_infos:
                                for band in info.band_futures.keys():
                                    exclude_bands.add(band)
                            remaining_resources = (
                                await self._global_slot_ref.get_remaining_slots()
                            )
                            logger.warning(
                                "%s subtasks in %s for group %s has not been finished in %s seconds on bands %s, "
                                "median duration is %s, average duration for %s finished subtasks "
                                "is %s. trying speculative running. "
                                "Current cluster remaining resources %s",
                                len(unfinished_subtask_infos),
                                parallelism,
                                logic_id,
                                duration_threshold,
                                exclude_bands,
                                median,
                                len(subtask_infos),
                                duration_array.mean(),
                                remaining_resources,
                            )
                            # TODO(chaokunyang) If too many subtasks got stale on same node, mark the node as slow node.
                            for subtask_info in unfinished_subtask_infos:
                                subtask = subtask_info.subtask
                                if subtask.retryable:
                                    logger.warning(
                                        "Subtask %s has not been finished in %s seconds on bands %s, "
                                        "trying speculative running.",
                                        subtask.subtask_id,
                                        now - subtask_info.start_time,
                                        list(subtask_info.band_futures.keys()),
                                    )
                                    await self._submit_speculative_subtask(
                                        subtask_info, exclude_bands
                                    )
                                else:
                                    logger.warning(
                                        "Unretryable subtask %s has not been finished in %s seconds "
                                        "on bands %s, median duration is %s, it may hang.",
                                        subtask.subtask_id,
                                        (now - subtask_info.start_time),
                                        list(subtask_info.band_futures.keys()),
                                        median,
                                    )
                            await self._queueing_ref.submit_subtasks.tell()

    async def _submit_speculative_subtask(self, subtask_info, exclude_bands):
        subtask = subtask_info.subtask
        if (
            subtask_info.num_speculative_concurrent_run
            == self._subtask_speculation_max_concurrent_run
        ):
            logger.debug(
                "Subtask %s speculative run has reached max limit %s, "
                "won't submit another speculative run.",
                subtask.subtask_id,
                self._subtask_speculation_max_concurrent_run,
            )
        else:
            if not self._subtask_speculation_dry:
                if (
                    len(subtask_info.band_futures)
                    < subtask_info.num_speculative_concurrent_run + 1
                ):
                    # ensure same subtask won't be submitted to same worker.
                    logger.info(
                        "Speculative execution for subtask %s has not been submitted to worker,"
                        "waiting for being submitted to worker."
                        "Cluster resources may be not enough after excluded %s",
                        subtask.subtask_id,
                        exclude_bands,
                    )
                else:
                    try:
                        await self._queueing_ref.add_subtasks(
                            [subtask],
                            [subtask.priority or tuple()],
                            exclude_bands=exclude_bands,
                            exclude_bands_force=True,
                        )
                        logger.info(
                            "Added subtask %s to queue excluded from %s.",
                            subtask.subtask_id,
                            exclude_bands,
                        )
                        subtask_info.num_speculative_concurrent_run += 1
                        if (
                            subtask_info.num_speculative_concurrent_run
                            == self._subtask_speculation_max_concurrent_run
                        ):
                            logger.info(
                                "Subtask %s reached max speculative execution: %s",
                                subtask.subtask_id,
                                self._subtask_speculation_max_concurrent_run,
                            )
                    except NoAvailableBand:
                        logger.warning(
                            "No bands available for subtask %s after excluded bands %s, "
                            "try resubmit later.",
                            subtask.subtask_id,
                            exclude_bands,
                        )
                    except KeyError as e:
                        # if the subtask happen to be finished, it's input chunk may got gc, if assigning to band
                        # needs to know input meta, we'll get KeyError or something else, just ignore it.
                        logger.warning(
                            "Subtask %s may happen to be finished just now, cannot add it to"
                            "subtask queue, got error %s, just ignore it.",
                            subtask.subtask_id,
                            e,
                        )
