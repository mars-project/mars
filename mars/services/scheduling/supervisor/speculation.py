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
from typing import Dict

from ....utils import parse_readable_size, create_task_with_error_log
from ..errors import NoAvailableBand
from .manager import SubtaskScheduleInfo

logger = logging.getLogger(__name__)

# the default times for speculative subtask execution.
DEFAULT_SUBTASK_SPECULATION_THRESHOLD = 0.75
DEFAULT_SUBTASK_SPECULATION_INTERVAL = 5  # time unit: seconds
DEFAULT_SUBTASK_SPECULATION_MIN_TASK_RUNTIME = 3
DEFAULT_SUBTASK_SPECULATION_MULTIPLIER = 1.5
DEFAULT_SUBTASK_MAX_CONCURRENT_RUN = 3


class SpeculativeScheduler:
    _grouped_unfinished_subtasks: Dict[
        str, Dict[str, SubtaskScheduleInfo]
    ]  # key is subtask logic key
    _grouped_finished_subtasks: Dict[
        str, Dict[str, SubtaskScheduleInfo]
    ]  # key is subtask logic key

    def __init__(
        self, queueing_ref, global_resource_ref, speculation_config: Dict[str, any]
    ):
        self._grouped_unfinished_subtasks = defaultdict(dict)
        self._grouped_finished_subtasks = defaultdict(dict)
        self._queueing_ref = queueing_ref
        self._global_resource_ref = global_resource_ref
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
            self._speculation_execution_task = create_task_with_error_log(
                self._speculative_execution_loop()
            )
            logger.info(
                "Speculative execution started with config %s.",
                self._speculation_config,
            )

    async def stop(self):
        if self._subtask_speculation_enabled:
            self._speculation_execution_task.cancel()
            try:
                await self._speculation_execution_task
            except asyncio.CancelledError:
                pass
            logger.info("Speculative execution stopped.")

    def add_subtask(self, subtask_info: SubtaskScheduleInfo):
        # duplicate subtask add will be handled in `_speculative_execution`.
        subtask = subtask_info.subtask
        self._grouped_unfinished_subtasks[subtask.logic_key][
            subtask.subtask_id
        ] = subtask_info

    def finish_subtask(self, subtask_info: SubtaskScheduleInfo):
        subtask = subtask_info.subtask
        grouped_finished_subtasks = self._grouped_finished_subtasks[subtask.logic_key]
        grouped_finished_subtasks[subtask.subtask_id] = subtask_info
        self._grouped_unfinished_subtasks[subtask.logic_key].pop(
            subtask.subtask_id, None
        )
        if len(grouped_finished_subtasks) == subtask.logic_parallelism:
            self._grouped_finished_subtasks.pop(subtask.logic_key)
            self._grouped_unfinished_subtasks.pop(subtask.logic_key, None)
            logger.info(
                "Subtask group with logic key %s parallelism %s finished.",
                subtask.logic_key,
                subtask.logic_parallelism,
            )

    async def _speculative_execution_loop(self):
        while True:
            # check subtasks in the same group which has same logic key periodically, if some subtasks hasn't been
            # finished in a considerably longer duration, then those subtasks maybe slow/hang subtasks, try resubmit
            # it to other bands too.
            await asyncio.sleep(self._subtask_speculation_interval)
            await self._speculative_execution()

    async def _speculative_execution(self):
        for logic_key, subtask_infos_dict in dict(
            self._grouped_finished_subtasks
        ).items():
            if not subtask_infos_dict:  # pragma: no cover
                continue
            subtask_infos = subtask_infos_dict.values()
            one_subtask = next(iter(subtask_infos)).subtask
            parallelism = one_subtask.logic_parallelism
            spec_threshold = max(
                1, int(self._subtask_speculation_threshold * parallelism)
            )
            # if finished subtasks reached the spec_threshold, try to find slow/hang unfinished subtasks
            if parallelism > len(subtask_infos) >= spec_threshold:
                unfinished_subtask_infos = self._grouped_unfinished_subtasks[
                    logic_key
                ].values()
                # sort finished subtasks by running time
                duration_array = np.sort(
                    np.array(
                        [info.end_time - info.start_time for info in subtask_infos]
                    )
                )
                median = np.percentile(duration_array, 50)
                duration_threshold = max(
                    median * self._subtask_speculation_multiplier,
                    self._subtask_speculation_min_task_runtime,
                )
                now = time.time()
                # find subtasks whose duration is large enough so that can be took as slow/hang subtasks
                unfinished_subtask_infos = [
                    info
                    for info in unfinished_subtask_infos
                    if info not in subtask_infos
                    and now - info.start_time > duration_threshold
                ]
                if not unfinished_subtask_infos:  # pragma: no cover
                    continue
                exclude_bands = set()
                for info in unfinished_subtask_infos:
                    exclude_bands.update(info.band_futures.keys())
                remaining_resources = (
                    await self._global_resource_ref.get_remaining_resources()
                )
                logger.warning(
                    "%s subtasks in %s for group %s has not been finished in %s seconds on bands %s, "
                    "median duration is %s, average duration for %s finished subtasks "
                    "is %s. trying speculative running. "
                    "Current cluster remaining resources %s",
                    len(unfinished_subtask_infos),
                    parallelism,
                    logic_key,
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
            return
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
                return
            try:
                await self._queueing_ref.add_subtasks(
                    [subtask],
                    [subtask.priority or tuple()],
                    exclude_bands=exclude_bands,
                    random_when_unavailable=False,
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
            except KeyError as e:  # pragma: no cover
                # if the subtask happen to be finished, it's input chunk may got gc, if assigning to band
                # needs to know input meta, we'll get KeyError or something else, just ignore it.
                logger.warning(
                    "Subtask %s may happen to be finished just now, cannot add it to "
                    "subtask queue, got error %s, just ignore it.",
                    subtask.subtask_id,
                    e,
                )
