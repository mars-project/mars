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

import os
import logging
from typing import Dict, List, Union

from .....core.operand import ShuffleFetchType
from .....resource import Resource
from ..api import ExecutionConfig, register_config_cls
from ..utils import get_band_resources_from_config


logger = logging.getLogger(__name__)

IN_RAY_CI = os.environ.get("MARS_CI_BACKEND", "mars") == "ray"
# The default interval seconds to update progress and collect garbage.
DEFAULT_MONITOR_INTERVAL_SECONDS = 0 if IN_RAY_CI else 1
DEFAULT_LOG_INTERVAL_SECONDS = 60
DEFAULT_CHECK_SLOW_SUBTASKS_INTERVAL_SECONDS = 120


@register_config_cls
class RayExecutionConfig(ExecutionConfig):
    name = "ray"

    def __init__(self, execution_config: Dict):
        super().__init__(execution_config)
        self._ray_execution_config = execution_config[self.backend]

    def get_band_resources(self):
        """
        Get the band resources from config for generating ray virtual
        resources.
        """
        return get_band_resources_from_config(self._ray_execution_config)

    def get_deploy_band_resources(self) -> List[Dict[str, Resource]]:
        return []

    def get_subtask_max_retries(self):
        return self._ray_execution_config["subtask_max_retries"]

    def get_subtask_num_cpus(self) -> Union[int, float]:
        return self._ray_execution_config.get("subtask_num_cpus", 1)

    def get_subtask_memory(self) -> Union[int, float]:
        return self._ray_execution_config.get("subtask_memory", None)

    def get_n_cpu(self):
        return self._ray_execution_config["n_cpu"]

    def get_n_worker(self):
        return self._ray_execution_config["n_worker"]

    def get_monitor_interval_seconds(self):
        """
        The interval seconds for the monitor task to update progress and
        collect garbage.
        """
        return self._ray_execution_config.get(
            "monitor_interval_seconds", DEFAULT_MONITOR_INTERVAL_SECONDS
        )

    def get_log_interval_seconds(self):
        return self._ray_execution_config.get(
            "log_interval_seconds", DEFAULT_LOG_INTERVAL_SECONDS
        )

    def get_check_slow_subtasks_interval_seconds(self) -> float:
        return self._ray_execution_config.get(
            "check_slow_subtasks_interval_seconds",
            DEFAULT_CHECK_SLOW_SUBTASKS_INTERVAL_SECONDS,
        )

    def get_check_slow_subtask_iqr_ratio(self) -> float:
        # https://en.wikipedia.org/wiki/Box_plot
        # iqr = q3 - q1
        # duration_threshold = q3 + check_slow_subtasks_iqr_ratio * (q3 - q1)
        # So, the value == 3, extremely slow(probably hang); value == 1.5, slow
        return self._ray_execution_config.get("check_slow_subtasks_iqr_ratio", 3)

    def get_shuffle_fetch_type(self) -> ShuffleFetchType:
        return ShuffleFetchType.FETCH_BY_INDEX

    def get_gc_method(self):
        method = self._ray_execution_config.get("gc_method", "submitted")
        assert method in ["submitted", "completed"]
        return method
