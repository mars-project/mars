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
DEFAULT_SUBTASK_MONITOR_INTERVAL = 0 if IN_RAY_CI else 1


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

    def get_n_cpu(self):
        return self._ray_execution_config["n_cpu"]

    def get_n_worker(self):
        return self._ray_execution_config["n_worker"]

    def get_subtask_cancel_timeout(self):
        return self._ray_execution_config["subtask_cancel_timeout"]

    def get_subtask_monitor_interval(self):
        """
        The interval seconds for the monitor task to update progress and
        collect garbage.
        """
        return self._ray_execution_config.get(
            "subtask_monitor_interval", DEFAULT_SUBTASK_MONITOR_INTERVAL
        )

    def get_shuffle_fetch_type(self) -> ShuffleFetchType:
        return ShuffleFetchType.FETCH_BY_INDEX
