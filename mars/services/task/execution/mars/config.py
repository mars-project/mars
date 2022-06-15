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

from typing import Dict, List

from .....core.operand.shuffle import ShuffleFetchType
from .....resource import Resource
from ..api import ExecutionConfig, register_config_cls
from ..utils import get_band_resources_from_config


@register_config_cls
class MarsExecutionConfig(ExecutionConfig):
    name = "mars"

    def __init__(self, execution_config: Dict):
        super().__init__(execution_config)
        self._mars_execution_config = execution_config[self.backend]

    def get_deploy_band_resources(self) -> List[Dict[str, Resource]]:
        return get_band_resources_from_config(self._mars_execution_config)

    def get_shuffle_fetch_type(self) -> ShuffleFetchType:
        return ShuffleFetchType.FETCH_BY_KEY
