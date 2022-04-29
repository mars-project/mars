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


from typing import List, Dict
from ....resource import Resource, build_band_resources


class _CommonPrivateConfigMixin:
    """This class should ONLY provide the common private APIs for all backend."""

    def _get_band_resources(self) -> List[Dict[str, Resource]]:
        """Get the band resources from config."""
        config = self._execution_config[self.backend]
        return build_band_resources(
            n_worker=config["n_worker"],
            n_cpu=config["n_cpu"],
            mem_bytes=config["mem_bytes"],
            cuda_devices=config["cuda_devices"],
        )
