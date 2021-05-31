# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import logging
from typing import Dict

from ... import oscar as mo
from ...lib.aio import alru_cache
from ..core import BandType
from .core import NodeInfo
from .gather import gather_node_env, gather_node_resource, gather_node_states

logger = logging.getLogger(__name__)

DEFAULT_INFO_UPLOAD_INTERVAL = 1


class NodeInfoUploaderActor(mo.Actor):
    def __init__(self, role=None, dirs=None, interval=None,
                 band_to_slots=None, use_gpu=True):
        self._info = NodeInfo(role=role)

        self._env_uploaded = False
        self._dirs = dirs
        self._band_to_slots = band_to_slots

        self._interval = interval or DEFAULT_INFO_UPLOAD_INTERVAL
        self._upload_task = None
        self._upload_enabled = False

        self._use_gpu = use_gpu

    async def __post_create__(self):
        await self.upload_node_info()

    async def __pre_destroy__(self):
        self._upload_task.cancel()

    @alru_cache(cache_exceptions=False)
    async def _get_node_info_ref(self):
        from .locator import SupervisorLocatorActor
        from .supervisor.node_info import NodeInfoCollectorActor

        locator_ref = await mo.actor_ref(
            SupervisorLocatorActor.default_uid(), address=self.address)
        supervisor_addr = await locator_ref.get_supervisor(
            NodeInfoCollectorActor.default_uid())
        return await mo.actor_ref(
            NodeInfoCollectorActor.default_uid(), address=supervisor_addr)

    async def mark_node_ready(self):
        self._upload_enabled = True
        # upload info in time to reduce latency
        await self.upload_node_info(False)

    def is_node_ready(self):
        return self._upload_enabled

    async def upload_node_info(self, call_next: bool = True):
        try:
            if not self._info.env:
                self._info.env = gather_node_env()
            self._info.state.update(gather_node_states(dirs=self._dirs))
            for band, res in gather_node_resource(
                    self._band_to_slots, use_gpu=self._use_gpu).items():
                try:
                    res_dict = self._info.resource[band]
                except KeyError:
                    res_dict = self._info.resource[band] = dict()
                res_dict.update(res)

            if self._upload_enabled:
                node_info_ref = await self._get_node_info_ref()
                await node_info_ref.update_node_info(
                    address=self.address, role=self._info.role,
                    env=self._info.env if not self._env_uploaded else None,
                    resource=self._info.resource, state=self._info.state,
                )
                self._env_uploaded = True
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
            logger.exception(f'Failed to upload node info')
            raise
        finally:
            if call_next:
                self._upload_task = self.ref().upload_node_info.tell_delay(delay=self._interval)

    def get_bands(self) -> Dict[BandType, int]:
        band_slots = dict()
        for resource_type, info in self._info.resource.items():
            if resource_type.startswith('numa'):
                # cpu
                band_slots[(self.address, resource_type)] = info['cpu_total']
            else:  # pragma: no cover
                assert resource_type.startswith('gpu')
                band_slots[(self.address, resource_type)] = info['gpu_total']
        return band_slots

    def set_state_value(self, key, value):
        self._info.state[key] = value

    def set_band_resource(self, band: str, values: Dict):
        self._info.resource[band].update(values)
