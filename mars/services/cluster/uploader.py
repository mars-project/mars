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
from collections import defaultdict
from typing import Dict, List

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...storage import StorageLevel
from ...typing import BandType
from .core import NodeInfo, NodeStatus, WorkerSlotInfo, QuotaInfo, \
    DiskInfo, StorageInfo
from .gather import gather_node_env, gather_node_resource, \
    gather_node_details

logger = logging.getLogger(__name__)

DEFAULT_INFO_UPLOAD_INTERVAL = 1


class NodeInfoUploaderActor(mo.Actor):
    _band_slot_infos: Dict[str, List[WorkerSlotInfo]]
    _band_quota_infos: Dict[str, QuotaInfo]
    _disk_infos: List[DiskInfo]
    _band_storage_infos: Dict[str, Dict[StorageLevel, StorageInfo]]

    def __init__(self, role=None, interval=None,
                 band_to_slots=None, use_gpu=True):
        self._info = NodeInfo(role=role)

        self._env_uploaded = False
        self._band_to_slots = band_to_slots

        self._interval = interval or DEFAULT_INFO_UPLOAD_INTERVAL
        self._upload_task = None
        self._upload_enabled = False
        self._node_ready_event = asyncio.Event()

        self._use_gpu = use_gpu

        self._band_slot_infos = dict()
        self._band_quota_infos = dict()
        self._band_storage_infos = defaultdict(dict)
        self._disk_infos = []

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
        if supervisor_addr is None:
            raise ValueError

        return await mo.actor_ref(
            NodeInfoCollectorActor.default_uid(), address=supervisor_addr)

    async def mark_node_ready(self):
        self._upload_enabled = True
        # upload info in time to reduce latency
        await self.upload_node_info(call_next=False, status=NodeStatus.READY)
        self._node_ready_event.set()

    def is_node_ready(self):
        return self._node_ready_event.is_set()

    async def wait_node_ready(self):
        return self._node_ready_event.wait()

    async def upload_node_info(self, call_next: bool = True, status: NodeStatus = None):
        try:
            if not self._info.env:
                self._info.env = await asyncio.to_thread(gather_node_env)
            self._info.detail.update(await asyncio.to_thread(
                gather_node_details,
                disk_infos=self._disk_infos,
                band_storage_infos=self._band_storage_infos,
                band_slot_infos=self._band_slot_infos,
                band_quota_infos=self._band_quota_infos
            ))

            band_resources = await asyncio.to_thread(
                gather_node_resource, self._band_to_slots, use_gpu=self._use_gpu)

            for band, res in band_resources.items():
                try:
                    res_dict = self._info.resource[band]
                except KeyError:
                    res_dict = self._info.resource[band] = dict()
                res_dict.update(res)

            if self._upload_enabled:
                try:
                    node_info_ref = await self._get_node_info_ref()
                    if not self._env_uploaded:
                        status = status or NodeStatus.READY
                    await node_info_ref.update_node_info(
                        address=self.address, role=self._info.role,
                        env=self._info.env if not self._env_uploaded else None,
                        resource=self._info.resource, detail=self._info.detail,
                        status=status
                    )
                    self._env_uploaded = True
                except ValueError:
                    pass
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

    def set_node_disk_info(self, node_disk_info: List[DiskInfo]):
        self._disk_infos = node_disk_info

    def set_band_storage_info(self, band_name: str, storage_info: StorageInfo):
        self._band_storage_infos[band_name][storage_info.storage_level] = storage_info

    def set_band_slot_infos(self, band_name, slot_infos: List[WorkerSlotInfo]):
        self._band_slot_infos[band_name] = slot_infos

    def set_band_quota_info(self, band_name, quota_info: QuotaInfo):
        self._band_quota_infos[band_name] = quota_info
