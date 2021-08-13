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
from typing import List, Dict, Optional, Set, Type, TypeVar

from .... import oscar as mo
from ....lib.aio import alru_cache
from ....typing import BandType
from ...core import NodeRole
from ..core import watch_method, NodeStatus, WorkerSlotInfo, QuotaInfo, \
    DiskInfo, StorageInfo
from .core import AbstractClusterAPI

APIType = TypeVar('APIType', bound='ClusterAPI')
logger = logging.getLogger(__name__)


class ClusterAPI(AbstractClusterAPI):
    def __init__(self, address: str):
        self._address = address
        self._locator_ref = None
        self._uploader_ref = None
        self._node_info_ref = None

    async def _init(self):
        from ..locator import SupervisorLocatorActor
        from ..uploader import NodeInfoUploaderActor

        self._locator_ref = await mo.actor_ref(SupervisorLocatorActor.default_uid(),
                                               address=self._address)
        self._uploader_ref = await mo.actor_ref(NodeInfoUploaderActor.default_uid(),
                                                address=self._address)

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls: Type[APIType], address: str) -> APIType:
        api_obj = cls(address)
        await api_obj._init()
        return api_obj

    @alru_cache(cache_exceptions=False)
    async def _get_node_info_ref(self):
        from ..supervisor.node_info import NodeInfoCollectorActor
        [node_info_ref] = await self.get_supervisor_refs(
            [NodeInfoCollectorActor.default_uid()])
        return node_info_ref

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        return await self._locator_ref.get_supervisors(filter_ready=filter_ready)

    @watch_method
    async def watch_supervisors(self,
                                version: Optional[int] = None):
        return await self._locator_ref.watch_supervisors(version=version)

    async def get_supervisors_by_keys(self, keys: List[str]) -> List[str]:
        """
        Get supervisor address hosting the specified key

        Parameters
        ----------
        keys
            key for a supervisor address
        watch
            if True, will watch changes of supervisor changes

        Returns
        -------
        out
            addresses of the supervisor
        """
        get_supervisor = self._locator_ref.get_supervisor
        return await get_supervisor.batch(
            *(get_supervisor.delay(k) for k in keys)
        )

    @watch_method
    async def watch_supervisors_by_keys(self, keys: List[str],
                                        version: Optional[int] = None):
        return await self._locator_ref.watch_supervisors_by_keys(keys, version=version)

    async def get_supervisor_refs(self, uids: List[str]) -> List[mo.ActorRef]:
        """
        Get actor references hosting the specified actor uid

        Parameters
        ----------
        uids
            uids for a supervisor address
        watch
            if True, will watch changes of supervisor changes

        Returns
        -------
        out : List[mo.ActorRef]
            references of the actors
        """
        addrs = await self.get_supervisors_by_keys(uids)
        return await asyncio.gather(*[
            mo.actor_ref(uid, address=addr) for addr, uid in zip(addrs, uids)
        ])

    async def watch_supervisor_refs(self, uids: List[str]):
        async for addrs in self.watch_supervisors_by_keys(uids):
            yield await asyncio.gather(*[
                mo.actor_ref(uid, address=addr) for addr, uid in zip(addrs, uids)
            ])

    @watch_method
    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, detail: bool = False,
                          version: Optional[int] = None,
                          statuses: Set[NodeStatus] = None,
                          exclude_statuses: Set[NodeStatus] = None) -> List[Dict[str, Dict]]:
        statuses = self._calc_statuses(statuses, exclude_statuses)
        node_info_ref = await self._get_node_info_ref()
        return await node_info_ref.watch_nodes(
            role, env=env, resource=resource, detail=detail,
            statuses=statuses, version=version)

    async def get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                             env: bool = False, resource: bool = False, detail: bool = False,
                             statuses: Set[NodeStatus] = None,
                             exclude_statuses: Set[NodeStatus] = None):
        statuses = self._calc_statuses(statuses, exclude_statuses)
        node_info_ref = await self._get_node_info_ref()
        return await node_info_ref.get_nodes_info(
            nodes=nodes, role=role, env=env, resource=resource,
            detail=detail, statuses=statuses)

    async def set_node_status(self, node: str, role: NodeRole, status: NodeStatus):
        """
        Set status of node

        Parameters
        ----------
        node : str
            address of node
        role: NodeRole
            role of node
        status : NodeStatus
            status of node
        """
        node_info_ref = await self._get_node_info_ref()
        await node_info_ref.update_node_info(node, role, status=status)

    async def get_all_bands(self, role: NodeRole = None,
                            statuses: Set[NodeStatus] = None,
                            exclude_statuses: Set[NodeStatus] = None) -> Dict[BandType, int]:
        statuses = self._calc_statuses(statuses, exclude_statuses)
        node_info_ref = await self._get_node_info_ref()
        return await node_info_ref.get_all_bands(role, statuses=statuses)

    @watch_method
    async def watch_all_bands(self, role: NodeRole = None,
                              version: Optional[int] = None,
                              statuses: Set[NodeStatus] = None,
                              exclude_statuses: Set[NodeStatus] = None):
        statuses = self._calc_statuses(statuses, exclude_statuses)
        node_info_ref = await self._get_node_info_ref()
        return await node_info_ref.watch_all_bands(
            role, statuses=statuses, version=version)

    async def get_mars_versions(self) -> List[str]:
        node_info_ref = await self._get_node_info_ref()
        return await node_info_ref.get_mars_versions()

    async def get_bands(self) -> Dict:
        """
        Get bands that can be used for computation on current node.

        Returns
        -------
        band_to_slots : dict
            Band to n_slot.
        """
        return await self._uploader_ref.get_bands()

    async def mark_node_ready(self):
        """
        Mark current node ready for work loads
        """
        await self._uploader_ref.mark_node_ready()

    async def wait_node_ready(self):
        """
        Wait current node to be ready
        """
        await self._uploader_ref.wait_node_ready()

    async def wait_all_supervisors_ready(self):
        """
        Wait till all expected supervisors are ready
        """
        await self._locator_ref.wait_all_supervisors_ready()

    async def set_band_slot_infos(self, band_name: str,
                                  slot_infos: List[WorkerSlotInfo]):
        await self._uploader_ref.set_band_slot_infos.tell(band_name, slot_infos)

    async def set_band_quota_info(self, band_name: str,
                                  quota_info: QuotaInfo):
        await self._uploader_ref.set_band_quota_info.tell(band_name, quota_info)

    async def set_node_disk_info(self, disk_info: List[DiskInfo]):
        await self._uploader_ref.set_node_disk_info(disk_info)

    @mo.extensible
    async def set_band_storage_info(self, band_name: str, storage_info: StorageInfo):
        await self._uploader_ref.set_band_storage_info(band_name, storage_info)


class MockClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls: Type[APIType], address: str, **kw) -> APIType:
        from ..supervisor.locator import SupervisorPeerLocatorActor
        from ..uploader import NodeInfoUploaderActor
        from ..supervisor.node_info import NodeInfoCollectorActor

        dones, _ = await asyncio.wait([
            mo.create_actor(SupervisorPeerLocatorActor, 'fixed', address,
                            uid=SupervisorPeerLocatorActor.default_uid(),
                            address=address),
            mo.create_actor(NodeInfoCollectorActor,
                            uid=NodeInfoCollectorActor.default_uid(),
                            address=address),
            mo.create_actor(NodeInfoUploaderActor, NodeRole.WORKER,
                            interval=kw.get('upload_interval'),
                            band_to_slots=kw.get('band_to_slots'),
                            use_gpu=kw.get('use_gpu', False),
                            uid=NodeInfoUploaderActor.default_uid(),
                            address=address),
        ])

        for task in dones:
            try:
                task.result()
            except mo.ActorAlreadyExist:  # pragma: no cover
                pass

        api = await super().create(address=address)
        await api.mark_node_ready()
        return api
