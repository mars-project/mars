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

import asyncio
from typing import List, Dict, Union, Type, TypeVar

from .... import oscar as mo
from ....lib.aio import alru_cache
from ...core import NodeRole, BandType
from .core import AbstractClusterAPI

APIType = TypeVar('APIType', bound='ClusterAPI')


class ClusterAPI(AbstractClusterAPI):
    def __init__(self, address: str):
        self._address = address
        self._locator_ref = None
        self._uploader_ref = None
        self._node_info_ref = None

    async def _init(self):
        from ..locator import SupervisorLocatorActor
        from ..uploader import NodeInfoUploaderActor
        from ..supervisor.node_info import NodeInfoCollectorActor

        self._locator_ref = await mo.actor_ref(SupervisorLocatorActor.default_uid(),
                                               address=self._address)
        self._uploader_ref = await mo.actor_ref(NodeInfoUploaderActor.default_uid(),
                                                address=self._address)
        [self._node_info_ref] = await self.get_supervisor_refs(
            [NodeInfoCollectorActor.default_uid()])

    @classmethod
    @alru_cache(cache_exceptions=False)
    async def create(cls: Type[APIType], address: str) -> APIType:
        api_obj = cls(address)
        await api_obj._init()
        return api_obj

    async def get_supervisors(self, watch=False) -> List[str]:
        """
        Get or watch supervisor addresses

        Returns
        -------
        out
            list of
        """
        if watch:
            return await self._locator_ref.watch_supervisors()
        else:
            return await self._locator_ref.get_supervisors()

    async def get_supervisors_by_keys(self, keys: List[str], watch: bool = False) -> List[str]:
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
        if not watch:
            get_supervisor = self._locator_ref.get_supervisor
            return await get_supervisor.batch(
                *(get_supervisor.delay(k) for k in keys)
            )
        else:
            return await self._locator_ref.watch_supervisors_by_keys(keys)

    async def get_supervisor_refs(self, uids: List[str], watch: bool = False) -> List[mo.ActorRef]:
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
        addrs = await self.get_supervisors_by_keys(uids, watch=watch)
        return await asyncio.gather(*[
            mo.actor_ref(uid, address=addr) for addr, uid in zip(addrs, uids)
        ])

    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, state: bool = False) -> List[Dict[str, Dict]]:
        """
        Watch changes of workers

        Returns
        -------
        out: List[Dict[str, Dict]]
            dict of worker resources by addresses and bands
        """
        return await self._node_info_ref.watch_nodes(
            role, env=env, resource=resource, state=state)

    async def get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                             env: bool = False, resource: bool = False, state: bool = False):
        """
        Get worker info

        Parameters
        ----------
        nodes
            address of nodes
        role
            roles of nodes
        env
            receive env info
        resource
            receive resource info
        state
            receive state info

        Returns
        -------
        out: Dict
            info of worker
        """
        return await self._node_info_ref.get_nodes_info(
            nodes=nodes, role=role, env=env, resource=resource, state=state)

    async def get_all_bands(self, role: NodeRole = None, watch: bool = False) -> Dict[BandType, int]:
        """
        Get all bands that can be used for computation.

        Returns
        -------
        band_to_slots : dict
            Band to n_slot.
        """
        if watch:
            return await self._node_info_ref.watch_all_bands(role)
        return await self._node_info_ref.get_all_bands(role)

    async def get_bands(self):
        """
        Get bands that can be used for computation on current node.

        Returns
        -------
        band_to_slots : dict
            Band to n_slot.
        """
        return await self._uploader_ref.get_bands()

    async def set_state_value(self, key: str, value: Union[List, Dict]):
        """
        Set state value for current node

        Parameters
        ----------
        key
            state key to set
        value
            value of the state
        """
        await self._uploader_ref.set_state_value.tell(key, value)

    async def set_band_resource(self, band: str, values: Dict):
        """
        Set resource usage for a band

        Parameters
        ----------
        band
        values

        Returns
        -------

        """
        await self._uploader_ref.set_band_resource.tell(band, values)

    async def mark_node_ready(self):
        """
        Mark current node ready for work loads
        """
        await self._uploader_ref.mark_node_ready()

    async def wait_all_supervisors_ready(self):
        """
        Wait till all expected supervisors are ready
        """
        await self._locator_ref.wait_all_supervisors_ready()


class MockClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls: Type[APIType], address: str, **kw) -> APIType:
        from ..locator import SupervisorLocatorActor
        from ..uploader import NodeInfoUploaderActor
        from ..supervisor.node_info import NodeInfoCollectorActor

        dones, _ = await asyncio.wait([
            mo.create_actor(SupervisorLocatorActor, 'fixed', address,
                            uid=SupervisorLocatorActor.default_uid(),
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
