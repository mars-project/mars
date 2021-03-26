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

from ... import oscar as mo
from ..core import NodeRole

APIType = TypeVar('APIType', bound='ClusterAPI')


class ClusterAPI:
    def __init__(self, address: str):
        self._address = address
        self._locator_ref = None
        self._uploader_ref = None
        self._node_info_ref = None

    async def _init(self):
        from .locator import SupervisorLocatorActor
        from .uploader import NodeInfoUploaderActor
        from .supervisor.node_info import NodeInfoCollectorActor

        self._locator_ref = await mo.actor_ref(SupervisorLocatorActor.default_uid(),
                                               address=self._address)
        self._uploader_ref = await mo.actor_ref(NodeInfoUploaderActor.default_uid(),
                                                address=self._address)
        [self._node_info_ref] = await self.get_supervisor_refs(
            [NodeInfoCollectorActor.default_uid()])

    @classmethod
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

    async def set_state_value(self, key: str, value: Union[List, Dict]):
        await self._uploader_ref.set_state_value(key, value)


class MockClusterAPI(ClusterAPI):
    @classmethod
    async def create(cls: Type[APIType], address: str, **kw) -> APIType:
        from .locator import SupervisorLocatorActor
        from .uploader import NodeInfoUploaderActor
        from .supervisor.node_info import NodeInfoCollectorActor

        dones, _ = await asyncio.wait([
            mo.create_actor(SupervisorLocatorActor, 'fixed', address,
                            uid=SupervisorLocatorActor.default_uid(),
                            address=address),
            mo.create_actor(NodeInfoCollectorActor,
                            uid=NodeInfoCollectorActor.default_uid(),
                            address=address),
            mo.create_actor(NodeInfoUploaderActor, NodeRole.WORKER,
                            interval=kw.get('upload_interval'),
                            uid=NodeInfoUploaderActor.default_uid(),
                            address=address),
        ])

        for task in dones:
            try:
                task.result()
            except mo.ActorAlreadyExist:  # pragma: no cover
                pass

        return await super().create(address=address)
