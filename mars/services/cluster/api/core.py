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

from abc import abstractmethod
from typing import List, Dict, Set

from ....typing import BandType
from ...core import NodeRole
from ..core import NodeStatus


class AbstractClusterAPI:
    @staticmethod
    def _calc_statuses(statuses: Set[NodeStatus] = None,
                       exclude_statuses: Set[NodeStatus] = None) -> Set[NodeStatus]:
        if statuses:
            return statuses
        elif exclude_statuses:
            return set(NodeStatus.__members__.values()).difference(exclude_statuses)
        else:
            return {NodeStatus.READY}

    @abstractmethod
    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        """
        Get supervisor addresses

        Returns
        -------
        out
            list of supervisors
        """

    @abstractmethod
    async def watch_supervisors(self):
        """
        Watch supervisor addresses

        Returns
        -------
        out
            generator of list of supervisors
        """

    @abstractmethod
    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, detail: bool = False,
                          statuses: Set[NodeStatus] = None,
                          exclude_statuses: Set[NodeStatus] = None) -> List[Dict[str, Dict]]:
        """
        Watch changes of workers

        Returns
        -------
        out: List[Dict[str, Dict]]
            dict of worker resources by addresses and bands
        """

    @abstractmethod
    async def get_nodes_info(self, nodes: List[str] = None,
                             role: NodeRole = None,
                             env: bool = False,
                             resource: bool = False,
                             detail: bool = False,
                             statuses: Set[NodeStatus] = None,
                             exclude_statuses: Set[NodeStatus] = None):
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
        detail
            receive detail info

        Returns
        -------
        out: Dict
            info of worker
        """

    @abstractmethod
    async def get_all_bands(self, role: NodeRole = None,
                            statuses: Set[NodeStatus] = None,
                            exclude_statuses: Set[NodeStatus] = None) -> Dict[BandType, int]:
        """
        Get all bands that can be used for computation.

        Returns
        -------
        band_to_slots : dict
            Band to n_slot.
        """

    @abstractmethod
    async def watch_all_bands(self, role: NodeRole = None,
                              statuses: Set[NodeStatus] = None,
                              exclude_statuses: Set[NodeStatus] = None):
        """
        Watch all bands that can be used for computation.

        Returns
        -------
        band_to_slots : dict
            Band to n_slot.
        """

    @abstractmethod
    async def get_mars_versions(self) -> List[str]:
        """
        Get versions used in current Mars cluster

        Returns
        -------
        version_list : list
            List of versions
        """
