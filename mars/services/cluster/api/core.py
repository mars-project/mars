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

from abc import abstractmethod
from typing import List, Dict

from ...core import NodeRole


class AbstractClusterAPI:
    @abstractmethod
    async def get_supervisors(self, watch=False) -> List[str]:
        """
        Get or watch supervisor addresses

        Returns
        -------
        out
            list of
        """

    @abstractmethod
    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, state: bool = False) -> List[Dict[str, Dict]]:
        """
        Watch changes of workers

        Returns
        -------
        out: List[Dict[str, Dict]]
            dict of worker resources by addresses and bands
        """

    @abstractmethod
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
