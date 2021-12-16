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

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Type, Any

from ..core import NodeRole


class AbstractClusterBackend(ABC):
    name = None

    @classmethod
    @abstractmethod
    async def create(
        cls, node_role: NodeRole, lookup_address: Optional[str], pool_address: str
    ) -> "AbstractClusterBackend":
        """

        Parameters
        ----------
        node_role
        lookup_address
        pool_address

        Returns
        -------

        """

    @abstractmethod
    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        """
        Watch changes of supervisors

        Returns
        -------
        out : AsyncGenerator[List[str]]
            Generator of list of schedulers
        """

    @abstractmethod
    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        """
        Get list of supervisors

        Parameters
        ----------
        filter_ready : bool
            True if return ready nodes only, or return starting and ready nodes

        Returns
        -------
        out : List[str]
            List of supervisors
        """

    @abstractmethod
    async def request_worker(
        self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None
    ) -> str:
        """
        Create a new worker

        Returns
        -------
        Address of the new created worker
        """

    @abstractmethod
    async def release_worker(self, address: str):
        """
        Return a worker
        """

    @abstractmethod
    async def reconstruct_worker(self, address: str):
        """
        Reconstruct a worker
        """

    async def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get cluster information

        Returns
        -------
        Cluster information is as follows:
        {
            'worker_cpu': 1,
            'worker_mem': 2147483648,
            'config': {
                'services': ['cluster', 'session',],
                'cluster': {
                    'backend': 'fixed',
                },
                'session': {
                    'custom_log_dir': '/tmp/logs',
                },
            }
            'worker_node_to_resources': {
                'mars_cluster_0': {
                    'cpu': 4,
                    'memory': 8589934592
                },
                'mars_cluster_1': {
                    'cpu': 4,
                    'memory': 8589934592
                }
            }
        }
        """
        return {}


_cluster_backend_types: Dict[str, Type[AbstractClusterBackend]] = dict()


def register_cluster_backend(backend: Type[AbstractClusterBackend]):
    _cluster_backend_types[backend.name] = backend
    return backend


def get_cluster_backend(backend_name: str) -> Type[AbstractClusterBackend]:
    return _cluster_backend_types[backend_name]
