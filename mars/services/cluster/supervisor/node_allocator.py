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

from typing import Optional

from .... import oscar as mo
from ...core import NodeRole
from ..backends import AbstractClusterBackend, get_cluster_backend


class NodeAllocatorActor(mo.StatelessActor):
    def __init__(self, backend_name: str, lookup_address: str):
        self._backend_name = backend_name
        self._lookup_address = lookup_address
        self._backend: Optional[AbstractClusterBackend] = None

    async def __post_create__(self):
        backend_cls = get_cluster_backend(self._backend_name)
        self._backend = await backend_cls.create(
            NodeRole.WORKER, self._lookup_address, self.address
        )

    async def request_worker(
        self, worker_cpu: int, worker_mem: int, timeout: int = None
    ) -> str:
        return await self._backend.request_worker(
            worker_cpu, worker_mem, timeout=timeout
        )

    async def release_worker(self, address: str):
        await self._backend.release_worker(address)

    async def reconstruct_worker(self, address: str):
        await self._backend.reconstruct_worker(address)
