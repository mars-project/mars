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
import os
from typing import Optional, List, AsyncGenerator

from ... import NodeRole
from ...cluster.backends import AbstractClusterBackend, register_cluster_backend

logger = logging.getLogger(__name__)


@register_cluster_backend
class TestClusterBackend(AbstractClusterBackend):
    name = "test"

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._modify_date = os.path.getmtime(file_path)

    @classmethod
    async def create(
        cls, node_role: NodeRole, lookup_address: Optional[str], pool_address: str
    ) -> "AbstractClusterBackend":
        return TestClusterBackend(lookup_address)

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        with open(self._file_path, "r") as inp_file:
            result = []
            for line in inp_file.read().strip().splitlines(False):
                line_parts = line.rsplit(",", 1)
                if len(line_parts) == 1 or (filter_ready and int(line_parts[1])):
                    result.append(line_parts[0])
            return result

    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        while True:
            mtime = os.path.getmtime(self._file_path)
            if mtime != self._modify_date:
                self._modify_date = mtime
                yield await self.get_supervisors()
            await asyncio.sleep(0.1)

    async def request_worker(
        self, worker_cpu: int = None, worker_mem: int = None, timeout: int = None
    ) -> str:
        raise NotImplementedError

    async def release_worker(self, address: str):
        raise NotImplementedError

    async def reconstruct_worker(self, address: str):
        raise NotImplementedError
