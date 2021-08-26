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

from typing import AsyncGenerator, List, Optional

from ..core import NodeRole
from .base import AbstractClusterBackend, register_cluster_backend


@register_cluster_backend
class FixedClusterBackend(AbstractClusterBackend):
    name = 'fixed'

    def __init__(self, lookup_address: str):
        self._supervisors = [n.strip() for n in lookup_address.split(',')]

    @classmethod
    async def create(cls, node_role: NodeRole, lookup_address: Optional[str],
                     pool_address: str):
        return cls(lookup_address)

    async def watch_supervisors(self) -> AsyncGenerator[List[str], None]:
        yield self._supervisors

    async def get_supervisors(self, filter_ready: bool = True) -> List[str]:
        return self._supervisors
