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

from typing import List

from ... import oscar as mo
from .core import ChunkLifecycleActor


class LifecycleAPI:
    def __init__(self,
                 session_id: str,
                 chunk_lifecycle_ref: mo.ActorRef,
                 cluster_api):
        self._session_id = session_id
        self._chunk_lifecycle_ref = chunk_lifecycle_ref
        self._cluster_api = cluster_api

    @classmethod
    async def create(cls, session_id: str, address: str) -> 'LifecycleAPI':
        from ..cluster.api import ClusterAPI

        cluster_api = await ClusterAPI.create(address)
        supervisor_addr = await cluster_api.get_supervisor(session_id)
        chunk_lifecycle_ref = await mo.actor_ref(
            ChunkLifecycleActor.gen_uid(session_id),
            address=supervisor_addr,
        )

        return LifecycleAPI(session_id, chunk_lifecycle_ref, cluster_api)

    async def incref_chunks(self, chunk_keys: List[str]):
        """
        Increase references of chunks

        Parameters
        ----------
        chunk_keys
            keys of chunks
        """
        await self._chunk_lifecycle_ref.incref(chunk_keys)

    async def decref_chunks(self, chunk_keys: List[str]):
        """
        Decrease references of chunks

        Parameters
        ----------
        chunk_keys
            keys of chunks
        """
        await self._chunk_lifecycle_ref.decref(chunk_keys)

    async def delete_zero_ref_chunks(self, chunk_keys: List[str]):
        """
        Delete chunks whose refs are zero

        Parameters
        ----------
        chunk_keys
            keys of chunks
        """
        await self._chunk_lifecycle_ref.delete_zero_ref(chunk_keys)
