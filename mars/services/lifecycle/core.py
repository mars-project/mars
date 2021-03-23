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

from ... import oscar as mo


class ChunkLifecycleActor(mo.Actor):
    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_chunk_lifecycle'

    def __init__(self, session_id):
        super().__init__()
        self._session_id = session_id
        self._ref_counts = dict()
        self._storage_client = None

    async def __post_create__(self):
        from ..storage.api import StorageAPI
        super().post_create()
        self._storage_client = await StorageAPI.create(self._session_id, self.address)

    def incref(self, keys):
        for k in keys:
            try:
                self._ref_counts[k] += 1
            except KeyError:
                self._ref_counts[k] = 1

    def decref(self, keys):
        del_req = []
        for k in keys:
            self._ref_counts[k] -= 1
            if self._ref_counts[k] == 0:
                del self._ref_counts[k]
                del_req.append((k,))
        asyncio.create_task(self._storage_client.delete.batch(del_req))

    def delete_zero_ref(self, keys):
        del_req = []
        for k in keys:
            if k not in self._ref_counts:
                del_req.append((k,))
        asyncio.create_task(self._storage_client.delete.batch(del_req))
