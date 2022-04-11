# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from typing import Dict

from .... import oscar as mo
from ....lib.aio import alru_cache
from ...cluster import ClusterAPI
from ..store import get_meta_store


class WorkerMetaStoreManagerActor(mo.Actor):
    def __init__(self, meta_store_name: str, config: Dict):
        self._meta_store_name = meta_store_name
        self._meta_store_type = get_meta_store(meta_store_name)
        self._config = config
        self._meta_init_kwargs = None

        self._cluster_api = None

    async def __post_create__(self):
        self._meta_init_kwargs = await self._meta_store_type.create(self._config)
        self._cluster_api = await ClusterAPI.create(self.address)

    @alru_cache(cache_exceptions=False)
    async def _get_supervisor_address(self, session_id: str):
        [address] = await self._cluster_api.get_supervisors_by_keys([session_id])
        return address

    async def new_session_meta_store(self, session_id: str) -> mo.ActorRef:
        from ..supervisor.core import MetaStoreActor

        try:
            ref = await mo.create_actor(
                WorkerMetaStoreActor,
                self._meta_store_name,
                session_id,
                uid=WorkerMetaStoreActor.gen_uid(session_id),
                address=self.address,
                **self._meta_init_kwargs,
            )
            supervisor_address = await self._get_supervisor_address(session_id)
            supervisor_meta_store_ref = await mo.actor_ref(
                uid=MetaStoreActor.gen_uid(session_id), address=supervisor_address
            )
            # register worker meta store,
            # when session destroyed, this worker meta store actor will be removed
            await supervisor_meta_store_ref.add_worker_meta_store(ref)
        except mo.ActorAlreadyExist:
            ref = await mo.actor_ref(
                uid=WorkerMetaStoreActor.gen_uid(session_id), address=self.address
            )
        return ref


class WorkerMetaStoreActor(mo.Actor):
    def __init__(self, meta_store_name: str, session_id: str, **meta_store_kwargs):
        meta_store_type = get_meta_store(meta_store_name)
        self._store = meta_store_type(session_id, **meta_store_kwargs)

    @staticmethod
    def gen_uid(session_id: str):
        return f"{session_id}_worker_meta"

    def __getattr__(self, attr):
        return getattr(self._store, attr)
