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

from typing import Dict

from .... import oscar as mo
from ...session import SessionAPI
from ..store import get_meta_store


class MetaStoreManagerActor(mo.Actor):
    def __init__(self,
                 meta_store_name: str,
                 config: Dict):
        self._meta_store_name = meta_store_name
        self._meta_store_type = get_meta_store(meta_store_name)
        self._config = config
        self._meta_init_kwargs = None

        # API
        self._session_api = None

    async def __post_create__(self):
        self._meta_init_kwargs = \
            await self._meta_store_type.create(self._config)
        self._session_api = await SessionAPI.create(self.address)

    async def new_session_meta_store(self,
                                     session_id: str) -> mo.ActorRef:
        session_address = await self._session_api.get_session_address(session_id)
        allocate_strategy = mo.allocate_strategy.AddressSpecified(session_address)
        return await mo.create_actor(MetaStoreActor,
                                     self._meta_store_name,
                                     session_id,
                                     address=self.address,
                                     uid=MetaStoreActor.gen_uid(session_id),
                                     allocate_strategy=allocate_strategy,
                                     **self._meta_init_kwargs)


class MetaStoreActor(mo.Actor):
    def __init__(self,
                 meta_store_name: str,
                 session_id: str,
                 **meta_store_kwargs):
        meta_store_type = get_meta_store(meta_store_name)
        self._store = meta_store_type(session_id, **meta_store_kwargs)

    @staticmethod
    def gen_uid(session_id):
        return f'{session_id}_meta'

    def __getattr__(self, attr):
        return getattr(self._store, attr)
