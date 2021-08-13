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

from .... import oscar as mo
from ...core import AbstractService
from .core import MetaStoreManagerActor, MetaStoreActor


class MetaSupervisorService(AbstractService):
    """
    Meta service on supervisor.

    Service Configuration
    ---------------------
    {
        "meta" : {
            "store": "<meta store name>",
            # other config related to each store
        }
    }
    """
    async def start(self):
        service_config = self._config['meta']
        meta_store_name = service_config.get('meta', 'dict')
        extra_config = service_config.copy()
        extra_config.pop('meta', None)
        await mo.create_actor(MetaStoreManagerActor,
                              meta_store_name,
                              extra_config,
                              uid=MetaStoreManagerActor.default_uid(),
                              address=self._address)

    async def stop(self):
        await mo.destroy_actor(mo.create_actor_ref(
            uid=MetaStoreManagerActor.default_uid(), address=self._address))

    async def create_session(self, session_id: str):
        # get MetaStoreManagerActor ref.
        meta_store_manager_ref = await mo.actor_ref(
            self._address, MetaStoreManagerActor.default_uid())
        await meta_store_manager_ref.new_session_meta_store(session_id)

    async def destroy_session(self, session_id: str):
        meta_store_ref = await mo.actor_ref(
            self._address, MetaStoreActor.gen_uid(session_id))
        await mo.destroy_actor(meta_store_ref)
