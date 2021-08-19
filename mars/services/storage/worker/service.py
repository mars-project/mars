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
from ..core import StorageManagerActor


class StorageWorkerService(AbstractService):
    """
    Storage service on worker

    Service Configuration
    ---------------------
    {
        "storage": {
            "backends": ["plasma"],
            "<storage backend name>"： "<setup params>",
        }
    }
    """
    async def start(self):
        storage_configs = self._config['storage']
        backends = storage_configs.get('backends')
        options = storage_configs.get('default_config', dict())
        transfer_block_size = options.get('transfer_block_size', None)
        backend_config = {backend: storage_configs.get(backend, dict())
                          for backend in backends}

        await mo.create_actor(StorageManagerActor,
                              backend_config,
                              transfer_block_size,
                              uid=StorageManagerActor.default_uid(),
                              address=self._address)

    async def stop(self):
        storage_manager_ref = await mo.actor_ref(
            address=self._address, uid=StorageManagerActor.default_uid())
        await mo.destroy_actor(storage_manager_ref)
