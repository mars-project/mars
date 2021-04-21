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

from .... import oscar as mo
from ..core import StorageManagerActor


async def start(config: dict, address: str):
    """
    Start storage service on worker
    Parameters
    ----------
    config
        storage service config.
        {
            "storage":
                {
                    "backends": ["plasma"],
                    "<storage backend name>"： "<setup params>",
                }
        }
    address
        address of actor pool
    """
    storage_configs = config['storage']
    backends = storage_configs.get('backends')
    backend_config = {backend: storage_configs.get(backend)
                      for backend in backends}

    await mo.create_actor(StorageManagerActor,
                          backend_config,
                          uid=StorageManagerActor.default_uid(),
                          address=address)


async def stop(address: str):
    """
    Stop storage service on worker
    Parameters
    ----------
    address:
        main pool address of worker
    """
    storage_manager_ref = await mo.actor_ref(
        address=address, uid=StorageManagerActor.default_uid())
    await mo.destroy_actor(storage_manager_ref)
