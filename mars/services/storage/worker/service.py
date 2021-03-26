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


async def start(config: dict, address: None):
    """
    Start cluster service on supervisor
    Parameters
    ----------
    config
        storage service config.
        {
            "storage_configs":
                {
                    "<storage backend name>"： "<setup params>",
                }
        }
    address
        address of actor pool
    """
    storage_configs = config['storage_configs']

    await mo.create_actor(StorageManagerActor,
                          storage_configs,
                          uid=StorageManagerActor.default_uid(),
                          address=address)
