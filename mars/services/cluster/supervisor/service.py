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
from ...core import NodeRole
from ..locator import SupervisorLocatorActor
from ..uploader import NodeInfoUploaderActor
from .node_info import NodeInfoCollectorActor


async def start(config: dict, address: str):
    """
    Start cluster service on supervisor

    Parameters
    ----------
    config
        service config.
        {
            "cluster": {
                "backend": "<cluster backend name>",
                "lookup_address": "<address of master>",
                "node_timeout": timeout seconds of nodes,
                "node_check_interval": check interval seconds for nodes
            }
        }
    address
        address of actor pool
    """
    svc_config = config['cluster']
    backend = svc_config.get('backend', 'fixed')
    lookup_address = svc_config.get('lookup_address',
                                    address if backend == 'fixed' else None)
    await mo.create_actor(
        SupervisorLocatorActor,
        backend_name=backend,
        lookup_address=lookup_address,
        uid=SupervisorLocatorActor.default_uid(),
        address=address)
    await mo.create_actor(
        NodeInfoCollectorActor,
        timeout=svc_config.get('node_timeout'),
        check_interval=svc_config.get('node_check_interval'),
        uid=NodeInfoCollectorActor.default_uid(),
        address=address)
    await mo.create_actor(
        NodeInfoUploaderActor,
        role=NodeRole.SUPERVISOR,
        interval=svc_config.get('node_check_interval'),
        uid=NodeInfoUploaderActor.default_uid(),
        address=address)
