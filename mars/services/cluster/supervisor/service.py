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
from ...core import NodeRole, AbstractService
from ..uploader import NodeInfoUploaderActor
from .locator import SupervisorPeerLocatorActor
from .node_info import NodeInfoCollectorActor


class ClusterSupervisorService(AbstractService):
    """
    Cluster service on supervisor

    Service Configuration
    ---------------------
    {
        "cluster": {
            "backend": "<cluster backend name>",
            "lookup_address": "<address of master>",
            "node_timeout": timeout seconds of nodes,
            "node_check_interval": check interval seconds for nodes
        }
    }
    """
    async def start(self):
        svc_config = self._config['cluster']
        address = self._address

        backend = svc_config.get('backend', 'fixed')
        lookup_address = svc_config.get('lookup_address',
                                        address if backend == 'fixed' else None)
        await mo.create_actor(
            NodeInfoCollectorActor,
            timeout=svc_config.get('node_timeout'),
            check_interval=svc_config.get('node_check_interval'),
            uid=NodeInfoCollectorActor.default_uid(),
            address=address)
        await mo.create_actor(
            SupervisorPeerLocatorActor,
            backend_name=backend,
            lookup_address=lookup_address,
            uid=SupervisorPeerLocatorActor.default_uid(),
            address=address)
        await mo.create_actor(
            NodeInfoUploaderActor,
            role=NodeRole.SUPERVISOR,
            interval=svc_config.get('node_check_interval'),
            uid=NodeInfoUploaderActor.default_uid(),
            address=address)

    async def stop(self):
        address = self._address

        await mo.destroy_actor(mo.create_actor_ref(
            uid=NodeInfoCollectorActor.default_uid(),
            address=address))
        await mo.destroy_actor(mo.create_actor_ref(
            uid=SupervisorPeerLocatorActor.default_uid(),
            address=address))
        await mo.destroy_actor(mo.create_actor_ref(
            uid=NodeInfoUploaderActor.default_uid(),
            address=address))
