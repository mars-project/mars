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

from typing import Dict, List

from ... import NodeRole
from ....lib.aio import alru_cache
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractClusterAPI


class ClusterWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = '/api/cluster'

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI
        return await ClusterAPI.create(self._supervisor_addr)

    @web_api('nodes', method=['get', 'post'])
    async def get_nodes_info(self):
        watch = bool(int(self.get_argument('watch', '0')))
        env = bool(int(self.get_argument('env', '0')))
        resource = bool(int(self.get_argument('resource', '0')))
        state = bool(int(self.get_argument('state', '0')))

        nodes_arg = self.get_argument('nodes', None)
        nodes = nodes_arg.split(',') if nodes_arg is not None else None

        role_arg = self.get_argument('role', None)
        role = NodeRole(int(role_arg)) if role_arg is not None else None

        cluster_api = await self._get_cluster_api()
        if watch:
            assert nodes is None
            result = await cluster_api.watch_nodes(
                role, env=env, resource=resource, state=state
            )
        else:
            result = await cluster_api.get_nodes_info(
                nodes=nodes, role=role, env=env, resource=resource, state=state
            )
        self.write(serialize_serializable(result))


web_handlers = {
    ClusterWebAPIHandler.get_root_pattern(): ClusterWebAPIHandler
}


class WebClusterAPI(AbstractClusterAPI, MarsWebAPIClientMixin):
    def __init__(self, address: str):
        self._address = address.rstrip('/')

    async def _get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                              env: bool = False, resource: bool = False, state: bool = False,
                              watch: bool = False):
        args = [
            ('nodes', ','.join(nodes) if nodes else None),
            ('role', role.value if role is not None else None),
            ('env', 1 if env else 0),
            ('resource', 1 if resource else 0),
            ('state', 1 if state else 0),
            ('watch', 1 if watch else 0),
        ]
        args_str = '&'.join(f'{key}={val}' for key, val in args if val is not None)

        path = f'{self._address}/api/cluster/nodes'
        res = await self._request_url(
            path, method='POST', body=args_str,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )
        return deserialize_serializable(res.body)

    async def get_supervisors(self, watch=False) -> List[str]:
        res = await self._get_nodes_info(role=NodeRole.SUPERVISOR, watch=watch)
        return list(res.keys())

    async def get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                             env: bool = False, resource: bool = False, state: bool = False):
        return await self._get_nodes_info(nodes, role=role, env=env, resource=resource,
                                          state=state, watch=False)

    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, state: bool = False) -> List[Dict[str, Dict]]:
        return await self._get_nodes_info(role=role, env=env, resource=resource,
                                          state=state, watch=True)
