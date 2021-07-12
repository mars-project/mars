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

import json
from typing import Dict, List, Optional

from ....lib.aio import alru_cache
from ....utils import serialize_serializable, deserialize_serializable
from ...core import NodeRole, BandType
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import watch_method
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
        detail = bool(int(self.get_argument('detail', '0')))

        nodes_arg = self.get_argument('nodes', None)
        nodes = nodes_arg.split(',') if nodes_arg is not None else None

        role_arg = self.get_argument('role', None)
        role = NodeRole(int(role_arg)) if role_arg is not None else None

        cluster_api = await self._get_cluster_api()
        result = {}
        if watch:
            assert nodes is None
            version = self.get_argument('version', '') or None
            if version:
                version = int(version)

            async for version, node_infos in cluster_api.watch_nodes(
                role, env=env, resource=resource, detail=detail, version=version
            ):
                result['version'] = version
                result['nodes'] = node_infos
                break
        else:
            result['nodes'] = await cluster_api.get_nodes_info(
                nodes=nodes, role=role, env=env, resource=resource, detail=detail
            )
        self.write(json.dumps(result))

    @web_api('bands', method='get')
    async def get_all_bands(self):
        role_arg = self.get_argument('role', None)
        role = NodeRole(int(role_arg)) if role_arg is not None else None
        watch = bool(int(self.get_argument('watch', '0')))

        cluster_api = await self._get_cluster_api()
        if watch:
            version = self.get_argument('version', '') or None
            if version:
                version = int(version)

            async for version, bands in cluster_api.watch_all_bands(role, version=version):
                self.write(serialize_serializable((version, bands)))
                break
        else:
            self.write(serialize_serializable(
                await cluster_api.get_all_bands(role)
            ))

    @web_api('versions', method='get')
    async def get_mars_versions(self):
        cluster_api = await self._get_cluster_api()
        self.write(json.dumps(list(await cluster_api.get_mars_versions())))


web_handlers = {
    ClusterWebAPIHandler.get_root_pattern(): ClusterWebAPIHandler
}


class WebClusterAPI(AbstractClusterAPI, MarsWebAPIClientMixin):
    def __init__(self, address: str):
        self._address = address.rstrip('/')

    async def _get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                              env: bool = False, resource: bool = False, detail: bool = False,
                              watch: bool = False, version: Optional[int] = None):
        args = [
            ('nodes', ','.join(nodes) if nodes else None),
            ('role', role.value if role is not None else None),
            ('env', 1 if env else 0),
            ('resource', 1 if resource else 0),
            ('detail', 1 if detail else 0),
            ('watch', 1 if watch else 0),
            ('version', str(version or '')),
        ]
        args_str = '&'.join(f'{key}={val}' for key, val in args if val is not None)

        path = f'{self._address}/api/cluster/nodes'
        res = await self._request_url(
            path=path, method='POST', data=args_str,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )
        result = json.loads(await res.read())
        if watch:
            return result['version'], result['nodes']
        else:
            return result['nodes']

    async def get_supervisors(self) -> List[str]:
        res = await self._get_nodes_info(role=NodeRole.SUPERVISOR)
        return list(res.keys())

    @watch_method
    async def watch_supervisors(self, version: Optional[int] = None):
        version, res = await self._get_nodes_info(role=NodeRole.SUPERVISOR,
                                                  watch=True, version=version)
        return version, list(res.keys())

    async def get_nodes_info(self, nodes: List[str] = None, role: NodeRole = None,
                             env: bool = False, resource: bool = False, detail: bool = False):
        return await self._get_nodes_info(nodes, role=role, env=env, resource=resource,
                                          detail=detail, watch=False)

    @watch_method
    async def watch_nodes(self, role: NodeRole, env: bool = False,
                          resource: bool = False, detail: bool = False,
                          version: Optional[int] = None) -> List[Dict[str, Dict]]:
        return await self._get_nodes_info(role=role, env=env, resource=resource,
                                          detail=detail, watch=True, version=version)

    async def get_all_bands(self, role: NodeRole = None) -> Dict[BandType, int]:
        path = f'{self._address}/api/cluster/bands'
        params = {}
        if role is not None:  # pragma: no cover
            params['role'] = role.value
        res = await self._request_url('GET', path, params=params)
        return deserialize_serializable(await res.read())

    @watch_method
    async def watch_all_bands(self, role: NodeRole = None,
                              version: Optional[int] = None):
        params = dict(watch=1, version=str(version or ''))
        path = f'{self._address}/api/cluster/bands'
        if role is not None:  # pragma: no cover
            params['role'] = role.value
        res = await self._request_url('GET', path, params=params)
        return deserialize_serializable(await res.read())

    async def get_mars_versions(self) -> List[str]:
        path = f'{self._address}/api/cluster/versions'
        res = await self._request_url('GET', path)
        return list(json.loads(await res.read()))
