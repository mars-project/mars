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

from typing import List, Union

from ..core import TaskResult
from ....core import TileableGraph, Tileable
from ....lib.aio import alru_cache
from ....utils import serialize_serializable, deserialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractTaskAPI


class TaskWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = '/api/session/(?P<session_id>[^/]+)/task'

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI
        return await ClusterAPI.create(self._supervisor_addr)

    @alru_cache(cache_exceptions=False)
    async def _get_oscar_task_api(self, session_id: str):
        from .oscar import TaskAPI
        cluster_api = await self._get_cluster_api()
        [address] = await cluster_api.get_supervisors_by_keys([session_id])
        return await TaskAPI.create(session_id, address)

    @web_api('', method='post')
    async def submit_tileable_graph(self, session_id: str):
        body_args = deserialize_serializable(self.request.body) \
            if self.request.body else None

        task_name = body_args.get('task_name', None) or None
        fuse_enabled = body_args.get('fuse')
        graph = body_args['graph']
        extra_config = body_args.get('extra_config', None)

        oscar_api = await self._get_oscar_task_api(session_id)
        task_id = await oscar_api.submit_tileable_graph(
            graph, task_name=task_name, fuse_enabled=fuse_enabled,
            extra_config=extra_config)
        self.write(task_id)

    @web_api('(?P<task_id>[^/]+)', method='get', arg_filter={'action': 'fetch_tileables'})
    async def get_fetch_tileables(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_fetch_tileables(task_id)
        self.write(serialize_serializable(res))

    @web_api('(?P<task_id>[^/]+)', method='get')
    async def get_task_result(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_task_result(task_id)
        self.write(serialize_serializable(res))

    @web_api('(?P<task_id>[^/]+)', method='get', arg_filter={'action': 'progress'})
    async def get_task_progress(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_task_progress(task_id)
        self.write(str(res))

    @web_api('', method='get', arg_filter={'action': 'last_idle_time'})
    async def get_last_idle_time(self, session_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.get_last_idle_time()
        if res:
            self.write(str(res))

    @web_api('(?P<task_id>[^/]+)', method='get', arg_filter={'action': 'wait'})
    async def wait_task(self, session_id: str, task_id: str):
        timeout = self.get_argument('timeout', None) or None
        timeout = float(timeout) if timeout is not None else None
        oscar_api = await self._get_oscar_task_api(session_id)
        res = await oscar_api.wait_task(task_id, timeout)
        self.write(serialize_serializable(res))

    @web_api('(?P<task_id>[^/]+)', method='delete')
    async def cancel_task(self, session_id: str, task_id: str):
        oscar_api = await self._get_oscar_task_api(session_id)
        await oscar_api.cancel_task(task_id)


web_handlers = {
    TaskWebAPIHandler.get_root_pattern(): TaskWebAPIHandler
}


class WebTaskAPI(AbstractTaskAPI, MarsWebAPIClientMixin):
    def __init__(self, session_id: str, address: str):
        self._session_id = session_id
        self._address = address

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        path = f'{self._address}/api/session/{self._session_id}/task'
        extra_config_ser = serialize_serializable(extra_config) \
            if extra_config else None
        body = serialize_serializable({
            'task_name': task_name if task_name else '',
            'fuse': fuse_enabled,
            'graph': graph,
            'extra_config': extra_config_ser,
        })
        res = await self._request_url(
            path, method='POST',
            headers={'Content-Type': 'application/octet-stream'},
            body=body
        )
        return res.body.decode()

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        path = f'{self._address}/api/session/{self._session_id}/task/{task_id}' \
               f'?action=fetch_tileables'
        res = await self._request_url(path)
        return deserialize_serializable(res.body)

    async def get_task_result(self, task_id: str) -> TaskResult:
        path = f'{self._address}/api/session/{self._session_id}/task/{task_id}'
        res = await self._request_url(path)
        return deserialize_serializable(res.body)

    async def get_task_progress(self,
                                task_id: str) -> float:
        path = f'{self._address}/api/session/{self._session_id}/task/{task_id}' \
               f'?action=progress'
        res = await self._request_url(path)
        return float(res.body)

    async def get_last_idle_time(self) -> Union[float, None]:
        path = f'{self._address}/api/session/{self._session_id}/task' \
               f'?action=last_idle_time'
        res = await self._request_url(path)
        return float(res.body) if res.body else None

    async def wait_task(self, task_id: str, timeout: float = None):
        path = f'{self._address}/api/session/{self._session_id}/task/{task_id}' \
               f'?action=wait&timeout={timeout or ""}'
        res = await self._request_url(path)
        return deserialize_serializable(res.body)

    async def cancel_task(self, task_id: str):
        path = f'{self._address}/api/session/{self._session_id}/task/{task_id}'
        await self._request_url(path, method='DELETE')
