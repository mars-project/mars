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

from typing import List

from mars.services.web.core import ServiceProxyHandlerBase, get_service_proxy_endpoint
from mars.services.web.core import ServiceWebAPIBase, _transfer_request_timeout
from ...core import Tileable
from ...lib.aio import alru_cache
from .api import TaskAPI, OscarTaskAPI
from .core import TileableGraph, TaskResult


class TaskAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = OscarTaskAPI


web_handlers = {
    get_service_proxy_endpoint('task'): TaskAPIProxyHandler,
}


class WebTaskAPI(ServiceWebAPIBase, TaskAPI):
    _service_name = 'task'

    @classmethod
    @alru_cache
    async def create(cls, web_address: str, session_id: str, address: str, **kwargs):
        return WebTaskAPI(web_address, 'create', session_id, address, **kwargs)

    async def submit_tileable_graph(self,
                                    graph: TileableGraph,
                                    task_name: str = None,
                                    fuse_enabled: bool = True,
                                    extra_config: dict = None) -> str:
        return await self._call_method(dict(request_timeout=_transfer_request_timeout),
                                       'submit_tileable_graph', graph, task_name, fuse_enabled, extra_config)

    async def get_fetch_tileables(self, task_id: str) -> List[Tileable]:
        return await self._call_method({}, 'get_fetch_tileables', task_id)

    async def wait_task(self, task_id: str, timeout: float = None):
        return await self._call_method({}, 'wait_task', task_id, timeout)

    async def get_task_progress(self, task_id: str) -> float:
        return await self._call_method({}, 'get_task_progress', task_id)

    async def get_task_result(self, task_id: str) -> TaskResult:
        return await self._call_method({}, 'get_task_result', task_id)
