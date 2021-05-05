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

from mars.services.web.core import ServiceProxyHandlerBase, ServiceWebAPIBase, get_service_proxy_endpoint
from .api import TaskAPI


class TaskAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = TaskAPI

    async def create(self, session_id: str, address: str):
        return await TaskAPI.create(session_id, address)


_service_name = 'task'
web_handlers = {
    get_service_proxy_endpoint(_service_name): TaskAPIProxyHandler,
}


class TaskWebAPI(ServiceWebAPIBase):
    _service_name = _service_name

    @classmethod
    async def create(cls, web_address: str, session_id: str, address: str):
        return TaskWebAPI(web_address, 'create', (session_id, address), {})
