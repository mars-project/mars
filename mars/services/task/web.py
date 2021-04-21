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

from mars.services.web.core import serialize, deserialize, get_web_address, ServiceWebHandlerBase, ServiceWebAPIBase
from tornado.httpclient import AsyncHTTPClient
from .api import TaskAPI


class TaskWebHandler(ServiceWebHandlerBase):

    async def create(self, session_id: str, address: str):
        api_instance = await TaskAPI.create(session_id, address)
        self._api_instances[id(api_instance)] = api_instance
        return id(api_instance)

    async def create_session(self, session_id: str, address: str):
        api_instance = await TaskAPI.create(session_id, address)
        self._api_instances[id(api_instance)] = api_instance
        return id(api_instance)

    async def destroy_session(self, session_id: str, address: str):
        return await TaskAPI.destroy_session(session_id, address)


_service_name = 'task'
web_handlers = {
    f'/api/service/{_service_name}/(.*)': TaskWebHandler,
}


class TaskWebAPI(ServiceWebAPIBase):

    @classmethod
    async def create(cls, session_id: str, address: str):
        http_client = AsyncHTTPClient()
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{_service_name}/create',
                                       method="POST", body=serialize((session_id, address)))
        api_id = deserialize(resp.body)
        return TaskWebAPI(http_client, _service_name, api_id)

    @classmethod
    async def create_session(cls, session_id: str, address: str):
        http_client = AsyncHTTPClient()
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{_service_name}/create_session',
                                       method="POST", body=serialize((session_id, address)))
        api_id = deserialize(resp.body)
        return TaskWebAPI(http_client, _service_name, api_id)

    @classmethod
    async def destroy_session(cls, session_id: str, address: str):
        http_client = AsyncHTTPClient()
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{_service_name}/destroy_session',
                                       method="POST", body=serialize((session_id, address)))
        api_id = deserialize(resp.body)
        return TaskWebAPI(http_client, _service_name, api_id)
