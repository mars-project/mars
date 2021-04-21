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
from .api import StorageAPI


class StorageWebHandler(ServiceWebHandlerBase):

    async def create(self, session_id: str, address: str, **kwargs):
        api_instance = await StorageAPI.create(session_id, address, **kwargs)
        self._api_instances[id(api_instance)] = api_instance
        return id(api_instance)


_service_name = 'service'
web_handlers = {
    f'/api/service/{_service_name}/(.*)': StorageWebHandler,
}


class StorageWebAPI(ServiceWebAPIBase):

    @classmethod
    async def create(cls, session_id: str, address: str, **kwargs):
        http_client = AsyncHTTPClient()
        resp = await http_client.fetch(f'{get_web_address()}/api/storage/{_service_name}/create',
                                       method="POST", body=serialize((session_id, address, kwargs)))
        api_id = deserialize(resp.body)
        return StorageWebAPI(http_client, _service_name, StorageAPI, api_id)
