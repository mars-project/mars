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

from typing import Any, List

from mars.services.web.core import ServiceProxyHandlerBase, get_service_proxy_endpoint
from mars.services.web.core import ServiceWebAPIBase, _transfer_request_timeout
from ...storage.base import StorageLevel
from .api import StorageAPI, OscarStorageAPI
from .core import DataInfo


class StorageAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = OscarStorageAPI


web_handlers = {
    get_service_proxy_endpoint('storage'): StorageAPIProxyHandler,
}


class WebStorageAPI(ServiceWebAPIBase, StorageAPI):
    _service_name = 'storage'

    @classmethod
    async def create(cls, web_address: str, session_id: str, address: str):
        return WebStorageAPI(web_address, 'create', session_id, address)

    async def get(self, data_key: str, conditions: List = None) -> Any:
        return await self._call_method(dict(request_timeout=_transfer_request_timeout),
                                       'get', data_key, conditions)

    async def put(self, data_key: str, obj: object,
                  level: StorageLevel = StorageLevel.MEMORY) -> DataInfo:
        return await self._call_method(dict(request_timeout=_transfer_request_timeout),
                                       'put', data_key, obj, level)
