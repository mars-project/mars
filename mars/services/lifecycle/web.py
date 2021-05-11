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

from ..web.core import ServiceProxyHandlerBase, ServiceWebAPIBase, \
    get_service_proxy_endpoint
from .api import AbstractLifecycleAPI, LifecycleAPI


class LifecycleAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = LifecycleAPI


lifecycle_handlers = {
    get_service_proxy_endpoint('lifecycle'): LifecycleAPIProxyHandler,
}


class WebLifecycleAPI(ServiceWebAPIBase, AbstractLifecycleAPI):
    _service_name = 'lifecycle'

    @classmethod
    async def create(cls, web_address: str, session_id: str, address: str):
        return WebLifecycleAPI(web_address, 'create', session_id, address)

    async def decref_tileables(self, tileable_keys: List[str]):
        return await self._call_method({}, 'decref_tileables',
                                       tileable_keys)
