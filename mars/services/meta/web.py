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
from mars.services.web.core import ServiceWebAPIBase
from ...utils import extensible
from .api import MetaAPI, OscarMetaAPI


class MetaAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = OscarMetaAPI


web_handlers = {
    get_service_proxy_endpoint('meta'): MetaAPIProxyHandler,
}


class WebMetaAPI(ServiceWebAPIBase, MetaAPI):
    _service_name = 'meta'

    @classmethod
    async def create(cls, web_address: str, session_id: str, address: str):
        return WebMetaAPI(web_address, 'create', session_id, address)

    @extensible
    async def get_chunk_meta(self,
                             object_id: str,
                             fields: List[str] = None):
        return await self._call_method({}, 'get_chunk_meta', object_id, fields)
