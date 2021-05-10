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

from typing import Union

from mars.services.web.core import ServiceProxyHandlerBase, get_service_proxy_endpoint
from mars.services.web.core import ServiceWebAPIBase, get_supervisor_address
from .api import SessionAPI, OscarSessionAPI


class SessionAPIProxyHandler(ServiceProxyHandlerBase):
    _api_cls = OscarSessionAPI


web_handlers = {
    get_service_proxy_endpoint('session'): SessionAPIProxyHandler
}


class WebSessionAPI(ServiceWebAPIBase, SessionAPI):
    _service_name = 'session'

    @classmethod
    async def create(cls, address: str, **kwargs):
        supervisor_address = await get_supervisor_address(address)
        return WebSessionAPI(address, 'create', supervisor_address, **kwargs)

    async def create_session(self, session_id: str) -> str:
        return await self._call_method({}, 'create_session', session_id)

    async def delete_session(self, session_id: str):
        return await self._call_method({}, 'delete_session', session_id)

    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        return await self._call_method({}, 'get_last_idle_time', session_id)
