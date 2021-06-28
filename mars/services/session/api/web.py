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
from typing import List, Union

from ....lib.aio import alru_cache
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import SessionInfo
from .core import AbstractSessionAPI


class SessionWebAPIHandler(MarsServiceWebAPIHandler):
    @classmethod
    def get_root_pattern(cls):
        return '/api/session(?:/(?P<sub_path>[^/]*)$|$)'

    @alru_cache(cache_exceptions=False)
    async def _get_cluster_api(self):
        from ...cluster import ClusterAPI
        return await ClusterAPI.create(self._supervisor_addr)

    @alru_cache(cache_exceptions=False)
    async def _get_oscar_session_api(self):
        from .oscar import SessionAPI
        cluster_api = await self._get_cluster_api()
        [address] = await cluster_api.get_supervisors_by_keys(['Session'])
        return await SessionAPI.create(address)

    @web_api('(?P<session_id>[^/]+)', method='put')
    async def create_session(self, session_id: str):
        oscar_api = await self._get_oscar_session_api()
        addr = await oscar_api.create_session(session_id)
        self.write(addr)

    @web_api('(?P<session_id>[^/]+)', method='delete')
    async def delete_session(self, session_id: str):
        oscar_api = await self._get_oscar_session_api()
        await oscar_api.delete_session(session_id)

    @web_api('(?P<session_id>[^/]+)', method='get',
             arg_filter={'action': 'check_exist'})
    async def has_session(self, session_id: str):
        oscar_api = await self._get_oscar_session_api()
        res = await oscar_api.has_session(session_id)
        self.write('1' if res else '0')

    @web_api('(?P<session_id>[^/]*)', method='get',
             arg_filter={'action': 'get_last_idle_time'})
    async def get_last_idle_time(self, session_id: str):
        session_id = session_id or None
        oscar_api = await self._get_oscar_session_api()
        res = await oscar_api.get_last_idle_time(session_id)
        self.write(str(res) if res else '')

    @web_api('', method='get')
    async def get_sessions(self):
        oscar_api = await self._get_oscar_session_api()
        res = await oscar_api.get_sessions()
        self.write(json.dumps({'sessions': [
            {'session_id': info.session_id} for info in res
        ]}))


web_handlers = {
    SessionWebAPIHandler.get_root_pattern(): SessionWebAPIHandler
}


class WebSessionAPI(AbstractSessionAPI, MarsWebAPIClientMixin):
    def __init__(self, address: str):
        self._address = address.rstrip('/')

    async def get_sessions(self) -> List[SessionInfo]:
        addr = f'{self._address}/api/session'
        res = await self._request_url('GET', addr)
        res_obj = json.loads((await res.read()).decode())
        return [SessionInfo(**kw) for kw in res_obj['sessions']]

    async def create_session(self, session_id: str) -> str:
        addr = f'{self._address}/api/session/{session_id}'
        res = await self._request_url(path=addr, method='PUT', data=b'')
        return (await res.read()).decode()

    async def delete_session(self, session_id: str):
        addr = f'{self._address}/api/session/{session_id}'
        await self._request_url(path=addr, method='DELETE')

    async def has_session(self, session_id: str):
        addr = f'{self._address}/api/session/{session_id}'
        params = dict(action='check_exist')
        res = await self._request_url('GET', addr, params=params)
        return bool(int(await res.read()))

    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        session_id = session_id or ''
        addr = f'{self._address}/api/session/{session_id}'
        params = dict(action='get_last_idle_time')
        res = await self._request_url('GET', addr, params=params)
        content = await res.read()
        return float(content) if content else None
