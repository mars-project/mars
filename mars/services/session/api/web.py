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
from typing import Dict, List, Union

from ....lib.aio import alru_cache
from ....utils import parse_readable_size
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from ..core import SessionInfo
from .core import AbstractSessionAPI


def _encode_size(size: Union[str, Dict[str, List[int]]]) -> str:
    if not isinstance(size, dict):
        return size
    else:
        return ','.join(f'{k}={v}' for k, v in size.items())


def _decode_size(encoded: str) -> Union[int, str, Dict[str, Union[int, List[int]]]]:
    if not encoded:
        return 0
    if ',' not in encoded and '=' not in encoded:
        try:
            return int(encoded)
        except ValueError:
            return int(parse_readable_size(encoded)[0])
    else:
        ret = dict()
        for kv in encoded.split(','):
            k, v = kv.split('=', 1)
            ret[k] = int(parse_readable_size(v)[0])
        return ret


class SessionWebAPIBaseHandler(MarsServiceWebAPIHandler):
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


class SessionWebAPIHandler(SessionWebAPIBaseHandler):
    @classmethod
    def get_root_pattern(cls):
        return '/api/session(?:/(?P<sub_path>[^/]*)$|$)'

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


class SessionWebLogAPIHandler(SessionWebAPIBaseHandler):
    _root_pattern = '/api/session/(?P<session_id>[^/]+)/op/(?P<op_key>[^/]+)/log'

    @web_api('', method='get')
    async def fetch_tileable_op_logs(self,
                                     session_id: str,
                                     op_key: str):
        oscar_api = await self._get_oscar_session_api()
        offsets = _decode_size(self.get_argument('offsets', None))
        sizes = _decode_size(self.get_argument('sizes', None))
        log_result = await oscar_api.fetch_tileable_op_logs(
            session_id, op_key, offsets, sizes)
        self.write(json.dumps(log_result))


web_handlers = {
    SessionWebAPIHandler.get_root_pattern(): SessionWebAPIHandler,
    SessionWebLogAPIHandler.get_root_pattern(): SessionWebLogAPIHandler
}


class WebSessionAPI(AbstractSessionAPI, MarsWebAPIClientMixin):
    def __init__(self, address: str):
        self._address = address.rstrip('/')

    async def get_sessions(self) -> List[SessionInfo]:
        addr = f'{self._address}/api/session'
        res = await self._request_url('GET', addr)
        res_obj = json.loads(res.body.decode())
        return [SessionInfo(**kw) for kw in res_obj['sessions']]

    async def create_session(self, session_id: str) -> str:
        addr = f'{self._address}/api/session/{session_id}'
        res = await self._request_url(path=addr, method='PUT', data=b'')
        return res.body.decode()

    async def delete_session(self, session_id: str):
        addr = f'{self._address}/api/session/{session_id}'
        await self._request_url(path=addr, method='DELETE')

    async def has_session(self, session_id: str):
        addr = f'{self._address}/api/session/{session_id}'
        params = dict(action='check_exist')
        res = await self._request_url('GET', addr, params=params)
        return bool(int(res.body.decode()))

    async def get_last_idle_time(self, session_id: Union[str, None] = None) -> Union[float, None]:
        session_id = session_id or ''
        addr = f'{self._address}/api/session/{session_id}'
        params = dict(action='get_last_idle_time')
        res = await self._request_url('GET', addr, params=params)
        content = res.body.decode()
        return float(content) if content else None

    async def fetch_tileable_op_logs(self,
                                     session_id: str,
                                     tileable_op_key: str,
                                     chunk_op_key_to_offsets: Dict[str, List[int]],
                                     chunk_op_key_to_sizes: Dict[str, List[int]]) -> Dict:
        addr = f'{self._address}/api/session/{session_id}/op/{tileable_op_key}/log'
        params = dict(offsets=_encode_size(chunk_op_key_to_offsets),
                      sizes=_encode_size(chunk_op_key_to_sizes))
        res = await self._request_url('GET', addr, params=params)
        return json.loads(res.body.decode())
