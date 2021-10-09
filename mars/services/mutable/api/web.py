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

from typing import Union

from ....utils import deserialize_serializable, serialize_serializable
from ...web import web_api, MarsServiceWebAPIHandler, MarsWebAPIClientMixin
from .core import AbstractMutableAPI


class MutableWebAPIHandler(MarsServiceWebAPIHandler):
    _root_pattern = '/api/session/(?P<session_id>[^/]+)/mutable'

    @web_api('(?P<name>[^/]+)', method='post')
    async def create_mutable_tensor(self, session_id: str, name: str=None):
        body_args = deserialize_serializable(self.request.body) if self.request.body else None
        shape = body_args.get('shape')
        dtype = body_args.get('dtype')
        chunk_size = body_args.get('chunk_size')
        default_value = body_args.get('default_value')

        from .oscar import MutableAPI
        oscar_api = await MutableAPI.create(session_id, self._supervisor_addr)

        res = await oscar_api.create_mutable_tensor(shape, dtype, chunk_size, name, default_value)
        self.write(serialize_serializable(res))

    @web_api('(?P<name>[^/]+)', method='get')
    async def get_mutable_tensor(self, session_id: str, name: str):
        self._session_id = session_id
        from .oscar import MutableAPI
        oscar_api = await MutableAPI.create(self._supervisor_addr)
        res = await oscar_api.get_mutable_tensor(name)
        self.write(serialize_serializable(res))


class WebMutableAPI(AbstractMutableAPI, MarsWebAPIClientMixin):
    def __init__(self,
                 session_id: str,
                 address: str):
        self._session_id = session_id
        self._address = address.rstrip('/')

    async def create_mutable_tensor(self,
                                    shape: tuple,
                                    dtype: str,
                                    chunk_size,
                                    name: str = None,
                                    default_value: Union[int, float] = 0):
        path = f'{self._address}/api/session/{self._session_id}/mutable/{name}'
        params = dict(shape=shape, dtype=dtype, chunk_size=chunk_size, default_value=default_value)
        body = serialize_serializable(params)
        res = await self._request_url(
            path=path, method='POST', data=body,
            headers={'Content-Type': 'application/octet-stream'},
        )
        return deserialize_serializable(res.body)

    async def get_mutable_tensor(self, name: str):
        path = f'{self._address}/api/session/{self._session_id}/mutable/{name}'
        res = await self._request_url(
            path=path, method='GET',
            headers={'Content-Type': 'application/octet-stream'},
        )
        return deserialize_serializable(res.body)
