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

import cloudpickle
import inspect
import threading

from tornado.httpclient import AsyncHTTPClient
from .supervisor import MarsRequestHandler


def serialize(*obj):
    from mars.serialization import serialize
    return cloudpickle.dumps(serialize(obj))


def deserialize(binary):
    from mars.serialization import deserialize
    return deserialize(*(cloudpickle.loads(binary)))


_web_address_local = threading.local()


def set_web_address(web_address):
    _web_address_local.web_address = web_address


def get_web_address():
    assert hasattr(_web_address_local, 'web_address'), 'Please set web_address first before get web address'
    return _web_address_local.web_address


class ServiceWebHandlerBase(MarsRequestHandler):

    def initialize(self, supervisor_addr):
        super().initialize(supervisor_addr)
        self._api_instances = dict()

    async def post(self, path, **kwargs):
        api_method = path
        params = deserialize(self.request.body)
        if hasattr(self, api_method):
            result = getattr(self, api_method)(*params)
            if inspect.iscoroutine(result):
                result = await result
            self.write(serialize(result))
        else:
            api_id = params[0]
            result = getattr(self._api_instances[api_id], api_method)(*params[1:])
            if inspect.iscoroutine(result):
                result = await result
            self.write(serialize(result))


class ServiceWebAPIBase:

    def __init__(self, http_client: AsyncHTTPClient, service_name: str, api_id: int):
        self._http_client = http_client
        self._service_name = service_name
        self._api_id = api_id

    def __getattr__(self, method_name):
        async def _func(*args, **kwargs):
            resp = await self._http_client.fetch(
                f'{get_web_address()}//api/service/{self._service_name}/{method_name}',
                method="POST", body=serialize((self._api_id, args, kwargs)))
            return deserialize(resp.body)

        return _func
