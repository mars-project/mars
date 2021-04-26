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

import asyncio
import cloudpickle
import inspect
import logging
import sys
import threading
import time

from collections import OrderedDict
from tornado.httpclient import AsyncHTTPClient
from .supervisor import MarsRequestHandler

logger = logging.getLogger(__name__)


def serialize(obj):
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


class _HandlerException(Exception):

    def __init__(self, exc_type, exc_value: BaseException, exc_traceback):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback


_expired_sec = 60
_keep_alive_interval = 3


class ApiRegistry:

    def __init__(self, expired_sec):
        self._apis = dict()
        self._apis_access_time = OrderedDict()
        self._expired_sec = expired_sec

    def add_instance(self, instance):
        api_id = id(instance)
        self._apis[api_id] = instance
        self.keep_alive_api(api_id)
        self.remove_expired_api()
        return api_id

    def get_instance(self, api_id):
        return self._apis[api_id]

    def keep_alive_api(self, api_id):
        assert api_id in self._apis, f'API of {api_id} is expired.'
        self._apis_access_time[api_id] = time.time()
        # Make order consistent with access time so that iteration can exit early.
        self._apis_access_time.move_to_end(api_id)

    def remove_expired_api(self):
        current_time = time.time()
        expired_api = []
        for api_id, last_access_time in self._apis_access_time.items():
            if current_time - last_access_time > self._expired_sec:
                expired_api.append(api_id)
            else:
                # Later last_access_time will be greater so that we can exit early
                break
        for api_id in expired_api:
            self._apis.pop(api_id)
            self._apis_access_time.pop(api_id)


class ServiceWebHandlerBase(MarsRequestHandler):
    _api_registry = ApiRegistry(_expired_sec)
    _api_cls = None

    async def post(self, path, **_):
        api_method_name = path
        # params format: api_id[None], args, kwargs
        api_id, args, kwargs = deserialize(self.request.body)
        try:
            if not api_id:
                if hasattr(self, api_method_name):
                    method = getattr(self, api_method_name)
                else:
                    method = getattr(self._api_cls, api_method_name)
            else:
                # call method of api instance
                method = getattr(self._api_registry.get_instance(api_id), api_method_name)
            if kwargs:
                # Some methods are decorated with @alru_cache, which doesn't support dict as part of cache key.
                result = method(*args, **kwargs)
            else:
                result = method(*args)
            if inspect.iscoroutine(result):
                result = await result
            self.write(serialize(result))
        except Exception as e:
            logger.exception(f'Execute method {api_method_name} with {api_id, args, kwargs} failed, got exception {e}')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.write(serialize(_HandlerException(exc_type, exc_value, exc_traceback)))

    def _keep_alive_api(self, api_id):
        self._api_registry.keep_alive_api(api_id)
        self._api_registry.remove_expired_api()


class ServiceWebAPIBase:
    _service_name = None

    def __init__(self, http_client: AsyncHTTPClient, api_id: int):
        self._http_client = http_client
        self._api_id = api_id
        # TODO(chaokunyang) how to cancel this task
        self._keep_api_alive_task = asyncio.create_task(self._keep_api_alive())

    async def _keep_api_alive(self):
        while True:
            await asyncio.sleep(_keep_alive_interval)
            await self._post(self._http_client, '_keep_alive_api', None, {}, self._api_id)

    def __getattr__(self, method_name):
        async def _func(*args, **kwargs):
            return await self._post(self._http_client, method_name, self._api_id, {}, *args, **kwargs)

        return _func

    @classmethod
    async def _post(cls, http_client: AsyncHTTPClient, api_method_name: str, api_id, req_config, *args, **kwarg):
        req_config = req_config or dict()
        if 'request_timeout' not in req_config:
            req_config['request_timeout'] = 2 * 60 * 60  # timeout for two hours
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{cls._service_name}/{api_method_name}',
                                       method="POST", body=cls._serialize_args(api_id, args, kwarg),
                                       **(req_config or dict()))
        return cls._deserialize_result(resp.body)

    @classmethod
    def _serialize_args(cls, *args):
        return serialize(args)

    @classmethod
    def _deserialize_result(cls, binary):
        result = deserialize(binary)
        if isinstance(result, _HandlerException):
            raise result.exc_value.with_traceback(result.exc_traceback)
        return result
