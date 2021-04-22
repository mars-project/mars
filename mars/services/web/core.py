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
import logging
import sys
import threading
import traceback

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


class ServiceWebHandlerBase(MarsRequestHandler):
    _api_instances = dict()
    _api_cls = None

    async def post(self, path, **_):
        api_method_name = path
        # params format: [api_id], args, kwargs
        params = deserialize(self.request.body)
        try:
            if len(params) == 2:
                if hasattr(self, api_method_name):
                    method = getattr(self, api_method_name)
                else:
                    method = getattr(self._api_cls, api_method_name)
            else:
                # call method of api instance
                assert len(params) == 3
                api_id, params = params[0], params[1:]
                method = getattr(self._api_instances[api_id], api_method_name)
            args, kwargs = params
            if kwargs:
                # Some methods are decorated with @alru_cache, which doesn't support dict as part of cache key.
                result = method(*args, **kwargs)
            else:
                result = method(*args)
            if inspect.iscoroutine(result):
                result = await result
            self.write(serialize(result))
        except Exception as e:
            logger.info(f'Execute method {api_method_name} with {params} failed, got exception {e}')
            traceback.format_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.write(serialize(_HandlerException(exc_type, exc_value, exc_traceback)))


class ServiceWebAPIBase:

    def __init__(self, http_client: AsyncHTTPClient, service_name: str, api_id: int):
        self._http_client = http_client
        self._service_name = service_name
        self._api_id = api_id

    def __getattr__(self, method_name):
        async def _func(*args, **kwargs):
            body = serialize((self._api_id, args, kwargs))
            resp = await self._http_client.fetch(
                f'{get_web_address()}/api/service/{self._service_name}/{method_name}',
                method="POST", body=body)
            return deserialize(resp.body)

        return _func

    @classmethod
    async def _post(cls, http_client, endpoint: str, *args, **kwarg):
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{endpoint}',
                                       method="POST", body=cls._serialize_args(*args, **kwarg))
        return cls._deserialize_result(resp.body)

    @classmethod
    def _serialize_args(cls, *args, **kwargs):
        return serialize((args, kwargs))

    @classmethod
    def _deserialize_result(cls, binary):
        result = deserialize(binary)
        if isinstance(result, _HandlerException):
            raise result.exc_value.with_traceback(result.exc_traceback)
        return result
