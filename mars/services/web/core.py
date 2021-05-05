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
import collections
import inspect
import logging
import sys

from tornado.httpclient import AsyncHTTPClient
from .supervisor import MarsRequestHandler

logger = logging.getLogger(__name__)


def serialize(obj):
    from mars.serialization import serialize
    return cloudpickle.dumps(serialize(obj))


def deserialize(binary):
    from mars.serialization import deserialize
    return deserialize(*(cloudpickle.loads(binary)))


class _HandlerException(Exception):

    def __init__(self, exc_type, exc_value: BaseException, exc_traceback):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback


_FuncSpec = collections.namedtuple('_FuncSpec', ['method_name', 'args', 'kwargs'])


class ServiceProxyHandlerBase(MarsRequestHandler):
    _api_cls = None

    async def post(self):
        api_creation_spec, api_call_spec = deserialize(self.request.body)
        try:
            try:
                api_creation_method = getattr(self._api_cls, api_creation_spec.method_name)
                if api_creation_spec.kwargs:
                    # Some methods are decorated with @alru_cache, which doesn't support dict as part of cache key.
                    api = api_creation_method(*api_creation_spec.args, **api_creation_spec.kwargs)
                else:
                    api = api_creation_method(*api_creation_spec.args)
                if inspect.isawaitable(api):
                    api = await api
            except Exception:
                logger.exception(f'Create api of {self._api_cls} with {api_creation_spec} failed')
                raise
            method = getattr(api, api_call_spec.method_name)
            if api_call_spec.kwargs:
                # Some methods are decorated with @alru_cache, which doesn't support dict as part of cache key.
                result = method(*api_call_spec.args, **api_call_spec.kwargs)
            else:
                result = method(*api_call_spec.args)
            if inspect.isawaitable(result):
                result = await result
            self.write(serialize(result))
        except Exception as e:
            logger.exception(f'Execute method with {api_call_spec} failed, got exception {e}')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.write(serialize(_HandlerException(exc_type, exc_value, exc_traceback)))


def get_service_proxy_endpoint(service_name: str):
    return f'/api/service/{service_name}/__proxy__'


class ServiceWebAPIBase:
    _service_name = None

    def __init__(self, address: str, api_creation_method_name, api_creation_args, api_creation_kwargs):
        self._http_client = AsyncHTTPClient()
        self._address = address
        self._func_spec = _FuncSpec(api_creation_method_name, api_creation_args, api_creation_kwargs)

    def __getattr__(self, method_name):
        async def _func(*args, **kwargs):
            return await self._post(self._http_client, {}, method_name, *args, **kwargs)

        return _func

    async def _post(self, http_client: AsyncHTTPClient, req_config, api_method_name: str, *args, **kwarg):
        resp = await http_client.fetch(
            self._address + get_service_proxy_endpoint(self._service_name),
            method="POST", body=self._serialize_args(self._func_spec, _FuncSpec(api_method_name, args, kwarg)),
            **(req_config or dict()))
        return self._deserialize_result(resp.body)

    @classmethod
    def _serialize_args(cls, *args):
        return serialize(args)

    @classmethod
    def _deserialize_result(cls, binary):
        result = deserialize(binary)
        if isinstance(result, _HandlerException):
            raise result.exc_value.with_traceback(result.exc_traceback)
        return result
