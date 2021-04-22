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
import requests
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
                method = getattr(self._api_instances[api_id], api_method_name)
            if kwargs:
                # Some methods are decorated with @alru_cache, which doesn't support dict as part of cache key.
                result = method(*args, **kwargs)
            else:
                result = method(*args)
            if inspect.iscoroutine(result):
                result = await result
            self.write(serialize(result))
        except Exception as e:
            logger.info(f'Execute method {api_method_name} with {api_id, args, kwargs} failed, got exception {e}')
            traceback.format_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.write(serialize(_HandlerException(exc_type, exc_value, exc_traceback)))

    def __destroy_api__(self, api_id):
        api = self._api_instances.pop(api_id, None)
        print('Destroyed api %s', api)


class ServiceWebAPIBase:
    _service_name = None

    def __init__(self, http_client: AsyncHTTPClient, api_id: int):
        self._http_client = http_client
        self._api_id = api_id

    def __getattr__(self, method_name):
        async def _func(*args, **kwargs):
            return await self._post(self._http_client, method_name, *args, api_id=self._api_id, **kwargs)

        return _func

    def __del__(self):
        try:
            self._sync_post('__destroy_api__', api_id=self._api_id, req_config=dict(timeout=(0.1, 0.5)))
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
            # server is closed
            pass

    @classmethod
    async def _post(cls, http_client: AsyncHTTPClient, api_method_name: str,
                    *args, api_id=None, req_config=None, **kwarg):
        req_config = req_config or dict()
        if 'request_timeout' not in req_config:
            req_config['request_timeout'] = 2 * 60 * 60  # timeout for two hours
        resp = await http_client.fetch(f'{get_web_address()}/api/service/{cls._service_name}/{api_method_name}',
                                       method="POST", body=cls._serialize_args(api_id, args, kwarg),
                                       **(req_config or dict()))
        return cls._deserialize_result(resp.body)

    @classmethod
    def _sync_post(cls, api_method_name: str,
                   *args, api_id=None, req_config=None, **kwarg):
        req_config = req_config or dict()
        if 'timeout' not in req_config:
            req_config['timeout'] = 2 * 60 * 60  # timeout for two hours
        r = requests.post(f'{get_web_address()}/api/service/{cls._service_name}/{api_method_name}',
                          data=cls._serialize_args(api_id, args, kwarg), **req_config)
        return cls._deserialize_result(r.content)

    @classmethod
    def _serialize_args(cls, *args):
        return serialize(args)

    @classmethod
    def _deserialize_result(cls, binary):
        result = deserialize(binary)
        if isinstance(result, _HandlerException):
            raise result.exc_value.with_traceback(result.exc_traceback)
        return result
