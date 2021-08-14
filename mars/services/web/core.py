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

import asyncio
import functools
import inspect
import logging
import re
import sys
import urllib.parse
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Union

from tornado import httpclient, web

from ...utils import serialize_serializable, deserialize_serializable

if sys.version_info[:2] == (3, 6):
    # make sure typing works
    re.Pattern = type(re.compile(r'.*'))

logger = logging.getLogger(__name__)
_ROOT_PLACEHOLDER = 'ROOT_PLACEHOLDER'


class MarsRequestHandler(web.RequestHandler):  # pragma: no cover
    def initialize(self, supervisor_addr: str = None):
        self._supervisor_addr = supervisor_addr


class _WebApiDef(NamedTuple):
    sub_pattern: str
    sub_pattern_compiled: re.Pattern
    method: str
    arg_filter: Optional[Dict] = None


def web_api(sub_pattern: str, method: Union[str, List[str]],
            arg_filter: Optional[Dict] = None):
    if not sub_pattern.endswith('$'):  # pragma: no branch
        sub_pattern += '$'
    methods = method if isinstance(method, list) else [method]

    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(self, *args, **kwargs):
            try:
                res = func(self, *args, **kwargs)
                if inspect.isawaitable(res):
                    res = await res
                return res
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                exc_type, exc, tb = sys.exc_info()
                err_msg = f'{exc_type.__name__} when handling request with ' \
                          f'{type(self).__name__}.{func.__name__}'
                logger.exception(err_msg)
                self.write(serialize_serializable((exc, tb)))
                self.set_status(500, err_msg)

        wrapped._web_api_defs = [
            _WebApiDef(sub_pattern, re.compile(sub_pattern), m, arg_filter)
            for m in methods
        ]
        return wrapped

    return wrapper


class MarsServiceWebAPIHandler(MarsRequestHandler):
    _root_pattern = None
    _method_to_handlers = None

    def __init__(self, *args, **kwargs):
        self._collect_services()
        super().__init__(*args, **kwargs)

    @classmethod
    def _collect_services(cls):
        if cls._method_to_handlers is not None:
            return

        cls._method_to_handlers = defaultdict(dict)
        for attr in dir(cls):
            handle_func = getattr(cls, attr, None)
            if not hasattr(handle_func, '_web_api_defs'):
                continue
            web_api_defs = getattr(handle_func, '_web_api_defs')  # type: List[_WebApiDef]
            for api_def in web_api_defs:
                cls._method_to_handlers[api_def.method.lower()][handle_func] = api_def

    @classmethod
    def get_root_pattern(cls):
        return cls._root_pattern + '(?:/(?P<sub_path>.*)$|$)'

    @functools.lru_cache(100)
    def _route_sub_path(self, http_method: str, sub_path: str):
        handlers = self._method_to_handlers[http_method.lower()]  # type: Dict[Callable, _WebApiDef]
        method, kwargs = None, None
        for handler_method, web_api_def in handlers.items():
            match = web_api_def.sub_pattern_compiled.match(sub_path)
            if match is not None:
                if web_api_def.arg_filter is not None:
                    if not all(self.get_argument(k, None) == v
                               for k, v in web_api_def.arg_filter.items()):
                        continue
                    method, kwargs = handler_method, dict(match.groupdict())
                elif method is None:
                    # method matched with arg_filter shall not be overwritten
                    method, kwargs = handler_method, dict(match.groupdict())
        if method is not None:
            return method, kwargs
        else:
            raise web.HTTPError(404, f'{sub_path} does not match any defined APIs '
                                f'with method {http_method.upper()}')

    def _make_handle_http_method(http_method: str):
        async def _handle_http_method(self: "MarsServiceWebAPIHandler", **kwargs):
            sub_path = kwargs.pop('sub_path', None) or ''
            method, kw = self._route_sub_path(http_method, sub_path)
            kw.update(kwargs)
            res = method(self, **kw)
            if inspect.isawaitable(res):
                await res

        _handle_http_method.__name__ = http_method.lower()
        return _handle_http_method

    get = _make_handle_http_method('get')
    put = _make_handle_http_method('put')
    post = _make_handle_http_method('post')
    patch = _make_handle_http_method('patch')
    delete = _make_handle_http_method('delete')

    del _make_handle_http_method


class MarsWebAPIClientMixin:
    @property
    def _client(self):
        try:
            return self._client_obj
        except AttributeError:
            self._client_obj = httpclient.AsyncHTTPClient()
            return self._client_obj

    def __del__(self):
        if hasattr(self, '_client_obj'):
            self._client_obj.close()

    async def _request_url(self, method, path, **kwargs):
        self._running_loop = asyncio.get_running_loop()

        if 'data' in kwargs:
            kwargs['body'] = kwargs.pop('data')

        if 'params' in kwargs:
            params = kwargs.pop('params')
            for k, v in params.items():
                if isinstance(v, (list, tuple, set)):
                    params[k] = ','.join(str(i) for i in v)
            url_params = urllib.parse.urlencode(params)
            path_connector = '?' if '?' not in path else '&'
            path += path_connector + url_params

        res = await self._client.fetch(path, method=method, raise_error=False, **kwargs)
        if res.code < 400:
            return res
        else:
            exc, tb = None, None
            try:
                exc, tb = deserialize_serializable(res.body)
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                pass

            if exc is None:
                raise res.error
            else:
                raise exc.with_traceback(tb)
