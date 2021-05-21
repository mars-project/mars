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

import functools
import logging
import os

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
from tornado import web

from ... import oscar as mo
from ...utils import get_next_port

logger = logging.getLogger(__name__)


class BokehStaticFileHandler(web.StaticFileHandler):  # pragma: no cover
    @staticmethod
    def _get_path_root(root, path):
        from bokeh import server
        path_parts = path.rsplit('/', 1)
        if 'bokeh' in path_parts[-1]:
            root = os.path.join(os.path.dirname(server.__file__), "static")
        return root

    @classmethod
    def get_absolute_path(cls, root, path):
        return super().get_absolute_path(cls._get_path_root(root, path), path)

    def validate_absolute_path(self, root, absolute_path):
        return super().validate_absolute_path(
            self._get_path_root(root, absolute_path), absolute_path)


class WebActor(mo.Actor):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._web_server = None

    async def __post_create__(self):
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        supervisor_addr = self.address

        host = self._config.get('host') or '0.0.0.0'
        port = self._config.get('port') or get_next_port()
        self._web_address = f'http://{host}:{port}'
        bokeh_apps = self._config.get('bokeh_apps', {})
        web_handlers = self._config.get('web_handlers', {})

        handlers = dict()
        for p, h in bokeh_apps.items():
            handlers[p] = Application(FunctionHandler(
                functools.partial(h, supervisor_addr=supervisor_addr)))

        handler_kwargs = {'supervisor_addr': supervisor_addr}
        extra_patterns = [
            (r'[^\?\&]*/static/(.*)', BokehStaticFileHandler, {'path': static_path})
        ]
        for p, h in web_handlers.items():
            extra_patterns.append((p, h, handler_kwargs))

        retrial = 5
        while retrial:
            try:
                if port is None:
                    port = get_next_port()

                self._web_server = Server(
                    handlers, allow_websocket_origin=['*'],
                    address=host, port=port,
                    extra_patterns=extra_patterns,
                    http_server_kwargs={'max_buffer_size': 2 ** 32},
                )
                self._web_server.start()
                logger.info('Mars Web started at %s:%d', host, port)
                break
            except OSError:  # pragma: no cover
                if port is not None:
                    raise
                retrial -= 1
                if retrial == 0:
                    raise

    async def __pre_destroy__(self):
        if self._web_server is not None:
            self._web_server.stop()

    def get_web_address(self):
        return self._web_address


async def start(config: dict, address: str = None):
    await mo.create_actor(WebActor, config=config.get('web', {}),
                          uid=WebActor.default_uid(), address=address)
