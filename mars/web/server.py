# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
import random
import threading
import os
from collections import defaultdict

import numpy as np
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
import jinja2
from tornado import web, ioloop

from .. import kvstore
from ..compat import six
from ..utils import get_next_port
from ..config import options
from ..scheduler import ResourceActor
from ..api import MarsAPI

logger = logging.getLogger(__name__)


def get_jinja_env():
    from datetime import datetime
    from ..utils import readable_size

    _jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
    )

    def format_ts(value):
        if value is None or np.isnan(value):
            return None
        return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')

    _jinja_env.filters['format_ts'] = format_ts
    _jinja_env.filters['readable_size'] = readable_size
    return _jinja_env


class BokehStaticFileHandler(web.StaticFileHandler):
    @classmethod
    def get_absolute_path(cls, root, path):
        from bokeh import server
        path_parts = path.rsplit('/', 1)
        if 'bokeh' in path_parts[-1]:
            root = os.path.join(os.path.dirname(server.__file__), "static")
        return super(BokehStaticFileHandler, cls).get_absolute_path(root, path)

    def validate_absolute_path(self, root, absolute_path):
        from bokeh import server
        path_parts = absolute_path.rsplit('/', 1)
        if 'bokeh' in path_parts[-1]:
            root = os.path.join(os.path.dirname(server.__file__), "static")
        return super(BokehStaticFileHandler, self).validate_absolute_path(root, absolute_path)


class MarsRequestHandler(web.RequestHandler):
    def initialize(self, scheduler_ip):
        self._scheduler = scheduler_ip
        self.web_api = MarsWebAPI(scheduler_ip)


class MarsWebAPI(MarsAPI):
    def __init__(self, scheduler_ip):
        super(MarsWebAPI, self).__init__(scheduler_ip)

    def get_tasks_info(self, select_session_id=None):
        from ..scheduler import GraphState

        sessions = defaultdict(dict)
        for session_id, session_ref in six.iteritems(self.session_manager.get_sessions()):
            if select_session_id and session_id != select_session_id:
                continue
            session_desc = sessions[session_id]
            session_desc['id'] = session_id
            session_desc['name'] = session_id
            session_desc['tasks'] = dict()
            session_ref = self.actor_client.actor_ref(session_ref)
            for graph_key, graph_meta_ref in six.iteritems(session_ref.get_graph_meta_refs()):
                task_desc = dict()

                state = self.get_graph_state(session_id, graph_key)
                if state == GraphState.PREPARING:
                    task_desc['state'] = state.name.lower()
                    session_desc['tasks'][graph_key] = task_desc
                    continue

                graph_meta_ref = self.actor_client.actor_ref(graph_meta_ref)
                task_desc['id'] = graph_key
                task_desc['state'] = graph_meta_ref.get_state().value
                start_time, end_time, graph_size = graph_meta_ref.get_graph_info()
                task_desc['start_time'] = start_time
                task_desc['end_time'] = end_time
                task_desc['graph_size'] = graph_size

                session_desc['tasks'][graph_key] = task_desc
        return sessions

    def get_task_detail(self, session_id, task_id):
        graph_meta_ref = self.get_graph_meta_ref(session_id, task_id)
        return graph_meta_ref.calc_stats()

    def get_workers_meta(self):
        resource_uid = ResourceActor.default_uid()
        resource_ref = self.get_actor_ref(resource_uid)
        return resource_ref.get_workers_meta()


class MarsWeb(object):
    def __init__(self, port=None, scheduler_ip=None):
        self._port = port
        self._scheduler_ip = scheduler_ip
        self._server = None
        self._server_thread = None

    @property
    def port(self):
        return self._port

    @staticmethod
    def _configure_loop():
        try:
            ioloop.IOLoop.current()
        except RuntimeError:
            if six.PY3:
                import asyncio
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = None
                try:
                    loop = ioloop.IOLoop.current()
                except:  # noqa: E722
                    pass
                if loop is None:
                    raise
            else:
                raise

    def _try_start_web_server(self):
        static_path = os.path.join(os.path.dirname(__file__), 'static')

        handlers = dict()
        for p, h in _bokeh_apps.items():
            handlers[p] = Application(FunctionHandler(functools.partial(h, self._scheduler_ip)))

        handler_kwargs = {'scheduler_ip': self._scheduler_ip}
        extra_patterns = [
            ('/static/(.*)', BokehStaticFileHandler, {'path': static_path})
        ]
        for p, h in _web_handlers.items():
            extra_patterns.append((p, h, handler_kwargs))

        retrial = 5
        while retrial:
            try:
                if self._port is None:
                    use_port = get_next_port()
                else:
                    use_port = self._port

                self._server = Server(
                    handlers, allow_websocket_origin=['*'],
                    address='0.0.0.0', port=use_port,
                    extra_patterns=extra_patterns,
                )
                self._server.start()
                self._port = use_port
                logger.info('Mars UI started at 0.0.0.0:%d', self._port)
                break
            except OSError:
                if self._port is not None:
                    raise
                retrial -= 1
                if retrial == 0:
                    raise

    def start(self, event=None, block=False):
        self._configure_loop()

        if self._scheduler_ip is None:
            kv_store = kvstore.get(options.kv_store)
            try:
                schedulers = [s.key.rsplit('/', 1)[1] for s in kv_store.read('/schedulers').children]
                self._scheduler_ip = random.choice(schedulers)
            except KeyError:
                raise KeyError('No scheduler is available')

        self._try_start_web_server()

        if not block:
            self._server_thread = threading.Thread(target=self._server.io_loop.start)
            self._server_thread.daemon = True
            self._server_thread.start()

            if event:
                event.set()
        else:
            if event:
                event.set()

            self._server.io_loop.start()

    def stop(self):
        if self._server is not None:
            self._server.io_loop.stop()
            self._server.stop()


_bokeh_apps = dict()
_web_handlers = dict()


def register_bokeh_app(pattern, handler):
    _bokeh_apps[pattern] = handler


def register_web_handler(pattern, handler):
    _web_handlers[pattern] = handler
