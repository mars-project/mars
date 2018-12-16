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

import os
import threading
import logging
import functools
from collections import defaultdict

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
import jinja2
from tornado import web, ioloop

from .. import kvstore
from ..compat import six
from ..utils import get_next_port
from ..config import options
from ..scheduler import GraphActor, ResourceActor
from ..api import MarsAPI

logger = logging.getLogger(__name__)


def get_jinja_env():
    from datetime import datetime
    from ..utils import readable_size

    _jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
        )

    def format_ts(value):
        if value is None:
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


class MarsWebAPI(MarsAPI):
    def __init__(self, scheduler_ip):
        super(MarsWebAPI, self).__init__(scheduler_ip)

    def get_tasks_info(self):
        sessions = defaultdict(dict)
        for session_id, session_ref in six.iteritems(self.session_manager.get_sessions()):
            session_desc = sessions[session_id]
            session_desc['id'] = session_id
            session_desc['name'] = session_id
            session_desc['tasks'] = dict()
            session_ref = self.actor_client.actor_ref(session_ref)
            for graph_key, graph_ref in six.iteritems(session_ref.get_graph_refs()):
                task_desc = dict()

                state = self.kv_store.read(
                    '/sessions/%s/graph/%s/state' % (session_id, graph_key)).value
                if state == 'PREPARING':
                    task_desc['state'] = state.lower()
                    session_desc['tasks'][graph_key] = task_desc
                    continue

                graph_ref = self.actor_client.actor_ref(graph_ref)
                task_desc['id'] = graph_key
                task_desc['state'] = graph_ref.get_state().value
                start_time, end_time, graph_size = graph_ref.get_graph_info()
                task_desc['start_time'] = start_time
                task_desc['end_time'] = end_time or 'N/A'
                task_desc['graph_size'] = graph_size or 'N/A'

                session_desc['tasks'][graph_key] = task_desc
        return sessions

    def get_task_detail(self, session_id, task_id):
        graph_uid = GraphActor.gen_name(session_id, task_id)
        graph_ref = self.get_actor_ref(graph_uid)
        return graph_ref.calc_stats()

    def get_workers_meta(self):
        resource_uid = ResourceActor.default_name()
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

    def start(self, event=None, block=False):
        try:
            ioloop.IOLoop.current()
        except RuntimeError:
            if six.PY3:
                import asyncio
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = None
                try:
                    loop = ioloop.IOLoop.current()
                except:
                    pass
                if loop is None:
                    raise
            else:
                raise

        if self._scheduler_ip is None:
            kv_store = kvstore.get(options.kv_store)
            try:
                schedulers = [s.key.rsplit('/', 1)[1] for s in kv_store.read('/schedulers').children]
                self._scheduler_ip = schedulers[0]
            except KeyError:
                raise KeyError('No scheduler is available')

        static_path = os.path.join(os.path.dirname(__file__), 'static')

        handlers = dict()
        for p, h in _ui_handlers.items():
            handlers[p] = Application(FunctionHandler(functools.partial(h, self._scheduler_ip)))
        extra_patterns = [
            ('/static/(.*)', BokehStaticFileHandler, {'path': static_path})
        ]
        for p, h in _api_handlers.items():
            extra_patterns.append((p, h, {'scheduler_ip': self._scheduler_ip}))

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
            except:
                if self._port is not None:
                    raise
                retrial -= 1
                if retrial == 0:
                    raise

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


_ui_handlers = dict()
_api_handlers = dict()


def register_ui_handler(pattern, handler):
    _ui_handlers[pattern] = handler


def register_api_handler(pattern, handler):
    _api_handlers[pattern] = handler
