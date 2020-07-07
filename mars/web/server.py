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
import json
import logging
import threading
import os
from collections import defaultdict

import numpy as np
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
import jinja2
from tornado import web, ioloop

from ..utils import get_next_port
from ..scheduler import SessionActor
from ..api import MarsAPI

logger = logging.getLogger(__name__)


def get_jinja_env():
    from datetime import datetime
    from ..utils import readable_size

    _jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
    )

    def format_ts(value):
        return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S') \
            if value is not None and not np.isnan(value) else None

    _jinja_env.filters['format_ts'] = format_ts
    _jinja_env.filters['readable_size'] = readable_size
    return _jinja_env


class BokehStaticFileHandler(web.StaticFileHandler):
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


class MarsRequestHandler(web.RequestHandler):
    def initialize(self, scheduler_ip):
        self._scheduler = scheduler_ip
        self.web_api = MarsWebAPI(scheduler_ip)


class MarsWebAPI(MarsAPI):
    _schedulers_cache = None

    def get_schedulers(self):
        cls = type(self)
        if not cls._schedulers_cache:
            cls._schedulers_cache = self.cluster_info.get_schedulers()
        return cls._schedulers_cache

    def get_tasks_info(self, select_session_id=None):
        from ..scheduler import GraphState

        sessions = defaultdict(dict)
        for session_id, session_ref in self.session_manager.get_sessions().items():
            if select_session_id and session_id != select_session_id:
                continue
            session_desc = sessions[session_id]
            session_desc['id'] = session_id
            session_desc['name'] = session_id
            session_desc['tasks'] = dict()
            session_ref = self.actor_client.actor_ref(session_ref)
            for graph_key, graph_meta_ref in session_ref.get_graph_meta_refs().items():
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

    def get_operand_info(self, session_id, task_id, state=None):
        graph_meta_ref = self.get_graph_meta_ref(session_id, task_id)
        return graph_meta_ref.get_operand_info(state=state)

    def query_worker_events(self, endpoint, category, time_start=None, time_end=None):
        from ..worker import EventsActor
        ref = self.actor_client.actor_ref(EventsActor.default_uid(), address=endpoint)
        return ref.query_by_time(category, time_start=time_start, time_end=time_end)

    def write_mutable_tensor(self, session_id, name, payload_type, body):
        import pyarrow

        from ..serialize import dataserializer
        from ..tensor.core import Indexes
        session_uid = SessionActor.gen_uid(session_id)
        session_ref = self.get_actor_ref(session_uid)

        index_json_size = np.frombuffer(body[0:8], dtype=np.int64).item()
        index_json = json.loads(body[8:8+index_json_size].decode('ascii'))
        index = Indexes.from_json(index_json).indexes
        if payload_type is None:
            value = dataserializer.loads(body[8+index_json_size:])
        elif payload_type == 'tensor':
            tensor_chunk_offset = 8 + index_json_size
            with pyarrow.BufferReader(body[tensor_chunk_offset:]) as reader:
                value = pyarrow.read_tensor(reader).to_numpy()
        elif payload_type == 'record_batch':
            schema_size = np.frombuffer(body[8+index_json_size:8+index_json_size+8], dtype=np.int64).item()
            schema_offset = 8 + index_json_size + 8
            with pyarrow.BufferReader(body[schema_offset:schema_offset+schema_size]) as reader:
                schema = pyarrow.read_schema(reader)
            record_batch_offset = schema_offset + schema_size
            with pyarrow.BufferReader(body[record_batch_offset:]) as reader:
                record_batch = pyarrow.read_record_batch(reader, schema)
                value = record_batch.to_pandas().to_records(index=False)
        else:
            raise ValueError('Not supported payload type: %s' % payload_type)
        return session_ref.write_mutable_tensor(name, index, value)


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
            import asyncio
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = None
            try:
                loop = ioloop.IOLoop.current()
            except:  # noqa: E722
                pass
            if loop is None:
                raise

    def _try_start_web_server(self):
        static_path = os.path.join(os.path.dirname(__file__), 'static')

        handlers = dict()
        for p, h in _bokeh_apps.items():
            handlers[p] = Application(FunctionHandler(
                functools.partial(h, scheduler_ip=self._scheduler_ip)))

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
                    http_server_kwargs={'max_buffer_size': 2 ** 32},
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

            try:
                self._server.io_loop.start()
            except KeyboardInterrupt:
                pass

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
