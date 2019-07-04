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

import sys
import base64
import json
import pickle
import uuid
import logging

from tornado import gen, concurrent, web, ioloop

from ..actors import new_client
from ..compat import six, futures
from ..errors import GraphNotExists
from ..lib.tblib import pickling_support
from ..serialize.dataserializer import CompressType
from ..utils import to_str
from .server import MarsWebAPI, MarsRequestHandler, register_web_handler

pickling_support.install()
_actor_client = new_client()

logger = logging.getLogger(__name__)


def _patch_futures():  # pragma: no cover
    _FUTURES = concurrent.FUTURES + (futures.Future, )

    def _is_future(x):
        return isinstance(x, _FUTURES)

    gen.is_future = _is_future
    ioloop.is_future = _is_future


if six.PY2:  # pragma: no cover
    _patch_futures()


class MarsApiRequestHandler(MarsRequestHandler):
    def set_default_headers(self):
        super(MarsApiRequestHandler, self).set_default_headers()
        self.set_header('Content-Type', 'application/json')

    def _dump_exception(self, exc_info):
        pickled_exc = pickle.dumps(exc_info)
        self.write(json.dumps(dict(
            exc_info=base64.b64encode(pickled_exc),
        )))
        raise web.HTTPError(500, 'Internal server error')


class ApiEntryHandler(MarsApiRequestHandler):
    def get(self):
        self.write(dict(msg='Mars API Entry'))


class SessionsApiHandler(MarsApiRequestHandler):
    def post(self):
        args = {k: self.get_argument(k) for k in self.request.arguments}
        session_id = str(uuid.uuid1())
        self.web_api.create_session(session_id, **args)
        self.write(json.dumps(dict(session_id=session_id)))


class SessionApiHandler(MarsApiRequestHandler):
    def delete(self, session_id):
        self.web_api.delete_session(session_id)


class GraphsApiHandler(MarsApiRequestHandler):
    def get(self, session_id):
        try:
            graph_states = self.web_api.get_tasks_info(session_id)
            tasks_dict = graph_states[session_id]['tasks']
            self.write(json.dumps(tasks_dict))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    def post(self, session_id):
        try:
            graph = self.get_argument('graph')
            target = self.get_argument('target').split(',')
            compose = bool(int(self.get_argument('compose', '1')))
        except web.MissingArgumentError as ex:
            self.write(json.dumps(dict(msg=str(ex))))
            raise web.HTTPError(400, 'Argument missing')

        try:
            graph_key = str(uuid.uuid4())
            self.web_api.submit_graph(session_id, graph, graph_key, target, compose)
            self.write(json.dumps(dict(graph_key=graph_key)))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())


class GraphApiHandler(MarsApiRequestHandler):
    @gen.coroutine
    def get(self, session_id, graph_key):
        from ..scheduler.utils import GraphState

        try:
            state = self.web_api.get_graph_state(session_id, graph_key)
        except GraphNotExists:
            raise web.HTTPError(404, 'Graph not exists')

        if state == GraphState.RUNNING:
            self.write(json.dumps(dict(state='running')))
        elif state == GraphState.SUCCEEDED:
            self.write(json.dumps(dict(state='success')))
        elif state == GraphState.FAILED:
            self.write(json.dumps(dict(state='failed')))
        elif state == GraphState.CANCELLED:
            self.write(json.dumps(dict(state='cancelled')))
        elif state == GraphState.CANCELLING:
            self.write(json.dumps(dict(state='cancelling')))
        elif state == GraphState.PREPARING:
            self.write(json.dumps(dict(state='preparing')))

    def delete(self, session_id, graph_key):
        self.web_api.stop_graph(session_id, graph_key)


class GraphDataHandler(MarsApiRequestHandler):
    @gen.coroutine
    def get(self, session_id, graph_key, tileable_key):
        type_arg = self.request.arguments.get('type')
        try:
            compressions_arg = self.request.arguments.get('compressions')
            if compressions_arg:
                compressions_arg = [CompressType(s) for s in to_str(compressions_arg[0]).split(',') if s]
        except (TypeError, ValueError):
            raise web.HTTPError(403, 'Malformed encodings')
        if type_arg:
            data_type = to_str(type_arg[0])
            if data_type == 'nsplits':
                nsplits = self.web_api.get_tileable_nsplits(session_id, graph_key, tileable_key)
                self.write(json.dumps(nsplits))
            else:
                raise web.HTTPError(403, 'Unknown data type requests')
        else:
            executor = futures.ThreadPoolExecutor(1)

            def _fetch_fun():
                web_api = MarsWebAPI(self._scheduler)
                return web_api.fetch_data(session_id, graph_key, tileable_key, compressions_arg)

            data = yield executor.submit(_fetch_fun)
            self.write(data)

    def delete(self, session_id, graph_key, tileable_key):
        self.web_api.delete_data(session_id, graph_key, tileable_key)


class WorkersApiHandler(MarsApiRequestHandler):
    def get(self):
        workers_num = self.web_api.count_workers()
        self.write(json.dumps(workers_num))


register_web_handler('/api', ApiEntryHandler)
register_web_handler('/api/session', SessionsApiHandler)
register_web_handler('/api/worker', WorkersApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)', SessionApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph', GraphsApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)', GraphApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)/data/(?P<tileable_key>[^/]+)',
                     GraphDataHandler)
