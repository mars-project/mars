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

from .server import register_api_handler
from ..compat import six, futures
from ..lib.tblib import pickling_support
from ..actors import new_client
from .server import MarsWebAPI

pickling_support.install()
_actor_client = new_client()

logger = logging.getLogger(__name__)


def _patch_futures():
    _FUTURES = concurrent.FUTURES + (futures.Future, )

    def _is_future(x):
        return isinstance(x, _FUTURES)

    gen.is_future = _is_future
    ioloop.is_future = _is_future


if six.PY2:
    _patch_futures()


class ApiRequestHandler(web.RequestHandler):
    def initialize(self, scheduler_ip):
         self._scheduler = scheduler_ip
         self.web_api = MarsWebAPI(scheduler_ip)


class ApiEntryHandler(ApiRequestHandler):
    def get(self):
        self.write(dict(msg='Mars API Entry'))


class SessionsApiHandler(ApiRequestHandler):
    def post(self):
        args = {k: self.get_argument(k) for k in self.request.arguments}
        session_id = str(uuid.uuid1())
        self.web_api.create_session(session_id, **args)
        self.write(json.dumps(dict(session_id=session_id)))


class SessionApiHandler(ApiRequestHandler):
    def delete(self, session_id):
        self.web_api.delete_session(session_id)


class GraphsApiHandler(ApiRequestHandler):
    def post(self, session_id):
        try:
            graph = self.get_argument('graph')
            target = self.get_argument('target').split(',')
        except web.MissingArgumentError as ex:
            self.write(json.dumps(dict(msg=str(ex))))
            raise web.HTTPError(400, 'Argument missing')

        try:
            graph_key = str(uuid.uuid4())
            self.web_api.submit_graph(session_id, graph, graph_key, target)
            self.write(json.dumps(dict(graph_key=graph_key)))
        except:
            pickled_exc = pickle.dumps(sys.exc_info())
            self.write(json.dumps(dict(
                exc_info=base64.b64encode(pickled_exc),
            )))
            raise web.HTTPError(500, 'Internal server error')


class GraphApiHandler(ApiRequestHandler):
    @gen.coroutine
    def get(self, session_id, graph_key):
        from ..scheduler.utils import GraphState

        state = self.web_api.get_graph_state(session_id, graph_key)
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


class GraphDataHandler(ApiRequestHandler):
    @gen.coroutine
    def get(self, session_id, graph_key, tensor_key):
        executor = futures.ThreadPoolExecutor(1)

        def _fetch_fun():
            web_api = MarsWebAPI(self._scheduler)
            return web_api.fetch_data(session_id, graph_key, tensor_key)

        data = yield executor.submit(_fetch_fun)
        self.write(data)

    def delete(self, session_id, graph_key, tensor_key):
        self.web_api.delete_data(session_id, graph_key, tensor_key)


class WorkersApiHandler(ApiRequestHandler):
    def get(self):
        workers_num = self.web_api.count_workers()
        self.write(json.dumps(workers_num))


register_api_handler('/api', ApiEntryHandler)
register_api_handler('/api/session', SessionsApiHandler)
register_api_handler('/api/worker', WorkersApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)', SessionApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph', GraphsApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)', GraphApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)/data/(?P<tensor_key>[^/]+)',
                     GraphDataHandler)
