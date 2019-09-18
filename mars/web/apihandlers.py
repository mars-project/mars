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
import logging
import pickle
import traceback
import uuid

from tornado import gen, concurrent, web, ioloop

from ..tensor.core import Indexes
from ..actors import new_client
from ..compat import six, futures
from ..errors import GraphNotExists
from ..lib.tblib import pickling_support
from ..serialize.dataserializer import CompressType
from ..utils import to_str, tokenize, numpy_dtype_from_descr_json
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

    def _dump_exception(self, exc_info, status_code=500):
        pickled_exc = pickle.dumps(exc_info)
        # return pickled exc_info for python client, and textual exc_info for web.
        self.write(json.dumps(dict(
            exc_info=base64.b64encode(pickled_exc).decode('ascii'),
            exc_info_text=traceback.format_exception(*exc_info),
        )))
        self.set_status(status_code)
        self.finish()


class ApiEntryHandler(MarsApiRequestHandler):
    def get(self):
        self.write(dict(msg='Mars API Entry'))


class SessionsApiHandler(MarsApiRequestHandler):
    def post(self):
        args = {k: self.get_argument(k) for k in self.request.arguments}

        try:
            python_version = tuple(int(p) for p in args.pop('pyver').split('.')) \
                if 'pyver' in args else sys.version_info
        except ValueError as ex:
            raise web.HTTPError(400, reason='Invalid version data: %s' % ex)
        if python_version[0] != sys.version_info[0]:
            raise web.HTTPError(400, reason='Python version not consistent')

        session_id = tokenize(str(uuid.uuid1()))
        self.web_api.create_session(session_id, **args)
        self.write(json.dumps(dict(session_id=session_id)))


class SessionApiHandler(MarsApiRequestHandler):
    def head(self, session_id):
        if not self.web_api.has_session(session_id):
            raise web.HTTPError(404, 'Session doesn\'t not exists')

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
            raise web.HTTPError(400, reason='Argument missing')

        try:
            graph_key = tokenize(str(uuid.uuid4()))
            self.web_api.submit_graph(session_id, graph, graph_key, target, compose)
            self.write(json.dumps(dict(graph_key=graph_key)))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())


class GraphApiHandler(MarsApiRequestHandler):
    _executor = futures.ThreadPoolExecutor(1)

    @gen.coroutine
    def get(self, session_id, graph_key):
        from ..scheduler.utils import GraphState
        wait_timeout = int(self.get_argument('wait_timeout', None))

        try:
            if wait_timeout:
                if wait_timeout <= 0:
                    wait_timeout = None

                def _wait_fun():
                    web_api = MarsWebAPI(self._scheduler)
                    return web_api.wait_graph_finish(session_id, graph_key, wait_timeout)

                _ = yield self._executor.submit(_wait_fun)  # noqa: F841

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
        try:
            self.web_api.stop_graph(session_id, graph_key)
        except:  # noqa: E722
            self._dump_exception(sys.exc_info(), 404)


class GraphDataApiHandler(MarsApiRequestHandler):
    _executor = futures.ThreadPoolExecutor(1)

    @gen.coroutine
    def get(self, session_id, graph_key, tileable_key):
        data_type = self.get_argument('type', None)
        try:
            compressions_arg = self.get_argument('compressions', None)
            if compressions_arg:
                compressions_arg = [CompressType(s) for s in compressions_arg.split(',') if s]
            slices_arg = self.request.arguments.get('slices')
            if slices_arg:
                slices_arg = Indexes.from_json(json.loads(to_str(slices_arg[0]))).indexes
        except (TypeError, ValueError):
            raise web.HTTPError(403, 'Malformed encodings')
        if data_type:
            if data_type == 'nsplits':
                nsplits = self.web_api.get_tileable_nsplits(session_id, graph_key, tileable_key)
                self.write(json.dumps(nsplits))
            else:
                raise web.HTTPError(403, 'Unknown data type requests')
        else:

            def _fetch_fun():
                web_api = MarsWebAPI(self._scheduler)
                return web_api.fetch_data(session_id, graph_key, tileable_key, index_obj=slices_arg,
                                          compressions=compressions_arg)

            data = yield self._executor.submit(_fetch_fun)
            self.write(data)

    def delete(self, session_id, graph_key, tileable_key):
        wait = int(self.get_argument('wait', '0'))
        self.web_api.delete_data(session_id, graph_key, tileable_key, wait=wait)


class WorkersApiHandler(MarsApiRequestHandler):
    def get(self):
        action = self.get_argument('action', None)
        if action == 'count':
            self.write(json.dumps(self.web_api.count_workers()))
        else:
            self.write(json.dumps(self.web_api.get_workers_meta()))


class MutableTensorApiHandler(MarsApiRequestHandler):
    def get(self, session_id, name):
        try:
            meta = self.web_api.get_mutable_tensor(session_id, name)
            self.write(json.dumps(meta))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    def post(self, session_id, name):
        try:
            action = self.get_argument('action', None)
            if action == 'create':
                req_json = json.loads(self.request.body.decode('ascii'))
                shape = req_json['shape']
                dtype = numpy_dtype_from_descr_json(req_json['dtype'])
                fill_value = req_json['fill_value']
                chunk_size = req_json['chunk_size']
                meta = self.web_api.create_mutable_tensor(session_id, name, shape, dtype,
                                                          fill_value=fill_value, chunk_size=chunk_size)
                self.write(json.dumps(meta))
            elif action == 'seal':
                info = self.web_api.seal(session_id, name)
                self.write(json.dumps(info))
            else:
                raise web.HTTPError(400, reason='Invalid argument')
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    def put(self, session_id, name):
        try:
            payload_type = self.get_argument('payload_type', None)
            body = self.request.body
            self.web_api.write_mutable_tensor(session_id, name, payload_type, body)
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())


register_web_handler('/api', ApiEntryHandler)
register_web_handler('/api/session', SessionsApiHandler)
register_web_handler('/api/worker', WorkersApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)', SessionApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph', GraphsApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)', GraphApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)/data/(?P<tileable_key>[^/]+)',
                     GraphDataApiHandler)
register_web_handler('/api/session/(?P<session_id>[^/]+)/mutable-tensor/(?P<name>[^/]+)', MutableTensorApiHandler)
