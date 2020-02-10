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

import sys
import base64
import json
import logging
import pickle
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from tornado import gen, web

from ..tensor.core import Indexes
from ..errors import GraphNotExists
from ..lib.tblib import pickling_support
from ..serialize.dataserializer import CompressType
from ..utils import to_str, tokenize, numpy_dtype_from_descr_json
from .server import MarsRequestHandler, register_web_handler

pickling_support.install()

logger = logging.getLogger(__name__)


class MarsApiRequestHandler(MarsRequestHandler):
    def set_default_headers(self):
        super().set_default_headers()
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
    async def post(self):
        args = {k: self.get_argument(k) for k in self.request.arguments}

        try:
            python_version = tuple(int(p) for p in args.pop('pyver').split('.')) \
                if 'pyver' in args else sys.version_info
        except ValueError as ex:
            raise web.HTTPError(400, reason='Invalid version data: %s' % ex)
        if python_version[0] != sys.version_info[0]:
            raise web.HTTPError(400, reason='Python version not consistent')

        session_id = tokenize(str(uuid.uuid1()))
        await self.web_api.create_session(session_id, **args)
        self.write(json.dumps(dict(session_id=session_id)))


class SessionApiHandler(MarsApiRequestHandler):
    async def head(self, session_id):
        if not await self.web_api.has_session(session_id):
            raise web.HTTPError(404, 'Session doesn\'t not exists')

    async def delete(self, session_id):
        await self.web_api.delete_session(session_id)


class GraphsApiHandler(MarsApiRequestHandler):
    async def get(self, session_id):
        try:
            graph_states = await self.web_api.get_tasks_info(session_id)
            tasks_dict = graph_states[session_id]['tasks']
            self.write(json.dumps(tasks_dict))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    async def post(self, session_id):
        try:
            graph = self.get_argument('graph')
            target = self.get_argument('target').split(',')
            names = self.get_argument('names').split(',')
            compose = bool(int(self.get_argument('compose', '1')))
        except web.MissingArgumentError as ex:
            self.write(json.dumps(dict(msg=str(ex))))
            raise web.HTTPError(400, reason='Argument missing')

        try:
            graph_key = tokenize(str(uuid.uuid4()))
            await self.web_api.submit_graph(session_id, graph, graph_key, target, names=names, compose=compose)
            self.write(json.dumps(dict(graph_key=graph_key)))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())


class GraphApiHandler(MarsApiRequestHandler):
    _executor = ThreadPoolExecutor(1)

    async def get(self, session_id, graph_key):
        from ..scheduler.utils import GraphState

        wait_timeout = int(self.get_argument('wait_timeout', None))

        try:
            await self.web_api.wait_graph_finish(session_id, graph_key, wait_timeout)
            state = await self.web_api.get_graph_state(session_id, graph_key)
        except GraphNotExists:
            raise web.HTTPError(404, 'Graph not exists')

        resp = dict(state=state.value)
        if state == GraphState.FAILED:
            exc_info = self.web_api.get_graph_exc_info(session_id, graph_key)
            if exc_info is not None:
                resp['exc_info'] = base64.b64encode(pickle.dumps(exc_info)).decode('ascii')
                resp['exc_info_text'] = ''.join(traceback.format_exception(*exc_info))
        self.write(json.dumps(resp))

    async def delete(self, session_id, graph_key):
        try:
            await self.web_api.stop_graph(session_id, graph_key)
        except:  # noqa: E722
            self._dump_exception(sys.exc_info(), 404)


class GraphDataApiHandler(MarsApiRequestHandler):
    _executor = ThreadPoolExecutor(1)

    async def get(self, session_id, graph_key, tileable_key):
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
                nsplits = await self.web_api.get_tileable_nsplits(session_id, graph_key, tileable_key)
                self.write(json.dumps(nsplits))
            else:
                raise web.HTTPError(403, 'Unknown data type requests')
        else:
            data = await self.web_api.fetch_data(session_id, graph_key, tileable_key, index_obj=slices_arg,
                                                 compressions=compressions_arg)
            self.write(data)

    async def delete(self, session_id, graph_key, tileable_key):
        wait = int(self.get_argument('wait', '0'))
        await self.web_api.delete_data(session_id, graph_key, tileable_key, wait=wait)


class WorkersApiHandler(MarsApiRequestHandler):
    async def get(self):
        action = self.get_argument('action', None)
        if action == 'count':
            self.write(json.dumps(await self.web_api.count_workers()))
        else:
            self.write(json.dumps(await self.web_api.get_workers_meta()))


class MutableTensorApiHandler(MarsApiRequestHandler):
    async def get(self, session_id, name):
        try:
            meta = await self.web_api.get_mutable_tensor(session_id, name)
            self.write(json.dumps(meta))
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    async def post(self, session_id, name):
        try:
            action = self.get_argument('action', None)
            if action == 'create':
                req_json = json.loads(self.request.body.decode('ascii'))
                shape = req_json['shape']
                dtype = numpy_dtype_from_descr_json(req_json['dtype'])
                fill_value = req_json['fill_value']
                chunk_size = req_json['chunk_size']
                meta = await self.web_api.create_mutable_tensor(
                    session_id, name, shape, dtype, fill_value=fill_value, chunk_size=chunk_size)
                self.write(json.dumps(meta))
            elif action == 'seal':
                info = await self.web_api.seal(session_id, name)
                self.write(json.dumps(info))
            else:
                raise web.HTTPError(400, reason='Invalid argument')
        except:  # noqa: E722
            self._dump_exception(sys.exc_info())

    async def put(self, session_id, name):
        try:
            payload_type = self.get_argument('payload_type', None)
            body = self.request.body
            await self.web_api.write_mutable_tensor(session_id, name, payload_type, body)
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
