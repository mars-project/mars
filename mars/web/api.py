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
from ..scheduler import SessionActor, GraphActor, GraphMetaActor, ResourceActor
from ..scheduler.session import SessionManagerActor
from ..actors import new_client
from .. import resource

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
    def initialize(self, sessions, cluster_info):
        self.sessions = sessions
        self.cluster_info = cluster_info

        uid = SessionManagerActor.default_name()
        scheduler_address = self.cluster_info.get_scheduler(uid)
        self.session_manager_ref = _actor_client.actor_ref(uid, address=scheduler_address)

    def get_session_ref(self, session_id):
        try:
            return self.sessions[session_id]
        except KeyError:
            uid = SessionActor.gen_name(session_id)
            scheduler_ip = self.cluster_info.get_scheduler(uid)
            actor_ref = _actor_client.actor_ref(uid, address=scheduler_ip)
            self.sessions[session_id] = actor_ref
            return actor_ref

    def get_graph_ref(self, session_id, graph_key):
        uid = GraphActor.gen_name(session_id, graph_key)
        scheduler_ip = self.cluster_info.get_scheduler(uid)
        actor_ref = _actor_client.actor_ref(uid, address=scheduler_ip)
        return actor_ref

    def get_resource_ref(self):
        uid = ResourceActor.default_name()
        scheduler_ip = self.cluster_info.get_scheduler(uid)
        resource_ref = _actor_client.actor_ref(uid, address=scheduler_ip)
        return resource_ref

    def get_graph_meta_ref(self, session_id, graph_key):
        uid = GraphMetaActor.gen_name(session_id, graph_key)
        scheduler_ip = self.cluster_info.get_scheduler(uid)
        actor_ref = _actor_client.actor_ref(uid, address=scheduler_ip)
        return actor_ref


class ApiEntryHandler(ApiRequestHandler):
    def get(self):
        self.write(dict(msg='Mars API Entry'))


class SessionsApiHandler(ApiRequestHandler):
    def post(self):
        args = {k: self.get_argument(k) for k in self.request.arguments}
        session_id = str(uuid.uuid1())

        session_ref = self.session_manager_ref.create_session(session_id, **args)
        session_ref = _actor_client.actor_ref(session_ref)
        self.sessions[session_id] = session_ref
        logger.info('Session %s created.' % session_id)

        self.write(json.dumps(dict(session_id=session_id)))


class SessionApiHandler(ApiRequestHandler):
    def delete(self, session_id):
        session_ref = self.get_session_ref(session_id)
        session_ref.destroy()
        try:
            del self.sessions[session_id]
        except KeyError:
            pass


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
            session_ref = self.get_session_ref(session_id)
            session_ref.submit_tensor_graph(graph, graph_key, target, _tell=True)
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

        state_obj = self.get_graph_meta_ref(session_id, graph_key).get_state()
        state = state_obj.value if state_obj else 'preparing'
        state = GraphState(state.lower())

        if state == GraphState.RUNNING:
            self.write(json.dumps(dict(state='running')))
        elif state == GraphState.SUCCEEDED:
            self.write(json.dumps(dict(state='success')))
        elif state == GraphState.FAILED:
            self.write(json.dumps(dict(state='failed')))
        elif state == GraphState.CANCELLED:
            self.write(json.dumps(dict(state='cancelled')))
        elif state == GraphState.PREPARING:
            self.write(json.dumps(dict(state='preparing')))

    def delete(self, session_id, graph_key):
        from ..scheduler.utils import GraphState
        graph_ref = self.get_graph_ref(session_id, graph_key)
        self.get_graph_meta_ref(session_id, graph_key).set_state(GraphState.CANCELLING)
        graph_ref.stop_graph()


class GraphDataHandler(ApiRequestHandler):
    _executor = futures.ThreadPoolExecutor(resource.cpu_count())

    @gen.coroutine
    def get(self, session_id, graph_key, tensor_key):
        from ..scheduler.graph import ResultReceiverActor
        uid = GraphActor.gen_name(session_id, graph_key)
        scheduler_ip = self.cluster_info.get_scheduler(uid)

        def _fetch_fun():
            client = new_client()
            merge_ref = client.create_actor(ResultReceiverActor, address=scheduler_ip)
            return merge_ref.fetch_tensor(session_id, graph_key, tensor_key)

        data = yield self._executor.submit(_fetch_fun)
        self.write(data)

    def delete(self, session_id, graph_key, tensor_key):
        uid = GraphActor.gen_name(session_id, graph_key)
        scheduler_ip = self.cluster_info.get_scheduler(uid)
        graph_ref = _actor_client.actor_ref(uid, address=scheduler_ip)
        graph_ref.free_tensor_data(tensor_key, _tell=True)


class WorkersApiHandler(ApiRequestHandler):
    def get(self):
        try:
            worker_count = self.get_resource_ref().get_worker_count()
            self.write(json.dumps(worker_count))
        except KeyError:
            self.write(json.dumps(0))


register_api_handler('/api', ApiEntryHandler)
register_api_handler('/api/session', SessionsApiHandler)
register_api_handler('/api/worker', WorkersApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)', SessionApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph', GraphsApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)', GraphApiHandler)
register_api_handler('/api/session/(?P<session_id>[^/]+)/graph/(?P<graph_key>[^/]+)/data/(?P<tensor_key>[^/]+)',
                     GraphDataHandler)
