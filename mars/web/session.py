#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import base64
import json
import time
import logging

import requests

from ..compat import six, TimeoutError
from ..serialize import dataserializer
from ..errors import ExecutionInterrupted
from ..graph import DirectedGraph

logger = logging.getLogger(__name__)


class Session(object):
    def __init__(self, endpoint, args=None):
        self._endpoint = endpoint
        self._args = args
        self._tensor_to_graph = dict()

        self._req_session = requests.Session()

        from requests.adapters import HTTPAdapter
        self._req_session.mount('http://stackoverflow.com', HTTPAdapter(max_retries=5))
        self._main()

    @property
    def session_id(self):
        return self._session_id

    def _main(self):
        resp = self._req_session.post(self._endpoint + '/api/session', self._args)
        if resp.status_code >= 400:
            raise SystemError('Failed to create mars session.')
        content = json.loads(resp.text)
        self._session_id = content['session_id']

    def run(self, *tensors, **kw):
        timeout = kw.pop('timeout', -1)
        compose = kw.pop('compose', True)
        wait = kw.pop('wait', True)
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        graph = DirectedGraph()
        for t in tensors:
            graph = t.build_graph(graph=graph, tiled=False, compose=compose)
        targets = [t.key for t in tensors]

        targets_join = ','.join(targets)
        session_url = self._endpoint + '/api/session/' + self._session_id
        graph_json = graph.to_json()

        resp_json = self._submit_graph(graph_json, targets_join)
        graph_key = resp_json['graph_key']
        graph_url = session_url + '/graph/' + graph_key

        for t in tensors:
            self._tensor_to_graph[t.key] = graph_key

        if wait:
            exec_start_time = time.time()
            while timeout <= 0 or time.time() - exec_start_time <= timeout:
                try:
                    time.sleep(1)
                    try:
                        resp = self._req_session.get(graph_url)
                    except requests.ConnectionError as ex:
                        err_msg = str(ex)
                        if 'ConnectionResetError' in err_msg or 'Connection refused' in err_msg:
                            continue
                        raise
                    if resp.status_code == 504:
                        logging.debug('Gateway Time-out, try again')
                        continue
                    if resp.status_code >= 400:
                        raise SystemError('Failed to read task status. Code: %d, Reason: %s, Content:\n%s' %
                                          (resp.status_code, resp.reason, resp.text))
                    resp_json = json.loads(resp.text)
                    if resp_json['state'] in ('running', 'preparing'):
                        continue
                    elif resp_json['state'] == 'success':
                        break
                    elif resp_json['state'] == ('cancelled', 'cancelling'):
                        raise ExecutionInterrupted
                    elif resp_json['state'] == 'failed':
                        # TODO add traceback
                        if 'traceback' in resp_json:
                            traceback = resp_json['traceback']
                            if isinstance(traceback, list):
                                traceback = ''.join(str(s) for s in traceback)
                            raise SystemError('Graph execution failed.\nMessage: %s\nTraceback from server:\n%s' %
                                              (resp_json['msg'], traceback))
                        else:
                            raise SystemError('Graph execution failed with unknown reason.')
                    else:
                        raise SystemError('Unknown graph execution state %s' % resp_json['state'])
                except KeyboardInterrupt:
                    resp = self._req_session.delete(graph_url)
                    if resp.status_code >= 400:
                        raise SystemError('Failed to stop graph execution. Code: %d, Reason: %s, Content:\n%s' %
                                          (resp.status_code, resp.reason, resp.text))
            if 0 < timeout < time.time() - exec_start_time:
                raise TimeoutError
            data_list = []
            for tk in targets:
                resp = self._req_session.get(session_url + '/graph/' + graph_key + '/data/' + tk)
                if resp.status_code >= 400:
                    continue
                data_list.append(dataserializer.loads(resp.content))
            return data_list
        else:
            return graph_key

    def decref(self, *keys):
        session_url = self._endpoint + '/api/session/' + self._session_id
        for k in keys:
            if k not in self._tensor_to_graph:
                continue
            data_url = session_url + '/graph/%s/data/%s' % (self._tensor_to_graph[k], k)
            self._req_session.delete(data_url)

    def stop(self, graph_key):
        session_url = self._endpoint + '/api/session/' + self._session_id
        graph_url = session_url + '/graph/' + graph_key
        resp = self._req_session.delete(graph_url)
        if resp.status_code >= 400:
            raise SystemError('Failed to stop graph execution. Code: %d, Reason: %s, Content:\n%s' %
                              (resp.status_code, resp.reason, resp.text))

    def _submit_graph(self, graph_json, targets):
        session_url = self._endpoint + '/api/session/' + self._session_id
        resp = self._req_session.post(session_url + '/graph', dict(
            graph=json.dumps(graph_json),
            target=targets,
        ))
        if resp.status_code >= 400:
            resp_json = json.loads(resp.text)
            exc_info = base64.b64decode(resp_json['exc_info'])
            six.reraise(*exc_info)
        resp_json = json.loads(resp.text)
        return resp_json

    def close(self):
        self.decref(*list(self._tensor_to_graph.keys()))

        resp = self._req_session.delete(self._endpoint + '/api/session/' + self._session_id)
        if resp.status_code >= 400:
            raise SystemError('Failed to close mars session.')

    def check_service_ready(self, timeout=1):
        try:
            resp = self._req_session.get(self._endpoint + '/api', timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            return False
        if resp.status_code >= 400:
            return False
        return True

    def count_workers(self):
        resp = self._req_session.get(self._endpoint + '/api/worker', timeout=1)
        return json.loads(resp.text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
