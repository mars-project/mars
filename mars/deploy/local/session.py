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

import uuid
import json
import time

from ...api import MarsAPI
from ...graph import DirectedGraph
from ...scheduler.graph import GraphState
from ...serialize import dataserializer


class LocalClusterSession(object):
    def __init__(self, endpoint):
        self._session_id = uuid.uuid4()
        self._endpoint = endpoint
        self._tensor_to_graph = dict()
        self._api = MarsAPI(self._endpoint)

        # create session on the cluster side
        self._api.create_session(self._session_id)

    def run(self, *tensors, **kw):
        timeout = kw.pop('timeout', -1)
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        graph = DirectedGraph()
        targets = [t.key for t in tensors]
        graph_key = uuid.uuid4()
        for t in tensors:
            t.build_graph(graph, tiled=False)
            self._tensor_to_graph[t.key] = graph_key

        # submit graph to local cluster
        self._api.submit_graph(self._session_id, json.dumps(graph.to_json()),
                               graph_key, targets)

        exec_start_time = time.time()
        while timeout <= 0 or time.time() - exec_start_time <= timeout:
            time.sleep(0.1)

            graph_state = self._api.get_graph_state(self._session_id, graph_key)
            if graph_state == GraphState.SUCCEEDED:
                break
            if graph_state == GraphState.FAILED:
                # TODO(qin): add traceback
                raise SystemError('Graph execution failed with unknown reason')

        if 0 < timeout < time.time() - exec_start_time:
            raise TimeoutError

        data_list = []
        for target in targets:
            resp = self._api.fetch_data(self._session_id, graph_key, target)
            data_list.append(dataserializer.loads(resp))

        return data_list

    def decref(self, *keys):
        for k in keys:
            if k not in self._tensor_to_graph:
                continue
            graph_key = self._tensor_to_graph[k]
            self._api.delete_data(self._session_id, graph_key, k)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._api.delete_session(self._session_id)
