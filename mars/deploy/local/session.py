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
from ...compat import TimeoutError  # pylint: disable=W0622
from ...scheduler.graph import GraphState
from ...serialize import dataserializer
from ...errors import ExecutionFailed
from ...utils import build_graph


class LocalClusterSession(object):
    def __init__(self, endpoint):
        self._session_id = uuid.uuid4()
        self._endpoint = endpoint
        # dict structure: {tensor_key -> graph_key, tensor_ids}
        # dict value is a tuple object which records graph key and tensor id
        self._executed_tensors = dict()
        self._api = MarsAPI(self._endpoint)

        # create session on the cluster side
        self._api.create_session(self._session_id)

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        self._endpoint = endpoint
        self._api = MarsAPI(self._endpoint)

    def _get_tensor_graph_key(self, tensor_key):
        return self._executed_tensors[tensor_key][0]

    def _set_tensor_graph_key(self, tensor, graph_key):
        tensor_key = tensor.key
        tensor_id = tensor.id
        if tensor_key in self._executed_tensors:
            self._executed_tensors[tensor_key][1].add(tensor_id)
        else:
            self._executed_tensors[tensor_key] = graph_key, {tensor_id}

    def _update_tensor_shape(self, tensor):
        graph_key = self._get_tensor_graph_key(tensor.key)
        new_nsplits = self._api.get_tensor_nsplits(self._session_id, graph_key, tensor.key)
        tensor._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tensor.nsplits = new_nsplits

    def run(self, *tensors, **kw):
        timeout = kw.pop('timeout', -1)
        fetch = kw.pop('fetch', True)
        compose = kw.pop('compose', True)
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        # those executed tensors should fetch data directly, submit the others
        run_tensors = [t for t in tensors if t.key not in self._executed_tensors]

        graph = build_graph(run_tensors, executed_keys=list(self._executed_tensors.keys()))
        targets = [t.key for t in run_tensors]
        graph_key = uuid.uuid4()

        # submit graph to local cluster
        self._api.submit_graph(self._session_id, json.dumps(graph.to_json()),
                               graph_key, targets, compose=compose)

        exec_start_time = time.time()
        while timeout <= 0 or time.time() - exec_start_time <= timeout:
            time.sleep(0.1)

            graph_state = self._api.get_graph_state(self._session_id, graph_key)
            if graph_state == GraphState.SUCCEEDED:
                break
            if graph_state == GraphState.FAILED:
                # TODO(qin): add traceback
                raise ExecutionFailed('Graph execution failed with unknown reason')

        if 0 < timeout < time.time() - exec_start_time:
            raise TimeoutError

        for t in tensors:
            self._set_tensor_graph_key(t, graph_key)

        if not fetch:
            return
        else:
            return self.fetch(*tensors)

    def fetch(self, *tensors):
        futures = []
        for tensor in tensors:
            key = tensor.key

            if key not in self._executed_tensors:
                raise ValueError('Cannot fetch the unexecuted tensor')

            graph_key = self._get_tensor_graph_key(tensor.key)
            future = self._api.fetch_data(self._session_id, graph_key, key, wait=False)
            futures.append(future)
        return [dataserializer.loads(f.result()) for f in futures]

    def decref(self, *keys):
        for tensor_key, tensor_id in keys:
            if tensor_key not in self._executed_tensors:
                continue
            graph_key, ids = self._executed_tensors[tensor_key]
            if tensor_id in ids:
                ids.remove(tensor_id)
                # for those same key tensors, do decref only when all those tensors are garbage collected
                if len(ids) != 0:
                    continue
                self._api.delete_data(self._session_id, graph_key, tensor_key)
                del self._executed_tensors[tensor_key]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._api.delete_session(self._session_id)
