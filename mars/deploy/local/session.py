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
    def __init__(self, endpoint, **kwargs):
        self._session_id = uuid.uuid4()
        self._endpoint = endpoint
        # dict structure: {tileable_key -> graph_key, tileable_ids}
        # dict value is a tuple object which records graph key and tilable id
        self._executed_tileables = dict()
        self._api = MarsAPI(self._endpoint)

        # create session on the cluster side
        self._api.create_session(self._session_id)

        if kwargs:
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError('Local cluster session got unexpected arguments: %s' % unexpected_keys)

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        self._endpoint = endpoint
        self._api = MarsAPI(self._endpoint)

    def _get_tileable_graph_key(self, tileable_key):
        return self._executed_tileables[tileable_key][0]

    def _set_tileable_graph_key(self, tileable, graph_key):
        tileable_key = tileable.key
        tileable_id = tileable.id
        if tileable_key in self._executed_tileables:
            self._executed_tileables[tileable_key][1].add(tileable_id)
        else:
            self._executed_tileables[tileable_key] = graph_key, {tileable_id}

    def _update_tileable_shape(self, tileable):
        graph_key = self._get_tileable_graph_key(tileable.key)
        new_nsplits = self._api.get_tileable_nsplits(self._session_id, graph_key, tileable.key)
        tileable._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tileable.nsplits = new_nsplits

    def create_mutable_tensor(self, name, shape, dtype, *args, **kwargs):
        return self._api.create_mutable_tensor(self._session_id, name, shape,
                                               dtype, *args, **kwargs)

    def get_mutable_tensor(self, name):
        return self._api.get_mutable_tensor(self._session_id, name)

    def send_chunk_records(self, name, chunk_records_to_send):
        return self._api.send_chunk_records(self._session_id, name, chunk_records_to_send)

    def seal(self, name):
        graph_key, tensor_key, tensor_id, tensor_meta = self._api.seal(self._session_id, name)
        self._executed_tileables[tensor_key] = graph_key, {tensor_id}
        return tensor_meta

    def run(self, *tileables, **kw):
        timeout = kw.pop('timeout', -1)
        fetch = kw.pop('fetch', True)
        compose = kw.pop('compose', True)
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        # those executed tileables should fetch data directly, submit the others
        run_tileables = [t for t in tileables if t.key not in self._executed_tileables]

        graph = build_graph(run_tileables, executed_keys=list(self._executed_tileables.keys()))
        targets = [t.key for t in run_tileables]
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

        for t in tileables:
            self._set_tileable_graph_key(t, graph_key)

        if not fetch:
            return
        else:
            return self.fetch(*tileables)

    def fetch(self, *tileables):
        futures = []
        for tileable in tileables:
            key = tileable.key

            if key not in self._executed_tileables:
                raise ValueError('Cannot fetch the unexecuted tileable')

            graph_key = self._get_tileable_graph_key(tileable.key)
            compressions = dataserializer.get_supported_compressions()
            future = self._api.fetch_data(self._session_id, graph_key, key, compressions, wait=False)
            futures.append(future)
        return [dataserializer.loads(f.result()) for f in futures]

    def decref(self, *keys):
        for tileable_key, tileable_id in keys:
            if tileable_key not in self._executed_tileables:
                continue
            graph_key, ids = self._executed_tileables[tileable_key]
            if tileable_id in ids:
                ids.remove(tileable_id)
                # for those same key tileables, do decref only when all those tileables are garbage collected
                if len(ids) != 0:
                    continue
                self._api.delete_data(self._session_id, graph_key, tileable_key)
                del self._executed_tileables[tileable_key]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._api.delete_session(self._session_id)
