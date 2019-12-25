#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import uuid
import json
import time
from numbers import Integral

from ...api import MarsAPI
from ...config import options
from ...errors import ExecutionFailed
from ...scheduler.graph import GraphState
from ...serialize import dataserializer
from ...utils import build_tileable_graph, sort_dataframe_result


class LocalClusterSession(object):
    def __init__(self, endpoint, session_id=None, **kwargs):
        self._endpoint = endpoint
        # dict structure: {tileable_key -> graph_key, tileable_ids}
        # dict value is a tuple object which records graph key and tilable id
        self._executed_tileables = dict()
        self._api = MarsAPI(self._endpoint)

        if session_id is None:
            # create session on the cluster side
            self._session_id = uuid.uuid4()
            self._api.create_session(self._session_id)
        else:
            # Get the session actor ref using given session_id
            self._session_id = session_id
            if not self._api.has_session(self._session_id):
                raise ValueError('The session with id = %s doesn\'t exist' % self._session_id)

        if kwargs:
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError('Local cluster session got unexpected arguments: %s' % unexpected_keys)

    @property
    def session_id(self):
        return self._session_id

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
        from ...tensor.utils import create_mutable_tensor
        shape, dtype, chunk_size, chunk_keys, chunk_eps = \
            self._api.create_mutable_tensor(self._session_id, name, shape,
                                            dtype, *args, **kwargs)
        return create_mutable_tensor(name, chunk_size, shape, dtype, chunk_keys, chunk_eps)

    def get_mutable_tensor(self, name):
        from ...tensor.utils import create_mutable_tensor
        shape, dtype, chunk_size, chunk_keys, chunk_eps = \
            self._api.get_mutable_tensor(self._session_id, name)
        return create_mutable_tensor(name, chunk_size, shape, dtype, chunk_keys, chunk_eps)

    def write_mutable_tensor(self, tensor, index, value):
        chunk_records_to_send = tensor._do_write(index, value)
        self._api.send_chunk_records(self._session_id, tensor.name, chunk_records_to_send)

    def seal(self, tensor):
        from ...tensor.utils import create_fetch_tensor
        chunk_records_to_send = tensor._do_flush()
        self._api.send_chunk_records(self._session_id, tensor.name, chunk_records_to_send)

        graph_key_hex, tensor_key, tensor_id, tensor_meta = self._api.seal(self._session_id, tensor.name)
        self._executed_tileables[tensor_key] = uuid.UUID(graph_key_hex), {tensor_id}

        # Construct Tensor on the fly.
        shape, dtype, chunk_size, chunk_keys, _ = tensor_meta
        return create_fetch_tensor(chunk_size, shape, dtype, tensor_key=tensor_key, chunk_keys=chunk_keys)

    def run(self, *tileables, **kw):
        timeout = kw.pop('timeout', -1)
        fetch = kw.pop('fetch', True)
        compose = kw.pop('compose', True)
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        # those executed tileables should fetch data directly, submit the others
        run_tileables = [t for t in tileables if t.key not in self._executed_tileables]

        graph = build_tileable_graph(run_tileables, set(self._executed_tileables.keys()))
        targets = [t.key for t in run_tileables]
        graph_key = uuid.uuid4()

        # submit graph to local cluster
        self._api.submit_graph(self._session_id, json.dumps(graph.to_json(), separators=(',', ':')),
                               graph_key, targets, compose=compose)

        exec_start_time = time.time()
        time_elapsed = 0
        check_interval = options.check_interval
        while timeout <= 0 or time_elapsed < timeout:
            timeout_val = min(check_interval, timeout - time_elapsed) if timeout > 0 else check_interval
            self._api.wait_graph_finish(self._session_id, graph_key, timeout=timeout_val)
            graph_state = self._api.get_graph_state(self._session_id, graph_key)
            if graph_state == GraphState.SUCCEEDED:
                break
            if graph_state == GraphState.FAILED:
                exc_info = self._api.get_graph_exc_info(self._session_id, graph_key)
                if exc_info is not None:
                    try:
                        raise exc_info[1].with_traceback(exc_info[2]) from None
                    except:  # noqa: E722
                        raise ExecutionFailed('Graph execution failed.')
                else:
                    raise ExecutionFailed('Graph execution failed with unknown reason')
            time_elapsed = time.time() - exec_start_time

        if 0 < timeout < time.time() - exec_start_time:
            raise TimeoutError

        for t in tileables:
            self._set_tileable_graph_key(t, graph_key)

        if not fetch:
            return
        else:
            return self.fetch(*tileables)

    def fetch(self, *tileables):
        from ...tensor.indexing import TensorIndex
        from ...dataframe.indexing.iloc import DataFrameIlocGetItem

        tileable_results = []
        for tileable in tileables:
            # TODO: support DataFrame getitem
            if tileable.key not in self._executed_tileables and \
                    isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem)):
                key = tileable.inputs[0].key
                indexes = tileable.op.indexes
                if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                    raise ValueError('Only support fetch data slices')
            else:
                key = tileable.key
                indexes = None
            if key not in self._executed_tileables:
                raise ValueError('Cannot fetch the unexecuted tileable')

            graph_key = self._get_tileable_graph_key(key)
            compressions = dataserializer.get_supported_compressions()
            result = self._api.fetch_data(self._session_id, graph_key, key, index_obj=indexes,
                                          compressions=compressions)
            result_data = dataserializer.loads(result)
            tileable_results.append(sort_dataframe_result(tileable, result_data))
        return tileable_results

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
                self.delete_data(tileable_key)

    def delete_data(self, tileable_key, wait=False):
        if tileable_key not in self._executed_tileables:
            return
        graph_key, _ids = self._executed_tileables[tileable_key]
        self._api.delete_data(self._session_id, graph_key, tileable_key, wait=wait)
        del self._executed_tileables[tileable_key]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for key in list(self._executed_tileables.keys()):
            self.delete_data(key, wait=True)
        self._api.delete_session(self._session_id)
