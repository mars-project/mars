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

import os
import json
import threading
import time
import uuid
from numbers import Integral

import numpy as np

from .core import Entity, Base
from .context import get_context, LocalContext
from .operands import Fetch
from .tiles import get_tiled
from .executor import Executor
from .config import options
from .utils import classproperty, calc_nsplits
try:
    from .resource import cpu_count, cuda_count
except ImportError:  # pragma: no cover
    from multiprocessing import cpu_count
    cuda_count = None


class LocalSession(object):
    def __init__(self, **kwargs):
        self._endpoint = None
        self._session_id = uuid.uuid4()
        self._context = LocalContext(self)
        self._executor = Executor(storage=self._context)

        self._mut_tensor = dict()
        self._mut_tensor_data = dict()

        if kwargs:
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError('Local session got unexpected arguments: %s' % unexpected_keys)

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        if endpoint is not None:
            raise ValueError('Local session cannot set endpoint')
        self._endpoint = endpoint

    @property
    def session_id(self):
        return self._session_id

    @property
    def executor(self):
        return self._executor

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, new_context):
        self._context = new_context
        self._executor.storage = new_context

    @property
    def executed_tileables(self):
        return self._executor.stored_tileables.keys()

    def run(self, *tileables, **kw):
        with self.context:
            if self._executor is None:
                raise RuntimeError('Session has closed')
            dest_gpu = all(tileable.op.gpu for tileable in tileables)
            if dest_gpu:
                self._executor._engine = 'cupy'
            else:
                self._executor._engine = None
            if 'n_parallel' not in kw:
                if dest_gpu:
                    # GPU
                    cnt = cuda_count() if cuda_count is not None else 0
                    if cnt == 0:
                        raise RuntimeError('No GPU found for execution')
                    kw['n_parallel'] = cnt
                else:
                    # CPU
                    kw['n_parallel'] = cpu_count()
            # set number of running cores
            self.context.set_ncores(kw['n_parallel'])
            res = self._executor.execute_tileables(tileables, **kw)
            return res

    def _update_tileable_shape(self, tileable):
        from .optimizes.tileable_graph import tileable_optimized

        new_nsplits = self._executor.get_tileable_nsplits(tileable)
        tiled = get_tiled(tileable, mapping=tileable_optimized)
        for t in (tileable, tiled):
            t._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tiled.nsplits = new_nsplits

    def fetch(self, *tileables, n_parallel: int = None, **kw):
        if self._executor is None:
            raise RuntimeError('Session has closed')
        if n_parallel is None:
            kw['n_parallel'] = cpu_count()
        return self._executor.fetch_tileables(tileables, **kw)

    def create_mutable_tensor(self, name, shape, dtype, fill_value=None, *args, **kwargs):
        from .tensor.core import MutableTensor, MutableTensorData
        if name in self._mut_tensor:
            raise ValueError("The mutable tensor named '%s' already exists." % name)
        mut_tensor = MutableTensor(data=MutableTensorData(name=name, op=None, shape=shape, dtype=dtype))
        self._mut_tensor[name] = mut_tensor
        if fill_value is None:
            self._mut_tensor_data[name] = np.zeros(shape, dtype=dtype)
        else:
            self._mut_tensor_data[name] = np.full(shape, fill_value, dtype=dtype)
        return mut_tensor

    def get_mutable_tensor(self, name):
        if name not in self._mut_tensor:
            raise ValueError("The mutable tensor named '%s' doesn't exist, or has already been sealed." % name)
        return self._mut_tensor.get(name)

    def write_mutable_tensor(self, tensor, index, value):
        if tensor.name not in self._mut_tensor:
            raise ValueError("The mutable tensor named '%s' doesn't exist, or has already been sealed." % tensor.name)
        tensor_data = self._mut_tensor_data.get(tensor.name)
        tensor_data[index] = value

    def seal(self, tensor):
        from .tensor.datasource.array import ArrayDataSource
        if tensor.name not in self._mut_tensor:
            raise ValueError("The mutable tensor named '%s' doesn't exist, or has already been sealed." % tensor.name)
        mut_tensor = self._mut_tensor.pop(tensor.name)
        tensor_data = self._mut_tensor_data.pop(tensor.name)
        sealed_tensor = ArrayDataSource(tensor_data, dtype=mut_tensor.dtype)(shape=mut_tensor.shape)
        self._executor.execute_tileables([sealed_tensor])
        return sealed_tensor

    def get_named_tileable_infos(self, name):
        return self._context.get_named_tileable_infos(name)

    def decref(self, *keys):
        self._executor.decref(*keys)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._executor = None


class ClusterSession(object):
    def __init__(self, endpoint, session_id=None, **kwargs):
        from .api import MarsAPI
        from .context import DistributedContext

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

        self._context = DistributedContext(endpoint, self._session_id)

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
        from .api import MarsAPI

        self._endpoint = endpoint
        self._api = MarsAPI(self._endpoint)

    @property
    def context(self):
        return self._context

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
        from .tensor.utils import create_mutable_tensor
        shape, dtype, chunk_size, chunk_keys, chunk_eps = \
            self._api.create_mutable_tensor(self._session_id, name, shape,
                                            dtype, *args, **kwargs)
        return create_mutable_tensor(name, chunk_size, shape, dtype, chunk_keys, chunk_eps)

    def get_mutable_tensor(self, name):
        from .tensor.utils import create_mutable_tensor
        shape, dtype, chunk_size, chunk_keys, chunk_eps = \
            self._api.get_mutable_tensor(self._session_id, name)
        return create_mutable_tensor(name, chunk_size, shape, dtype, chunk_keys, chunk_eps)

    def write_mutable_tensor(self, tensor, index, value):
        chunk_records_to_send = tensor._do_write(index, value)
        self._api.send_chunk_records(self._session_id, tensor.name, chunk_records_to_send)

    def get_workers_meta(self):
        return self._api.get_workers_meta()

    def seal(self, tensor):
        from .tensor.utils import create_fetch_tensor
        chunk_records_to_send = tensor._do_flush()
        self._api.send_chunk_records(self._session_id, tensor.name, chunk_records_to_send)

        graph_key_hex, tensor_key, tensor_id, tensor_meta = self._api.seal(self._session_id, tensor.name)
        self._executed_tileables[tensor_key] = uuid.UUID(graph_key_hex), {tensor_id}

        # Construct Tensor on the fly.
        shape, dtype, chunk_size, chunk_keys, _ = tensor_meta
        return create_fetch_tensor(chunk_size, shape, dtype, tensor_key=tensor_key, chunk_keys=chunk_keys)

    def run(self, *tileables, **kw):
        from .utils import build_tileable_graph
        from .scheduler.graph import GraphState
        from .errors import ExecutionFailed

        timeout = kw.pop('timeout', -1)
        fetch = kw.pop('fetch', True)
        compose = kw.pop('compose', True)
        name = kw.pop('name', None)
        if not isinstance(name, (tuple, list)):
            name = [name]
        if kw:
            raise TypeError('run got unexpected key arguments {0}'.format(', '.join(kw.keys())))

        # those executed tileables should fetch data directly, submit the others
        run_tileables = [t for t in tileables if t.key not in self._executed_tileables]

        graph = build_tileable_graph(run_tileables, set(self._executed_tileables.keys()))
        targets = [t.key for t in run_tileables]
        graph_key = uuid.uuid4()

        # submit graph to local cluster
        self._api.submit_graph(self._session_id, json.dumps(graph.to_json(), separators=(',', ':')),
                               graph_key, targets, compose=compose, names=name)

        ctx = get_context()
        yield_info = None
        if ctx is not None:
            yield_info = ctx.yield_execution_pool()

        try:
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
                        exc = exc_info[1].with_traceback(exc_info[2])
                        raise ExecutionFailed('Graph execution failed.') from exc
                    else:
                        raise ExecutionFailed('Graph execution failed with unknown reason')
                time_elapsed = time.time() - exec_start_time

            if 0 < timeout < time.time() - exec_start_time:
                raise TimeoutError
        finally:
            if ctx is not None:
                ctx.acquire_execution_pool(yield_info)

        for t in tileables:
            self._set_tileable_graph_key(t, graph_key)

        if not fetch:
            return
        else:
            return self.fetch(*tileables)

    def _is_executed(self, tileable):
        # if tileble.key in executed tileables
        # or it's a fetch already
        return tileable.key in self._executed_tileables or \
               isinstance(tileable.op, Fetch)

    def fetch(self, *tileables):
        from .utils import sort_dataframe_result
        from .serialize import dataserializer
        from .tensor.indexing import TensorIndex
        from .dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        tileable_results = []
        for tileable in tileables:
            # TODO: support DataFrame getitem
            if not self._is_executed(tileable) and \
                    isinstance(tileable.op, (TensorIndex, DataFrameIlocGetItem, SeriesIlocGetItem)):
                to_fetch_tileable = tileable.inputs[0]

                indexes = tileable.op.indexes
                if not all(isinstance(ind, (slice, Integral)) for ind in indexes):
                    raise ValueError('Only support fetch data slices')
            else:
                to_fetch_tileable = tileable
                indexes = None
            if not self._is_executed(to_fetch_tileable):
                raise ValueError('Cannot fetch the unexecuted tileable')

            key = to_fetch_tileable.key
            compressions = dataserializer.get_supported_compressions()
            if getattr(to_fetch_tileable, 'chunks', None) is not None:
                # fetch which owns fetch chunks inside remote
                chunk_indexes = [c.index for c in to_fetch_tileable.chunks]
                chunk_keys = [c.key for c in to_fetch_tileable.chunks]
                nsplits = to_fetch_tileable.nsplits
                if any(np.isnan(np.sum(ns)) for ns in nsplits):
                    # unknown chunk shape exists
                    # try to calculate nsplits
                    chunk_metas = self._api.get_chunk_metas(self._session_id, chunk_keys)
                    chunk_idx_to_shape = {idx: cm.chunk_shape for idx, cm in
                                          zip(chunk_indexes, chunk_metas)}
                    nsplits = calc_nsplits(chunk_idx_to_shape)
                result = self._api.fetch_chunks_data(
                    self._session_id, chunk_indexes, chunk_keys, nsplits,
                    serial=False, index_obj=indexes, compressions=compressions)
            else:
                graph_key = self._get_tileable_graph_key(key)
                result = self._api.fetch_data(self._session_id, graph_key, key, serial=False,
                                              index_obj=indexes, compressions=compressions)
            tileable_results.append(sort_dataframe_result(tileable, result))
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

    def get_named_tileable_infos(self, name):
        return self._context.get_named_tileable_infos(name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for key in list(self._executed_tileables.keys()):
            self.delete_data(key, wait=True)
        self._api.delete_session(self._session_id)


class Session(object):
    _default_session_local = threading.local()

    def __init__(self, endpoint=None, **kwargs):
        self._endpoint = endpoint
        self._kws = kwargs
        self._init()

    def _init(self):
        endpoint, kwargs = self._endpoint, self._kws
        if endpoint is not None:
            if 'http' in endpoint:
                # connect to web
                from .web.session import Session as WebSession

                self._sess = WebSession(endpoint, **kwargs)
            else:
                # connect to local cluster

                self._sess = ClusterSession(endpoint, **kwargs)
        else:
            try:
                endpoint = os.environ['MARS_SCHEDULER_ADDRESS']
                session_id = os.environ.get('MARS_SESSION_ID', None)
                self._sess = ClusterSession(endpoint, session_id=session_id)
            except KeyError:
                self._sess = LocalSession(**kwargs)

    def __getstate__(self):
        return self._endpoint, self._kws, self.session_id

    def __setstate__(self, state):
        self._endpoint, self._kws, session_id = state
        self._init()
        self._sess._session_id = session_id

    def run(self, *tileables, **kw):
        from . import tensor as mt

        fetch = kw.get('fetch', True)
        ret_list = False
        if len(tileables) == 1 and isinstance(tileables[0], (tuple, list)):
            ret_list = True
            tileables = tileables[0]
        elif len(tileables) > 1:
            ret_list = True

        tileables = tuple(mt.tensor(t) if not isinstance(t, (Entity, Base)) else t
                          for t in tileables)
        result = self._sess.run(*tileables, **kw)

        for t in tileables:
            t._attach_session(self)

        for t in tileables:
            if getattr(t, 'shape', None) is not None and \
                    any(np.isnan(s) for s in t.shape):
                self._sess._update_tileable_shape(t)

        if fetch:
            ret = []
            for r, t in zip(result, tileables):
                if hasattr(t, 'isscalar') and t.isscalar() and getattr(r, 'size', None) == 1:
                    ret.append(r.item())
                else:
                    ret.append(r)
            if ret_list:
                return ret
            return ret[0]

    def fetch(self, *tileables, **kw):
        ret_list = False
        if len(tileables) == 1 and isinstance(tileables[0], (tuple, list)):
            ret_list = True
            tileables = tileables[0]
        elif len(tileables) > 1:
            ret_list = True

        result = self._sess.fetch(*tileables, **kw)

        ret = []
        for r, t in zip(result, tileables):
            if hasattr(t, 'isscalar') and t.isscalar() and hasattr(r, 'item'):
                ret.append(r.item())
            else:
                ret.append(r)
        if ret_list:
            return ret
        return ret[0]

    @property
    def endpoint(self):
        return self._sess.endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        self._sess.endpoint = endpoint

    @property
    def session_id(self):
        return self._sess.session_id

    def decref(self, *keys):
        if hasattr(self._sess, 'decref'):
            self._sess.decref(*keys)

    def __getattr__(self, attr):
        obj = self._sess.__getattribute__(attr)
        return obj

    def __enter__(self):
        self._sess.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self.__exit__(None, None, None)

    def as_default(self):
        Session._default_session_local.default_session = self
        return self

    @classmethod
    def _set_default_session(cls, session):
        cls._default_session_local.default_session = session
        return session

    @classproperty
    def default(self):
        return getattr(Session._default_session_local, 'default_session', None)

    @classmethod
    def default_or_local(cls):
        default_session = getattr(Session._default_session_local, 'default_session', None)
        if default_session is not None:
            return default_session

        return cls._set_default_session(Session())

    def create_mutable_tensor(self, name, shape, dtype, fill_value=None, *args, **kwargs):
        return self._sess.create_mutable_tensor(name, shape, dtype, fill_value=fill_value, *args, **kwargs)

    def get_mutable_tensor(self, name):
        return self._sess.get_mutable_tensor(name)

    def write_mutable_tensor(self, tensor, index, value):
        self._sess.write_mutable_tensor(tensor, index, value)

    def seal(self, tensor):
        return self._sess.seal(tensor)


def new_session(scheduler=None, **kwargs):
    return Session(scheduler, **kwargs)
