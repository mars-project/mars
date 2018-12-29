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

import numpy as np

try:
    from .resource import cpu_count
except ImportError:
    from multiprocessing import cpu_count


class LocalSession(object):
    def __init__(self):
        from .tensor.execution.core import Executor

        self._executor = Executor()
        self._endpoint = None

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        if endpoint is not None:
            raise ValueError('Local session cannot set endpoint')
        self._endpoint = endpoint

    def run(self, *tensors, **kw):
        if self._executor is None:
            raise RuntimeError('Session has closed')
        if 'n_parallel' not in kw:
            kw['n_parallel'] = cpu_count()
        return self._executor.execute_tensors(tensors, **kw)

    def fetch(self, tensor):
        from .tensor.expressions.datasource import TensorFetchChunk

        if self._executor is None:
            raise RuntimeError('Session has closed')

        if len(tensor.chunks) == 1:
            return self._executor.chunk_result[tensor.chunks[0].key]

        chunks = []
        for c in tensor.chunks:
            op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key)
            chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key)
            chunks.append(chunk)

        new_op = tensor.op.copy()
        tensor = new_op.new_tensor([None], tensor.shape, chunks=chunks,
                                   nsplits=tensor.nsplits)

        return self._executor.execute_tensor(tensor, concat=True)[0]

    def decref(self, *keys):
        self._executor.decref(*keys)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._executor = None


class Session(object):
    _default_session = None

    def __init__(self, endpoint=None):
        if endpoint is not None:
            if 'http' in endpoint:
                # connect to web
                from .web.session import Session as WebSession

                self._sess = WebSession(endpoint)
            else:
                # connect to local cluster
                from .deploy.local.session import LocalClusterSession

                self._sess = LocalClusterSession(endpoint)
        else:
            self._sess = LocalSession()

        self._executed_keys = set()

    def run(self, *tensors, **kw):
        from . import tensor as mt

        ret_list = False
        if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
            ret_list = True
            tensors = tensors[0]
        elif len(tensors) > 1:
            ret_list = True

        tensors = tuple(mt.tensor(t) for t in tensors)
        run_tensors = []
        fetch_results = dict()

        # those executed tensors should fetch data directly, submit the others
        for t in tensors:
            if t.key in self._executed_keys:
                fetch_results[t.key] = self.fetch(t)
            else:
                run_tensors.append(t)
        if all([t.key in fetch_results for t in tensors]):
            results = [fetch_results[t.key] for t in tensors]
            return results if ret_list else results[0]

        result = self._sess.run(*run_tensors, **kw)
        self._executed_keys.update(t.key for t in run_tensors)
        for t in run_tensors:
            t._execute_session = self

        ret = []
        for r, t in zip(result, tensors):
            if r is None:
                ret.append(r)
                continue
            if t.isscalar() and hasattr(r, 'item'):
                ret.append(np.asscalar(r))
            else:
                ret.append(r)

        results = []
        result_iter = iter(ret)
        for k in [t.key for t in tensors]:
            if k in fetch_results:
                results.append(fetch_results[k])
            else:
                results.append(next(result_iter))
        if ret_list:
            return results
        return results[0]

    def fetch(self, tensor):
        if tensor.key not in self._executed_keys:
            raise ValueError('Cannot fetch the unexecuted tensor')
        return self._sess.fetch(tensor)

    @property
    def endpoint(self):
        return self._sess.endpoint

    @endpoint.setter
    def endpoint(self, endpoint):
        self._sess.endpoint = endpoint

    def decref(self, *keys):
        self._executed_keys = self._executed_keys.difference(keys)
        if hasattr(self._sess, 'decref'):
            self._sess.decref(*keys)

    def __getattr__(self, attr):
        try:
            obj = self._sess.__getattribute__(attr)
            return obj
        except AttributeError:
            raise

    def __enter__(self):
        self._sess.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self.__exit__(None, None, None)

    def as_default(self):
        Session._default_session = self
        return self

    @classmethod
    def default_or_local(cls):
        if cls._default_session is not None:
            return cls._default_session

        cls._default_session = Session()
        return cls._default_session


def new_session(scheduler=None):
    return Session(scheduler)
