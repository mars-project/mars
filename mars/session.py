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

from .core import Entity, Base
try:
    from .resource import cpu_count
except ImportError:  # pragma: no cover
    from multiprocessing import cpu_count


class LocalSession(object):
    def __init__(self, **kwargs):
        from .executor import Executor

        self._executor = Executor()
        self._endpoint = None

        if kwargs:
            unexpected_keys = ', '.join(list(kwargs.keys()))
            raise TypeError('Local session got unexpected arguments: %s' % unexpected_keys)

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def executed_tileables(self):
        return self._executor.stored_tileables.keys()

    @endpoint.setter
    def endpoint(self, endpoint):
        if endpoint is not None:
            raise ValueError('Local session cannot set endpoint')
        self._endpoint = endpoint

    def run(self, *tileables, **kw):
        if self._executor is None:
            raise RuntimeError('Session has closed')
        if 'n_parallel' not in kw:
            kw['n_parallel'] = cpu_count()
        res = self._executor.execute_tileables(tileables, **kw)
        return res

    def _update_tileable_shape(self, tileable):
        new_nsplits = self._executor.get_tileable_nsplits(tileable)
        tileable._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tileable.nsplits = new_nsplits

    def fetch(self, *tileables, **kw):
        for t in tileables:
            if t.key not in self.executed_tileables:
                raise ValueError('Cannot fetch the unexecuted tileable')
        if self._executor is None:
            raise RuntimeError('Session has closed')
        if 'n_parallel' not in kw:
            kw['n_parallel'] = cpu_count()
        return self._executor.fetch_tileables(tileables, **kw)

    def decref(self, *keys):
        self._executor.decref(*keys)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._executor = None


class Session(object):
    _default_session = None

    def __init__(self, endpoint=None, **kwargs):
        if endpoint is not None:
            if 'http' in endpoint:
                # connect to web
                from .web.session import Session as WebSession

                self._sess = WebSession(endpoint, **kwargs)
            else:
                # connect to local cluster
                from .deploy.local.session import LocalClusterSession

                self._sess = LocalClusterSession(endpoint, **kwargs)
        else:
            self._sess = LocalSession(**kwargs)

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
            t._execute_session = self

        for t in tileables:
            if np.nan in t.shape:
                self._sess._update_tileable_shape(t)

        if fetch:
            ret = []
            for r, t in zip(result, tileables):
                if hasattr(t, 'isscalar') and t.isscalar() and hasattr(r, 'item'):
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
        Session._default_session = self
        return self

    @classmethod
    def default_or_local(cls):
        if cls._default_session is not None:
            return cls._default_session

        cls._default_session = Session()
        return cls._default_session

    def create_mutable_tensor(self, name, shape, dtype, *args, **kwargs):
        from .tensor.core import MutableTensor, MutableTensorData
        from .tensor.expressions.utils import create_fetch_tensor
        self._ensure_local_cluster()
        shape, dtype, chunk_size, chunk_keys = \
                self._sess.create_mutable_tensor(name, shape, dtype, *args, **kwargs)
        # Construct MutableTensor on the fly.
        tensor = create_fetch_tensor(chunk_size, shape, dtype, chunk_keys=chunk_keys)
        return MutableTensor(data=MutableTensorData(_name=name, _op=None, _shape=shape, _dtype=dtype,
                                                    _nsplits=tensor.nsplits, _chunks=tensor.chunks))

    def get_mutable_tensor(self, name):
        from .tensor.core import MutableTensor, MutableTensorData
        from .tensor.expressions.utils import create_fetch_tensor
        self._ensure_local_cluster()
        shape, dtype, chunk_size, chunk_keys = self._sess.get_mutable_tensor(name)
        # Construct MutableTensor on the fly.
        tensor = create_fetch_tensor(chunk_size, shape, dtype, chunk_keys=chunk_keys)
        return MutableTensor(data=MutableTensorData(_name=name, _op=None, _shape=shape, _dtype=dtype,
                                                    _nsplits=tensor.nsplits, _chunks=tensor.chunks))

    def write_mutable_tensor(self, tensor, index, value):
        self._ensure_local_cluster()
        chunk_records_to_send = tensor._do_write(index, value)
        return self._sess.send_chunk_records(tensor.name, chunk_records_to_send)

    def seal(self, tensor):
        from .tensor.expressions.utils import create_fetch_tensor
        self._ensure_local_cluster()
        chunk_records_to_send = tensor._do_flush()
        self._sess.send_chunk_records(tensor.name, chunk_records_to_send)
        shape, dtype, chunk_size, chunk_keys = self._sess.seal(tensor.name)
        # Construct Tensor on the fly.
        return create_fetch_tensor(chunk_size, shape, dtype, chunk_keys=chunk_keys)

    def _ensure_local_cluster(self):
        from .deploy.local.session import LocalClusterSession
        if not isinstance(self._sess, LocalClusterSession):
            raise RuntimeError("Only local cluster session can be used to manipulate mutable tensors.")

def new_session(scheduler=None, **kwargs):
    return Session(scheduler, **kwargs)
