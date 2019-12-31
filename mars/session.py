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

import numpy as np

from .core import Entity, Base
from .context import LocalContext
from .tiles import get_tiled
from .executor import Executor
try:
    from .resource import cpu_count, cuda_count
except ImportError:  # pragma: no cover
    from multiprocessing import cpu_count
    cuda_count = None


class LocalSession(object):
    def __init__(self, **kwargs):
        self._endpoint = None
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

    @property
    def session_id(self):
        return None

    @property
    def executor(self):
        return self._executor

    @property
    def context(self):
        return self._context

    @property
    def executed_tileables(self):
        return self._executor.stored_tileables.keys()

    @endpoint.setter
    def endpoint(self, endpoint):
        if endpoint is not None:
            raise ValueError('Local session cannot set endpoint')
        self._endpoint = endpoint

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

    def fetch(self, *tileables, **kw):
        if self._executor is None:
            raise RuntimeError('Session has closed')
        if 'n_parallel' not in kw:
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
            if getattr(t, 'shape', None) is not None and np.nan in t.shape:
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
        Session._default_session = self
        return self

    @classmethod
    def default_or_local(cls):
        if cls._default_session is not None:
            return cls._default_session

        cls._default_session = Session()
        return cls._default_session

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
