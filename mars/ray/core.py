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
from functools import lru_cache

import ray

from ..operands import Fetch
from ..graph import DAG
from ..utils import build_fetch_chunk
from ..executor import Executor, GraphExecution


class _OperandWrapper:
    __slots__ = 'op', 'chunks'

    def __init__(self, op, chunks):
        """
        As we only serde op for Ray executors, but op only weakly reference chunks,
        So we create a wrapper here to keep the reference
        """
        self.op = op
        self.chunks = chunks


def operand_serializer(op):
    graph = DAG()
    inputs = [build_fetch_chunk(inp) for inp in op.inputs or []]
    new_op = op.copy()

    kws = []
    for c in op.outputs:
        params = c.params.copy()
        params['_key'] = c.key
        params.update(c.extra_params)
        kws.append(params)

    chunks = new_op.new_chunks(inputs, kws=kws, output_limit=len(kws))
    for obj in chunks + inputs:
        graph.add_node(obj)

    return graph.to_json()


def operand_deserializer(value):
    graph = DAG.from_json(value)
    if len(graph) == 1:
        chunks = [list(graph)[0]]
    else:
        chunks = [c for c in graph if not isinstance(c.op, Fetch)]
    op = chunks[0].op
    return _OperandWrapper(op, chunks)


@lru_cache(500)
def _register_ray_serializer(op):
    # register a custom serializer for Mars operand
    ray.register_custom_serializer(
        type(op), serializer=operand_serializer,
        deserializer=operand_deserializer)


class GraphExecutionForRay(GraphExecution):
    def handle_op(self, *args, **kw):
        return RayExecutor.handle(*args, **kw)


class RayStorage:
    """
    `RayStorage` is a dict-like class. When executed in local, Mars executor will store chunk result in a
    dict(chunk_key -> chunk_result), here uses Ray actor to store them as remote objects.
    """

    @ray.remote
    class RemoteDict:
        def __init__(self):
            self._dict = dict()

        def keys(self):
            return list(self._dict.keys())

        def setitem(self, key, value):
            self._dict[key] = value

        def getitem(self, item):
            return self._dict[item]

        def update(self, mapping):
            self._dict.update(mapping)

        def delitem(self, key):
            del self._dict[key]

    def __init__(self, ray_dict_ref=None):
        self.ray_dict_ref = ray_dict_ref or RayStorage.RemoteDict.remote()

    def __getitem__(self, item):
        return ray.get(self.ray_dict_ref.getitem.remote(item))

    def __setitem__(self, key, value):
        ray.get(self.ray_dict_ref.setitem.remote(key, value))

    def copy(self):
        return RayStorage(ray_dict_ref=self.ray_dict_ref)

    def update(self, mapping):
        ray.get(self.ray_dict_ref.update.remote(mapping))

    def __iter__(self):
        return iter(ray.get(self.ray_dict_ref.keys.remote()))

    def __delitem__(self, key):
        ray.get(self.ray_dict_ref.delitem.remote(key))


class RayExecutor(Executor):
    """
    Wraps the execute function as a Ray remote function, the type of `results` is `RayStorage`,
    when operand is executed, it will fetch dependencies from a Ray actor.
    """

    _graph_execution_cls = GraphExecutionForRay

    @classmethod
    def handle(cls, op, results, mock=False):
        method_name, mapper = ('execute', cls._op_runners) if not mock else \
            ('estimate_size', cls._op_size_estimators)
        try:
            runner = mapper[type(op)]
        except KeyError:
            runner = getattr(op, method_name)

        # register a custom serializer for Mars operand
        _register_ray_serializer(op)

        @lru_cache(500)
        def build_remote_funtion(func):

            @ray.remote
            def remote_runner(results, op_wrapper: _OperandWrapper):
                op = op_wrapper.op
                return func(results, op)

            return remote_runner

        try:
            return ray.get(build_remote_funtion(runner).remote(results, op))
        except NotImplementedError:
            for op_cls in mapper.keys():
                if isinstance(op, op_cls):
                    mapper[type(op)] = mapper[op_cls]
                    runner = mapper[op_cls]

                    return ray.get(
                        build_remote_funtion(runner).remote(results, op))
            raise KeyError(f'No handler found for op: {op}')


class RaySession:
    """
    Session to submit Mars job to Ray cluster.

    If Ray is not initialized, kwargs will pass to initialize Ray.
    """
    def __init__(self, **kwargs):
        if not ray.is_initialized():
            ray.init(**kwargs)
        self._session_id = uuid.uuid4()
        self._executor = RayExecutor(storage=RayStorage())

    @property
    def session_id(self):
        return self._session_id

    @property
    def executor(self):
        return self._executor

    def fetch(self, *tileables, **kw):
        return self._executor.fetch_tileables(tileables, **kw)

    def run(self, *tileables, **kw):
        """
        Parallelism equals to Ray cluster CPUs.
        """
        if 'n_parallel' not in kw:  # pragma: no cover
            kw['n_parallel'] = ray.cluster_resources()['CPU']
        return self._executor.execute_tileables(tileables, **kw)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._executor = None
