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
from collections import namedtuple
from functools import lru_cache
from typing import Dict

import ray

from ..graph import DAG
from ..operands import Fetch
from ..tiles import get_tiled
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
    try:
        ray.register_custom_serializer(
            type(op), serializer=operand_serializer,
            deserializer=operand_deserializer)
    except AttributeError:  # ray >= 1.0
        from ray.worker import global_worker

        global_worker.check_connected()
        context = global_worker.get_serialization_context()
        context.register_custom_serializer(
            type(op), serializer=operand_serializer,
            deserializer=operand_deserializer)


class GraphExecutionForRay(GraphExecution):
    def handle_op(self, *args, **kw):
        return RayExecutor.handle(*args, **kw)


ChunkMeta = namedtuple('ChunkMeta', ['shape', 'object_id'])


class RayStorage:
    """
    `RayStorage` is a dict-like class. When executed in local, Mars executor will store chunk result in a
    dict(chunk_key -> chunk_result), here uses Ray actor to store them as remote objects.
    """

    @ray.remote
    class RemoteMetaStore:
        def __init__(self):
            self._store = dict()

        def set_meta(self, chunk_key, meta):
            self._store[chunk_key] = meta

        def get_meta(self, chunk_key):
            return self._store[chunk_key]

        def get_shape(self, chunk_key):
            return self._store[chunk_key].shape

        def chunk_keys(self):
            return list(self._store.keys())

        def delete_keys(self, keys):
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            for k in keys:
                del self._store[k]

    def __init__(self, meta_store=None):
        self.meta_store = meta_store or RayStorage.RemoteMetaStore.remote()

    def __getitem__(self, item):
        meta: ChunkMeta = ray.get(self.meta_store.get_meta.remote(item))
        return ray.get(meta.object_id)

    def __setitem__(self, key, value):
        object_id = ray.put(value)
        shape = getattr(value, 'shape', None)
        meta = ChunkMeta(shape=shape, object_id=object_id)
        set_meta = self.meta_store.set_meta.remote(key, meta)
        ray.wait([object_id, set_meta])

    def copy(self):
        return RayStorage(meta_store=self.meta_store)

    def update(self, mapping: Dict):
        tasks = []
        for k, v in mapping.items():
            object_id = ray.put(v)
            tasks.append(object_id)
            shape = getattr(v, 'shape', None)
            meta = ChunkMeta(shape=shape, object_id=object_id)
            set_meta = self.meta_store.set_meta.remote(k, meta)
            tasks.append(set_meta)
        ray.wait(tasks)

    def __iter__(self):
        return iter(ray.get(self.meta_store.chunk_keys.remote()))

    def __delitem__(self, key):
        ray.wait([self.meta_store.delete_keys.remote(key)])


@ray.remote
def execute_on_ray(func, results, op_wrapper: _OperandWrapper):
    op = op_wrapper.op
    func(results, op)


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

        try:
            ray.wait([execute_on_ray.remote(runner, results, op)])
        except NotImplementedError:
            for op_cls in mapper.keys():
                if isinstance(op, op_cls):
                    mapper[type(op)] = mapper[op_cls]
                    runner = mapper[op_cls]

                    ray.wait(
                        [execute_on_ray.remote(runner, results, op)])
            raise KeyError(f'No handler found for op: {op}')

    @classmethod
    def _get_chunk_shape(cls, chunk_key, chunk_result):
        assert isinstance(chunk_result, RayStorage)
        return ray.get(chunk_result.meta_store.get_shape.remote(chunk_key))


class RaySession:
    """
    Session to submit Mars job to Ray cluster.

    If Ray is not initialized, kwargs will pass to initialize Ray.
    """
    def __init__(self, **kwargs):
        # as we cannot serialize fuse chunk for now,
        # we just disable numexpr for ray executor
        engine = kwargs.pop('engine', ['numpy', 'dataframe'])
        if not ray.is_initialized():
            ray.init(**kwargs)
        self._session_id = uuid.uuid4()
        self._executor = RayExecutor(engine=engine,
                                     storage=RayStorage())

    @property
    def session_id(self):
        return self._session_id

    @property
    def executor(self):
        return self._executor

    def fetch(self, *tileables, **kw):
        return self._executor.fetch_tileables(tileables, **kw)

    def fetch_log(self, tileables, offsets=None, sizes=None):  # pragma: no cover
        raise NotImplementedError('`fetch_log` is not implemented for ray executor')

    def run(self, *tileables, **kw):
        """
        Parallelism equals to Ray cluster CPUs.
        """
        if 'n_parallel' not in kw:  # pragma: no cover
            kw['n_parallel'] = ray.cluster_resources()['CPU']
        return self._executor.execute_tileables(tileables, **kw)

    def _update_tileable_shape(self, tileable):
        from ..optimizes.tileable_graph import tileable_optimized

        new_nsplits = self._executor.get_tileable_nsplits(tileable)
        tiled = get_tiled(tileable, mapping=tileable_optimized)
        for t in (tileable, tiled):
            t._update_shape(tuple(sum(nsplit) for nsplit in new_nsplits))
        tiled.nsplits = new_nsplits

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._executor = None
