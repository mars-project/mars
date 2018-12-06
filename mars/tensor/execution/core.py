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

import datetime
import threading
import weakref
import itertools
from collections import deque, defaultdict

import numpy as np

from ...operands import Fetch
from ...graph import DirectedGraph
from ...compat import futures
from ..core import build_mode
from ..expressions.merge.concatenate import TensorConcatenate


class Executor(object):
    _op_runners = {}

    def __init__(self, engine=None, storage=None, prefetch=False):
        self._engine = engine
        self._chunk_result = storage if storage is not None else dict()
        self._prefetch = prefetch

        # only record the executed tensor
        self.tensor_to_tiled = weakref.WeakKeyDictionary()
        self.tensor_key_to_chunk_keys = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)

    @property
    def chunk_result(self):
        return self._chunk_result

    def _preprocess(self, graph, keys):
        from .optimizes.core import Optimizer

        Optimizer(graph, self._engine).optimize(keys=keys)
        return graph

    def handle(self, chunk, results):
        try:
            cls = type(chunk.op)

            return self._op_runners[cls](results, chunk)
        except KeyError:
            for cls in self._op_runners.keys():
                if isinstance(chunk.op, cls):
                    self._op_runners[type(chunk.op)] = self._op_runners[cls]
                    return self._op_runners[cls](results, chunk)

            raise KeyError('No handler found for op: %s' % chunk.op)

    def execute_graph(self, graph, keys, n_parallel=None, n_thread=None, show_progress=False,
                      mock=False, sparse_mock_percent=1.0):
        with build_mode():
            optimized_graph = self._preprocess(graph, keys)

        return execute_graph(optimized_graph, keys, self, n_parallel=n_parallel or n_thread,
                             show_progress=show_progress, mock=mock,
                             sparse_mock_percent=sparse_mock_percent,
                             prefetch=self._prefetch, retval=True)

    def execute_tensor(self, tensor, n_parallel=None, n_thread=None, concat=False,
                       show_progress=False, mock=False, sparse_mock_percent=1.0):
        if concat:
            # only for tests
            tensor.tiles()
            if len(tensor.chunks) > 1:
                op = TensorConcatenate(dtype=tensor.op.dtype)
                chunk = TensorConcatenate(dtype=op.dtype).new_chunk(tensor.chunks, tensor.shape)
                tensor = op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                                       nsplits=[(s,) for s in tensor.shape])

        graph = tensor.build_graph(cls=DirectedGraph, tiled=True)

        return self.execute_graph(graph, [c.key for c in tensor.chunks],
                                  n_parallel=n_parallel or n_thread,
                                  show_progress=show_progress, mock=mock,
                                  sparse_mock_percent=sparse_mock_percent)

    def execute_tensors(self, tensors, n_parallel=None, n_thread=None,
                        show_progress=False, mock=False, sparse_mock_percent=1.0):
        graph = DirectedGraph()

        result_keys = []
        to_release_keys = []
        concat_keys = []
        for tensor in tensors:
            tensor.tiles()
            chunk_keys = [c.key for c in tensor.chunks]
            self.tensor_key_to_chunk_keys[(tensor.key, tensor.id)] = chunk_keys
            result_keys.extend(chunk_keys)

            if len(tensor.chunks) > 1:
                # if chunks more than 1, we concatenate them into 1
                op = TensorConcatenate(dtype=tensor.op.dtype)
                chunk = TensorConcatenate(dtype=op.dtype).new_chunk(tensor.chunks, tensor.shape)
                result_keys.append(chunk.key)
                # the concatenated key
                concat_keys.append(chunk.key)
                # after return the data to user, we release the reference
                to_release_keys.append(chunk.key)
                tensor = op.new_tensor([tensor], tensor.shape, chunks=[chunk],
                                       nsplits=[(s,) for s in tensor.shape])
            else:
                concat_keys.append(tensor.chunks[0].key)

            tensor.build_graph(graph=graph, tiled=True)

        self.execute_graph(graph, result_keys, n_parallel=n_parallel or n_thread,
                           show_progress=show_progress, mock=mock,
                           sparse_mock_percent=sparse_mock_percent)

        results = self._chunk_result
        try:
            return [results[k] for k in concat_keys]
        finally:
            for k in to_release_keys:
                del results[k]

    def decref(self, *keys):
        for key in keys:
            for chunk_key in self.tensor_key_to_chunk_keys.get(key, []):
                if chunk_key in self.chunk_result:
                    del self.chunk_result[chunk_key]


def execute_chunk(chunk, executor=None,
                  ref_counts=None, chunk_result=None,
                  finishes=None, visited=None, q=None,
                  lock=None, semaphore=None, has_error=None,
                  preds=None, succs=None, mock=False, sparse_mock_percent=1.0):
    try:
        with lock:
            if (chunk.key, chunk.id) in visited:
                return
            visited.add((chunk.key, chunk.id))
            finished = finishes.get(chunk.key)
        if not finished:
            if not mock:
                # do real execution
                if chunk.key not in chunk_result:
                    executor.handle(chunk, chunk_result)
            else:
                percent = sparse_mock_percent if chunk.op.sparse else 1.0
                # we put the estimated size of data into the chunk_result
                chunk_result[chunk.key] = np.prod(chunk.shape) * chunk.dtype.itemsize * percent
            with lock:
                for output in chunk.op.outputs:
                    finishes[output.key] = True

        for pred_key in preds[chunk.key]:
            with lock:
                if pred_key not in ref_counts:
                    continue
                ref_counts[pred_key] -= 1
                if ref_counts[pred_key] == 0:
                    del chunk_result[pred_key]

        for succ in succs[chunk.key, chunk.id]:
            with lock:
                if (succ.key, succ.id) in visited:
                    continue
                if len(preds[succ.key]) == 0 or \
                        all(finishes.get(k, False) for k in preds[succ.key]):
                    q.insert(0, succ)
    except Exception:
        has_error.set()
        raise
    finally:
        semaphore.release()


def _order_starts(graph):
    visited = set()
    starts = deque(graph.iter_indep())
    stack = deque([starts.popleft()])

    while stack:
        node = stack.popleft()
        if node not in visited:
            preds = graph.predecessors(node)
            if not preds or all(pred in visited for pred in preds):
                if len(preds) == 0:
                    yield node
                visited.add(node)
                stack.extend(n for n in graph[node] if n not in visited)
            else:
                stack.appendleft(node)
                stack.extendleft(reversed(list(n for n in graph.predecessors(node)
                                               if n not in visited)))
        if not stack and starts:
            stack.appendleft(starts.popleft())


def execute_graph(graph, keys, executor, n_parallel=None, show_progress=False,
                  mock=False, sparse_mock_percent=1.0, prefetch=False, retval=True):
    pool_executor = futures.ThreadPoolExecutor(n_parallel or 1)
    prefetch_executor = futures.ThreadPoolExecutor(n_parallel or 1) if prefetch else None
    chunk_result = executor.chunk_result
    q = list()
    lock = threading.Lock()
    semaphore = threading.Semaphore(pool_executor._max_workers)
    has_error = threading.Event()

    preds = dict()
    preds.update(
        dict((t.key, [i.key for i in v]) for t, v in graph.iter_predecessor_items())
    )
    succs = dict()
    succs.update(
        dict(((t.key, t.id), v) for t, v in graph.iter_successor_items())
    )

    starts = list(_order_starts(graph)) if len(graph) > 0 else list()
    assert len(starts) == sum(1 for _ in graph.iter_indep())
    q.extend(starts)
    finishes = dict()
    visited = set()
    fs = dict()
    ref_counts = dict()
    key_set = set(keys)
    for chunk in graph:
        if chunk.key not in key_set:
            ref_counts[chunk.key] = ref_counts.get(chunk.key, 0) + len(graph[chunk])
    node_keys_set = {n.key for n in graph}
    count = itertools.count(0)
    maximum_usage = 0

    def fetch(chunk):
        with lock:
            try:
                to_fetch_chunk = next(c for c in list(succs.get((chunk.key, chunk.id), []))
                                      if all(i in finishes for i in preds[c.key] if i != chunk.key))
            except StopIteration:
                if len(q) > 0:
                    to_fetch_chunk = q[0]
                else:
                    return

        [chunk_result.get(k) for k in preds[to_fetch_chunk.key]]

    def submit_to_execute():
        if show_progress:
            c = next(count)
            if c % 30 == 0 or c >= len(graph):
                print('[{0}] {1:.2f}% percent of graph has been submitted'.format(str(datetime.datetime.now()), float(c) * 100 / len(graph)))

        with lock:
            if len(q) == 0:
                semaphore.release()
                return
            chunk = q.pop(0)

        if prefetch:
            prefetch_executor.submit(fetch, chunk)
        future = pool_executor.submit(execute_chunk, chunk, executor=executor,
                                      ref_counts=ref_counts, chunk_result=chunk_result,
                                      finishes=finishes, visited=visited, q=q,
                                      lock=lock, semaphore=semaphore, has_error=has_error,
                                      preds=preds, succs=succs,
                                      mock=mock, sparse_mock_percent=sparse_mock_percent)
        fs[chunk.key] = future

    while len(node_keys_set - set(finishes.keys())) > 0:
        if has_error.is_set():
            break
        semaphore.acquire()
        if mock:
            maximum_usage = max(maximum_usage, np.sum(list(chunk_result.values())))
            # if maximum_usage > 40 * (1024 ** 3):
            #     raise RuntimeError('memory exceed')
        submit_to_execute()

    if mock:
        [f.result() for f in fs.values()]
        return maximum_usage

    [f.result() for f in fs.values()]
    if retval:
        return [chunk_result[key] for key in keys]


def ignore(*_):
    pass
Executor._op_runners[Fetch] = ignore


def register(op, handler):
    Executor._op_runners[op] = handler


from .datasource import register_data_source_handler
from .random import register_random_handler
from .base import register_basic_handler
from .arithmetic import register_arithmetic_handler
from .indexing import register_indexing_handler
from .reduction import register_reduction_handler
from .merge import register_merge_handler
from .fft import register_fft_handler
from .linalg import register_linalg_handler

NUMEXPR_INSTALLED = False
try:
    import numexpr
    NUMEXPR_INSTALLED = True
    from .ne import register_numexpr_handler
    register_numexpr_handler()
except ImportError:
    pass

CP_INSTALLED = False
try:
    import cupy
    CP_INSTALLED = True
    from .cp import register_cp_handler
    register_cp_handler()
except ImportError:
    pass

register_data_source_handler()
register_random_handler()
register_basic_handler()
register_arithmetic_handler()
register_indexing_handler()
register_reduction_handler()
register_merge_handler()
register_fft_handler()
register_linalg_handler()
