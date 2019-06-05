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
import itertools
import functools
import threading
import weakref
from collections import deque, defaultdict

import numpy as np

from ...tensor.expressions.datasource import TensorFetchChunk
from ...graph import DirectedGraph
from ...compat import futures, OrderedDict
from ..core import build_mode
from ..expressions.merge.concatenate import TensorConcatenate


class Executor(object):
    _op_runners = {}
    _op_size_estimators = {}

    def __init__(self, engine=None, storage=None, prefetch=False):
        self._engine = engine
        self._chunk_result = storage if storage is not None else dict()
        self._prefetch = prefetch

        # only record the executed tensor
        self.tensor_to_tiled = weakref.WeakKeyDictionary()
        self.tensor_key_to_chunk_keys = dict()
        # executed key to ref counts
        self.key_to_ref_counts = defaultdict(lambda: 0)

        self._mock_max_memory = 0

    @property
    def chunk_result(self):
        return self._chunk_result

    @property
    def mock_max_memory(self):
        return self._mock_max_memory

    def _preprocess(self, graph, keys):
        from .optimizes.core import Optimizer

        Optimizer(graph, self._engine).optimize(keys=keys)
        return graph

    @staticmethod
    def _get_op_runner(chunk, mapper):
        try:
            op_cls = type(chunk.op)
            return mapper[op_cls]
        except KeyError:
            for op_cls in mapper.keys():
                if isinstance(chunk.op, op_cls):
                    mapper[type(chunk.op)] = mapper[op_cls]
                    return mapper[op_cls]

            raise KeyError('No handler found for op: %s' % chunk.op)

    @classmethod
    def handle(cls, chunk, results, mock=False):
        if not mock:
            return cls._get_op_runner(chunk, cls._op_runners)(results, chunk)
        else:
            return cls._get_op_runner(chunk, cls._op_size_estimators)(results, chunk)

    def execute_graph(self, graph, keys, n_parallel=None, n_thread=None, show_progress=False,
                      compose=True, mock=False, no_intermediate=False):
        """
        :param graph: graph to execute
        :param keys: result keys
        :param n_parallel: num of max parallelism
        :param n_thread: num of threads to execute
        :param show_progress:
        :param compose: if True. fuse nodes when possible
        :param mock: if True, only estimate data sizes without execution
        :param no_intermediate: exclude intermediate data sizes when estimating memory size
        :return: execution result
        """
        with build_mode():
            optimized_graph = self._preprocess(graph, keys) if compose else graph

        res = execute_graph(optimized_graph, keys, self, n_parallel=n_parallel or n_thread,
                            show_progress=show_progress, prefetch=self._prefetch, retval=True,
                            mock=mock, no_intermediate=no_intermediate)
        if mock:
            self._mock_max_memory = max(self._mock_max_memory, self._chunk_result.get('_mock_max_memory', 0))
            self._chunk_result.clear()
        return res

    def execute_tensor(self, tensor, n_parallel=None, n_thread=None, concat=False,
                       show_progress=False, mock=False):
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
                                  show_progress=show_progress, mock=mock)

    def execute_tensors(self, tensors, fetch=True, n_parallel=None, n_thread=None,
                        show_progress=False, mock=False, compose=True):
        graph = DirectedGraph()

        result_keys = []
        to_release_keys = []
        concat_keys = []
        for tensor in tensors:
            tensor.tiles()
            chunk_keys = [c.key for c in tensor.chunks]
            self.tensor_key_to_chunk_keys[(tensor.key, tensor.id)] = chunk_keys
            result_keys.extend(chunk_keys)

            if not fetch:
                # no need to generate concat keys
                pass
            elif len(tensor.chunks) > 1:
                # if need to fetch data and chunks more than 1, we concatenate them into 1
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

            tensor.build_graph(graph=graph, tiled=True, compose=compose)

        self.execute_graph(graph, result_keys, n_parallel=n_parallel or n_thread,
                           show_progress=show_progress, mock=mock)

        results = self._chunk_result
        try:
            if fetch:
                return [results[k] for k in concat_keys]
            else:
                return
        finally:
            for k in to_release_keys:
                del results[k]

    def fetch_tensors(self, tensors, **kw):
        from ..expressions.datasource import TensorFetchChunk

        results = []
        to_concat_tensors = OrderedDict()

        for i, tensor in enumerate(tensors):
            if (tensor.key, tensor.id) not in self.tensor_key_to_chunk_keys:
                # check if the tensor is executed before
                raise ValueError(
                    'Tensor to fetch must be executed before, got {0}'.format(tensor))

            if len(tensor.chunks) == 1:
                result = self._chunk_result[tensor.chunks[0].key]
                results.append(result)
                continue

            # generate TensorFetchChunk op for each chunk
            chunks = []
            for c in tensor.chunks:
                op = TensorFetchChunk(dtype=c.dtype, to_fetch_key=c.key, sparse=c.op.sparse)
                chunk = op.new_chunk(None, c.shape, index=c.index, _key=c.key)
                chunks.append(chunk)

            new_op = tensor.op.copy()
            tensor = new_op.new_tensor([None], tensor.shape, chunks=chunks,
                                       nsplits=tensor.nsplits)

            # add this concat tensor into the list which shall be executed later
            to_concat_tensors[i] = tensor
            results.append(None)

        # execute the concat tensors together
        if to_concat_tensors:
            concat_results = self.execute_tensors(list(to_concat_tensors.values()), **kw)
            for j, concat_result in zip(to_concat_tensors, concat_results):
                results[j] = concat_result

        return results

    def get_tensor_nsplits(self, tensor):
        chunk_indexes = [c.index for c in tensor.chunks]
        chunk_shapes = [r.shape for r in [self._chunk_result[c.key] for c in tensor.chunks]]

        ndim = len(chunk_shapes[0])
        tensor_nsplits = []
        for i in range(ndim):
            splits = []
            for index, shape in zip(chunk_indexes, chunk_shapes):
                if all(idx == 0 for j, idx in enumerate(index) if j != i):
                    splits.append(shape[i])
            tensor_nsplits.append(tuple(splits))

        return tuple(tensor_nsplits)

    def decref(self, *keys):
        for key in keys:
            for chunk_key in self.tensor_key_to_chunk_keys.get(key, []):
                if chunk_key in self.chunk_result:
                    del self.chunk_result[chunk_key]


def execute_chunk(chunk, executor=None,
                  ref_counts=None, chunk_result=None,
                  finishes=None, visited=None, q=None,
                  lock=None, semaphore=None, has_error=None,
                  preds=None, succs=None, fetch_keys=None,
                  mock=False, no_intermediate=False):
    try:
        with lock:
            if (chunk.key, chunk.id) in visited:
                return
            visited.add((chunk.key, chunk.id))
            finished = finishes.get(chunk.key)
        if not finished:
            # do real execution
            if chunk.key not in chunk_result:
                executor.handle(chunk, chunk_result, mock)

                # update maximal memory usage during execution
                if mock:
                    # we ignore sizes of Fetch inputs as they are not part of memory needed
                    fetch_keys = fetch_keys or set()
                    output_keys = set(o.key for o in chunk.op.outputs or ())

                    cur_memory = sum(chunk_result[op_output.key][1] for op_output in chunk.op.outputs
                                     if chunk_result.get(op_output.key) is not None)
                    if not no_intermediate:
                        cur_memory += sum(tp[0] for key, tp in chunk_result.items()
                                          if key not in fetch_keys and key not in output_keys
                                          and isinstance(tp, tuple))
                    chunk_result['_mock_max_memory'] = max(cur_memory, chunk_result.get('_mock_max_memory', 0))

            with lock:
                for output in chunk.op.outputs:
                    finishes[output.key] = True
                    if output.key in ref_counts and ref_counts[output.key] == 0 and \
                            output.key in chunk_result:
                        # some op have more than 1 outputs,
                        # and some of the outputs are not in the result ones
                        del chunk_result[output.key]

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
                  mock=False, no_intermediate=False, prefetch=False, retval=True):
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

    if not mock:
        # fetch_keys only useful when calculating sizes
        fetch_keys = set()
    else:
        fetch_keys = set(v.key for v in graph if isinstance(v.op, TensorFetchChunk))
        for c in graph:
            if graph.count_predecessors(c) != 0:
                continue
            fetch_keys.update(inp.key for inp in c.inputs or ())

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
                print('[{0}] {1:.2f}% percent of graph has been submitted'.format(
                    str(datetime.datetime.now()), float(c) * 100 / len(graph)))

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
                                      preds=preds, succs=succs, fetch_keys=fetch_keys,
                                      mock=mock, no_intermediate=no_intermediate)
        fs[chunk.key] = future

    while len(node_keys_set - set(finishes.keys())) > 0:
        if has_error.is_set():
            break
        semaphore.acquire()
        submit_to_execute()

    [f.result() for f in fs.values()]
    if retval:
        return [chunk_result[key] for key in keys]


def default_size_estimator(ctx, chunk, multiplier=1):
    exec_size = 0
    outputs = chunk.op.outputs

    if all(not c.is_sparse() and not np.isnan(c.nbytes) for c in outputs):
        for c in outputs:
            ctx[c.key] = (c.nbytes, c.nbytes * multiplier)
        return

    for inp in chunk.inputs or ():
        if chunk.is_sparse() or np.isnan(inp.nbytes):
            exec_size += ctx[inp.key][0]
        else:
            exec_size += inp.nbytes
    exec_size = int(exec_size * multiplier)

    total_out_size = 0
    chunk_sizes = dict()
    for out in outputs:
        try:
            chunk_size = out.nbytes if not out.is_sparse() else exec_size
            if np.isnan(chunk_size):
                raise TypeError
            chunk_sizes[out.key] = chunk_size
            total_out_size += chunk_size
        except (AttributeError, TypeError, ValueError):
            pass
    exec_size = max(exec_size, total_out_size)
    for out in outputs:
        if out.key in ctx:
            continue
        if out.key in chunk_sizes:
            store_size = chunk_sizes[out.key]
        else:
            store_size = max(exec_size // len(outputs),
                             total_out_size // max(len(chunk_sizes), 1))
        try:
            max_sparse_size = out.nbytes + np.dtype(np.int64).itemsize * np.prod(out.shape) * out.ndim
        except TypeError:  # pragma: no cover
            max_sparse_size = np.nan
        if not np.isnan(max_sparse_size):
            store_size = min(store_size, max_sparse_size)
        ctx[out.key] = (store_size, exec_size // len(outputs))


def size_estimator_wrapper(ctx, chunk, original_estimator=None):
    try:
        return original_estimator(ctx, chunk)
    except NotImplementedError:
        return default_size_estimator(ctx, chunk)


def ignore(*_):
    pass
Executor._op_runners[TensorFetchChunk] = ignore


def register(op, handler, size_estimator=None, size_multiplier=1):
    Executor._op_runners[op] = handler
    if size_estimator:
        Executor._op_size_estimators[op] = \
            functools.partial(size_estimator_wrapper, original_estimator=size_estimator)
    else:
        Executor._op_size_estimators[op] = size_estimator or \
            functools.partial(default_size_estimator, multiplier=size_multiplier)


from .datasource import register_data_source_handler
from .datastore import register_data_store_handler
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
    import numexpr  # noqa: F401
    NUMEXPR_INSTALLED = True
    from .ne import register_numexpr_handler
    register_numexpr_handler()
except ImportError:  # pragma: no cover
    pass

CP_INSTALLED = False
try:
    import cupy  # noqa: F401
    CP_INSTALLED = True
    from .cp import register_cp_handler
    register_cp_handler()
except ImportError:  # pragma: no cover
    pass

register_data_source_handler()
register_data_store_handler()
register_random_handler()
register_basic_handler()
register_arithmetic_handler()
register_indexing_handler()
register_reduction_handler()
register_merge_handler()
register_fft_handler()
register_linalg_handler()
